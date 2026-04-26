from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

from tqdm import tqdm
from dotenv import load_dotenv

from reasoning_agent_optimized import build_agent

load_dotenv()

INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")
CHECKPOINT_PATH = Path("agent_checkpoint.json")

def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data

def validate_results(questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]) -> None:
    if len(questions) != len(answers):
        raise ValueError(f"Mismatched lengths: {len(questions)} vs {len(answers)}")
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(f"Answer at index {idx} has non-string output.")
        if len(answer["output"]) >= 5000:
            raise ValueError(f"Answer at index {idx} exceeds 5000 chars.")

def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def save_checkpoint(path: Path, next_index: int, answers: List[Dict[str, str]]) -> None:
    payload = {"next_index": next_index, "answers": answers}
    write_json(path, payload)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final project agent (parallel).")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--dev-data", default=None)
    parser.add_argument("--backend", choices=["api", "nearest"], default="api")
    parser.add_argument("--max-calls-per-question", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-at", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--skip-validation", action="store_true")
    
    ##for cpu greeter utilization
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent threads (default 10). Increase if API allows.")
    parser.add_argument("--timeout", type=int, default=120,
                        help="API timeout per request (seconds).")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    questions = load_questions(args.input)
    total_questions = len(questions)

    checkpoint = load_checkpoint(CHECKPOINT_PATH) if args.resume else {}
    answers = checkpoint.get("answers", [])
    start_index = checkpoint.get("next_index", args.start_at)

    if not answers and start_index > 0:
        answers = [{"output": ""} for _ in range(start_index)]

    end_index = total_questions
    if args.limit is not None:
        end_index = min(end_index, start_index + args.limit)

    tasks = [(idx, questions[idx]["input"]) for idx in range(start_index, end_index)]

    results = [None] * total_questions
    # Copy existing answers from checkpoint
    for i, ans in enumerate(answers):
        results[i] = ans["output"]

    total_calls = 0
    domain_counts: Dict[str, int] = {}
    stats_lock = Lock()
    checkpoint_lock = Lock()
    last_checkpoint_idx = start_index

    pbar = tqdm(total=len(tasks), desc="Solving questions", unit="q")

    def solve_one(idx: int, question: str) -> tuple[int, str, int, str]:
        """Worker function: solves a single question and returns (idx, answer, calls, domain)."""
        agent = build_agent(
            dev_data_path=args.dev_data,
            backend=args.backend,
            max_calls_per_question=args.max_calls_per_question,
        )
        if hasattr(agent.model_client, "timeout"):
            agent.model_client.timeout = args.timeout

        result = agent.solve(question)
        return idx, result.answer, result.calls_used, result.domain

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(solve_one, idx, q): idx for idx, q in tasks}
        for future in as_completed(futures):
            idx, answer, calls, domain = future.result()
            results[idx] = answer
            with stats_lock:
                total_calls += calls
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            print(f"\n[{idx+1}/{total_questions}] domain={domain:<17} calls={calls:<2} answer={answer[:80]!r}")
            pbar.update(1)

            solved_count = idx - start_index + 1
            if args.checkpoint_every > 0 and solved_count % args.checkpoint_every == 0:
                with checkpoint_lock:
                    if solved_count > last_checkpoint_idx - start_index:
                        contiguous_end = start_index
                        while contiguous_end < total_questions and results[contiguous_end] is not None:
                            contiguous_end += 1
                        answers_up_to = [{"output": results[i]} for i in range(start_index, contiguous_end)]
                        save_checkpoint(CHECKPOINT_PATH, contiguous_end, answers_up_to)
                        last_checkpoint_idx = contiguous_end

    final_answers = [{"output": results[i]} for i in range(start_index, end_index)]
    write_json(args.output, final_answers)

    if not args.skip_validation and len(final_answers) == total_questions:
        validate_results(questions, final_answers)
        print(f"Validated {len(final_answers)} answers successfully.")
    else:
        print("Skipped full validation.")

    print(f"Output written to {args.output}")
    print(f"Total model calls used in this run: {total_calls}")
    if domain_counts:
        print("Domain counts:", domain_counts)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)