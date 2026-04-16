"""
Generate answers for the CSE 476 final project submission file.

This script keeps the same filename as the provided template, but now runs a
real reasoning agent instead of writing placeholders.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from reasoning_agent import build_agent


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
        raise ValueError(
            f"Mismatched lengths: {len(questions)} questions vs {len(answers)} answers."
        )
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(
                f"Answer at index {idx} has non-string output: {type(answer['output'])}"
            )
        if len(answer["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} exceeds 5000 characters "
                f"({len(answer['output'])} chars)."
            )


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_checkpoint(path: Path, next_index: int, answers: List[Dict[str, str]]) -> None:
    payload = {
        "next_index": next_index,
        "answers": answers,
    }
    write_json(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final project reasoning agent.")
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="Question JSON file.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Answer JSON file.")
    parser.add_argument(
        "--dev-data",
        default=None,
        help="Path to cse476_final_project_dev_data.json. If omitted, the script searches common locations.",
    )
    parser.add_argument(
        "--backend",
        choices=["api", "nearest"],
        default="api",
        help="Use the real API backend or a local nearest-neighbor baseline for offline smoke tests.",
    )
    parser.add_argument(
        "--max-calls-per-question",
        type=int,
        default=6,
        help="Hard cap on model calls for one question.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only solve the first N questions from the current starting point.",
    )
    parser.add_argument(
        "--start-at",
        type=int,
        default=0,
        help="Start from this zero-based question index.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from agent_checkpoint.json if it exists.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Write a checkpoint every N solved questions.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip full-length validation, useful for partial development runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_questions(args.input)

    checkpoint = load_checkpoint(CHECKPOINT_PATH) if args.resume else {}
    answers: List[Dict[str, str]] = checkpoint.get("answers", [])
    start_index = checkpoint.get("next_index", args.start_at)

    if not answers and start_index > 0:
        answers = [{"output": ""} for _ in range(start_index)]

    agent = build_agent(
        dev_data_path=args.dev_data,
        backend=args.backend,
        max_calls_per_question=args.max_calls_per_question,
    )

    end_index = len(questions)
    if args.limit is not None:
        end_index = min(end_index, start_index + args.limit)

    total_calls = 0
    domain_counts: Dict[str, int] = {}

    for idx in range(start_index, end_index):
        result = agent.solve(questions[idx]["input"])
        answers.append({"output": result.answer})
        total_calls += result.calls_used
        domain_counts[result.domain] = domain_counts.get(result.domain, 0) + 1

        question_number = idx + 1
        print(
            f"[{question_number}/{len(questions)}] "
            f"domain={result.domain:<17} calls={result.calls_used:<2} "
            f"answer={result.answer[:80]!r}"
        )

        solved_count = idx - start_index + 1
        if args.checkpoint_every > 0 and solved_count % args.checkpoint_every == 0:
            save_checkpoint(CHECKPOINT_PATH, idx + 1, answers)

    write_json(args.output, answers)

    if not args.skip_validation and len(answers) == len(questions):
        validate_results(questions, answers)
        print(f"Validated {len(answers)} answers successfully.")
    else:
        print("Skipped full validation because this was a partial run or --skip-validation was used.")

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
