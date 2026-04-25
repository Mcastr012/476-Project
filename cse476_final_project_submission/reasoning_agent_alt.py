# FINAL PROJECT
from __future__ import annotations
import ast
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from functools import lru_cache

API_BASE_DEFAULT = "https://openai.rc.asu.edu/v1"
MODEL_DEFAULT = "qwen3-30b-a3b-instruct-2507"
WORD_RE = re.compile(r"[\w]+", re.UNICODE)

##add sparse matric operations support
try:
    import scipy.sparse as sp
    SCIPY_AVAILABLE = True
    print(f"Using SCRIPY")
except ImportError:
    SCIPY_AVAILABLE = False
    print(f"Using SCRIPY")

def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

def truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."

def tokenize(text: str) -> List[str]:
    return [token.lower() for token in WORD_RE.findall(text)]

def string_output(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

def strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip("\n")

def extract_last_boxed(text: str) -> Optional[str]:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text or "")
    if not matches:
        return None
    return matches[-1].strip()

def extract_last_number(text: str) -> Optional[str]:
    matches = re.findall(r"[-+]?\d+(?:/\d+|\.\d+)?", text or "")
    if not matches:
        return None
    return matches[-1]

def split_fact_list(text: str) -> List[str]:
    clean = (text or "").strip().rstrip(".")
    clean = clean.replace(", and ", ", ")
    clean = clean.replace(" and ", ", ")
    return [part.strip() for part in clean.split(",") if part.strip()]

@dataclass
class Example:
    input_text: str
    output_text: str
    domain: str
    tokens: Counter = field(default_factory=Counter)
    length: int = 0

@dataclass
class ParsedQuestion:
    raw: str
    context: str = ""
    options: Dict[str, str] = field(default_factory=dict)
    expects_boolean: bool = False
    expects_number: bool = False
    expects_boxed: bool = False
    code_prefix: str = ""
@dataclass
class SolveResult:
    answer: str
    domain: str
    calls_used: int
    notes: List[str] = field(default_factory=list)

class ExampleIndex:
    def __init__(self, examples: Iterable[Example]):
        self.examples = list(examples)
        self.by_domain: Dict[str, List[Example]] = defaultdict(list)
        self.doc_freq: Counter = Counter()
        self.avg_doc_len = 1.0

        total_length = 0
        for example in self.examples:
            if not example.tokens:
                example.tokens = Counter(tokenize(example.input_text))
            example.length = sum(example.tokens.values())
            total_length += example.length
            self.by_domain[example.domain].append(example)
            for token in example.tokens:
                self.doc_freq[token] += 1
        if self.examples:
            self.avg_doc_len = total_length / len(self.examples)

    @classmethod
    def from_json(cls, path: Path) -> "ExampleIndex":
        raw = json.loads(path.read_text(encoding="utf-8"))
        examples = [
            Example(
                input_text=row["input"],
                output_text=string_output(row["output"]),
                domain=row["domain"],
            )
            for row in raw
        ]
        return cls(examples)
    def _idf(self, token: str) -> float:
        n = len(self.examples)
        df = self.doc_freq.get(token, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def _score(self, query: Counter, example: Example) -> float:
        if not query:
            return 0.0
        score = 0.0
        k1 = 1.5
        b = 0.75
        for token, q_freq in query.items():
            tf = example.tokens.get(token, 0)
            if not tf:
                continue
            idf = self._idf(token)
            denom = tf + k1 * (1 - b + b * example.length / max(self.avg_doc_len, 1.0))
            score += q_freq * idf * (tf * (k1 + 1)) / max(denom, 1e-9)
        return score

    def search(self, query_text: str, domain: Optional[str] = None, limit: int = 3) -> List[Example]:
        query = Counter(tokenize(query_text))
        pool = self.by_domain.get(domain, []) if domain else self.examples
        scored = [(self._score(query, example), example) for example in pool]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [example for score, example in scored[:limit] if score > 0]

    def majority_domain(self, query_text: str, limit: int = 7) -> Optional[str]:
        hits = self.search(query_text, limit=limit)
        if not hits:
            return None
        vote = Counter(example.domain for example in hits)
        return vote.most_common(1)[0][0]

class QuestionParser:
    BOOLEAN_PREFIXES = (
        "is ",
        "are ",
        "do ",
        "does ",
        "did ",
        "can ",
        "could ",
        "would ",
        "should ",
        "has ",
        "have ",
        "was ",
        "were ",
    ) #asking for a yes/no style answer

    def parse(self, question: str) -> ParsedQuestion:
        lower = question.strip().lower()
        context = ""
        if "Context:" in question:
            context = question.split("Context:", 1)[1].strip()

        code_prefix = self._extract_code_prefix(question)
        options = self._extract_options(question)
        expects_boxed = "\\boxed{" in question
        expects_boolean = lower.startswith(self.BOOLEAN_PREFIXES)
        expects_boolean = expects_boolean or "is the following sentence plausible" in lower
        expects_boolean = expects_boolean or "answer with exactly true or false" in lower
        expects_number = self._looks_numeric(question)

        return ParsedQuestion(
            raw=question,
            context=context,
            options=options,
            expects_boolean=expects_boolean and not expects_boxed,
            expects_number=expects_number,
            expects_boxed=expects_boxed,
            code_prefix=code_prefix,
        )

    def _looks_numeric(self, question: str) -> bool:
        lower = question.lower()
        if "answer with just the number" in lower or "answer with just the integer" in lower:
            return True
        math_terms = [
            "calculate",
            "how many",
            "how much",
            "total",
            "remaining",
            "difference",
            "sum",
            "product",
            "area",
            "probability",
            "percent",
            "commission",
            "miles",
            "hours",
            "weeks",
            "price",
            "earned",
            "saved",
            "solve for",
        ]
        hits = sum(term in lower for term in math_terms)
        number_count = len(re.findall(r"\d", question))
        symbol_count = len(re.findall(r"[$=%^*/+\\-]", question))
        return (hits >= 1 and number_count >= 1) or symbol_count >= 3

    def _extract_code_prefix(self, question: str) -> str:
        match = re.search(
            r"starting with:\s*```(?:python)?\s*(.*?)```",
            question,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return ""
        return match.group(1).strip("\n")

    def _extract_options(self, question: str) -> Dict[str, str]:
        segments = []
        if "Options:" in question:
            segments.append(question.split("Options:", 1)[1])
        segments.append(question[-1200:])

        for segment in segments:
            options = self._extract_parenthesized_options(segment)
            if len(options) >= 2:
                return options
            options = self._extract_dotted_options(segment)
            if len(options) >= 2:
                return options
        return {}

    def _extract_parenthesized_options(self, text: str) -> Dict[str, str]:
        matches = re.findall(r"\(([A-Z])\)\s*(.*?)(?=\s*\([A-Z]\)\s*|$)", text, flags=re.DOTALL)
        cleaned = {}
        for key, value in matches:
            cleaned_value = normalize_whitespace(value.strip(' "\''))
            if cleaned_value:
                cleaned[key] = cleaned_value
        return cleaned

    def _extract_dotted_options(self, text: str) -> Dict[str, str]:
        pattern = re.compile(r"(?<!\w)([A-Z])\.\s*(.*?)(?=(?<!\w)[A-Z]\.\s*|$)", re.DOTALL)
        matches = pattern.findall(text)
        cleaned = {}
        for key, value in matches:
            cleaned_value = normalize_whitespace(value.strip(' "\''))
            if cleaned_value and len(cleaned_value) <= 200:
                cleaned[key] = cleaned_value
        return cleaned

class DomainRouter:
    def __init__(self, index: ExampleIndex):
        self.index = index

    def route(self, question: str) -> str:
        lower = question.lower()
        if "write self-contained code starting with" in lower:
            return "coding"
        if "future events" in lower or "\\boxed{" in question or "do not refuse to make a prediction" in lower:
            return "future_prediction"
        if "my plan is as follows" in lower or "[plan]" in lower:
            return "planning"

        retrieval_vote = self.index.majority_domain(question)
        math_score = self._math_score(lower)
        common_score = self._common_sense_score(lower)

        if retrieval_vote in {"math", "common_sense"}:
            if math_score >= common_score + 2:
                return "math"
            if common_score >= math_score + 2:
                return "common_sense"
            return retrieval_vote
        return "math" if math_score > common_score else "common_sense"

    def _math_score(self, lower: str) -> int:
        score = 0
        if len(re.findall(r"\d", lower)) >= 6:
            score += 2
        terms = [
            "calculate",
            "solve",
            "triangle",
            "quadrilateral",
            "integer",
            "commission",
            "allowance",
            "miles",
            "hours",
            "probability",
            "area",
            "percent",
            "per hour",
            "per week",
            "per day",
            "ratio",
            "product",
            "sum",
            "difference",
            "remaining",
            "total",
        ]
        score += sum(term in lower for term in terms)
        if len(re.findall(r"[$=%^*/+\\-]", lower)) >= 3:
            score += 1
        if "how many" in lower or "how much" in lower:
            score += 2
        return int(score)

    def _common_sense_score(self, lower: str) -> int:
        score = 0
        terms = [
            "context:",
            "answer the question using the context",
            "which city",
            "what nationality",
            "which record label",
            "plausible",
            "who is",
            "what type",
            "which country",
            "when was",
        ]
        score += sum(term in lower for term in terms)
        if lower.startswith(("who ", "which ", "what ", "when ", "where ")):
            score += 1
        return score

class APIModelClient:
    def __init__(
        self,
        api_key: str,
        api_base: str = API_BASE_DEFAULT,
        model_name: str = MODEL_DEFAULT,
        timeout: int = 60,
        max_retries: int = 4,
    ):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(self.max_retries):
            try:
                request = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    status_code = response.getcode()
                    body = response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                status_code = exc.code
                body = exc.read().decode("utf-8", errors="replace")
            except urllib.error.URLError as exc:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"API request failed: {exc}") from exc
                time.sleep(1.5 * (attempt + 1))
                continue

            if status_code == 200:
                data = json.loads(body)
                return data["choices"][0]["message"]["content"].strip()

            retryable = status_code in {408, 409, 429, 500, 502, 503, 504}
            if not retryable or attempt == self.max_retries - 1:
                try:
                    error_text = json.loads(body)
                except Exception:
                    error_text = body
                raise RuntimeError(
                    f"API error {status_code}: {error_text}"
                )
            time.sleep(2.0 * (attempt + 1))

        raise RuntimeError("API request failed after retries.")


class PlanningValidator:
    def validate(self, question: str, answer: str) -> Optional[str]:
        actions = re.findall(r"\(([^()]+)\)", answer or "")
        if not actions:
            return "The plan should contain parenthesized actions like (pick-up blue)."

        lower = question.lower()
        if lower.startswith("i am playing with a set of blocks"):
            return self._validate_blocks(question, actions)
        if lower.startswith("i am playing with a set of objects"):
            return self._validate_mystery_objects(question, actions)
        if lower.startswith("i have to plan"):
            return None
        return None

    def _extract_target_statement(self, question: str) -> str:
        matches = re.findall(r"\[STATEMENT\](.*?)My plan is as follows:", question, flags=re.DOTALL)
        if not matches:
            raise ValueError("Could not find planning statement.")
        return matches[-1]

    def _validate_blocks(self, question: str, actions: List[str]) -> Optional[str]:
        statement = self._extract_target_statement(question)
        match = re.search(
            r"As initial conditions I have that,\s*(.*?)\.\s*My goal is to have that\s*(.*?)\.",
            statement,
            flags=re.DOTALL,
        )
        if not match:
            return None
        initial_facts = split_fact_list(match.group(1))
        goal_facts = split_fact_list(match.group(2))
        state = {
            "on": {},
            "clear": set(),
            "holding": None,
            "hand_empty": False,
        }

        for fact in initial_facts:
            if fact == "the hand is empty":
                state["hand_empty"] = True
            else:
                clear_match = re.match(r"the (\w+) block is clear", fact)
                if clear_match:
                    state["clear"].add(clear_match.group(1))
                    continue
                table_match = re.match(r"the (\w+) block is on the table", fact)
                if table_match:
                    state["on"][table_match.group(1)] = "table"
                    continue
                top_match = re.match(r"the (\w+) block is on top of the (\w+) block", fact)
                if top_match:
                    block, support = top_match.groups()
                    state["on"][block] = support
        for step_number, action_text in enumerate(actions, start=1):
            parts = action_text.split()
            verb = parts[0]
            try:
                if verb == "pick-up":
                    block = parts[1]
                    if not state["hand_empty"] or state["on"].get(block) != "table" or block not in state["clear"]:
                        return f"Step {step_number} is invalid: cannot pick up {block}."
                    state["holding"] = block
                    state["hand_empty"] = False
                    state["on"].pop(block, None)
                    state["clear"].discard(block)
                elif verb == "put-down":
                    block = parts[1]
                    if state["holding"] != block:
                        return f"Step {step_number} is invalid: not holding {block}."
                    state["holding"] = None
                    state["hand_empty"] = True
                    state["on"][block] = "table"
                    state["clear"].add(block)
                elif verb == "unstack":
                    block, support = parts[1], parts[2]
                    if not state["hand_empty"] or state["on"].get(block) != support or block not in state["clear"]:
                        return f"Step {step_number} is invalid: cannot unstack {block} from {support}."
                    state["holding"] = block
                    state["hand_empty"] = False
                    state["on"].pop(block, None)
                    state["clear"].discard(block)
                    state["clear"].add(support)
                elif verb == "stack":
                    block, support = parts[1], parts[2]
                    if state["holding"] != block or support not in state["clear"]:
                        return f"Step {step_number} is invalid: cannot stack {block} on {support}."
                    state["holding"] = None
                    state["hand_empty"] = True
                    state["on"][block] = support
                    state["clear"].add(block)
                    state["clear"].discard(support)
                else:
                    return f"Step {step_number} uses an unknown action: {verb}."
            except IndexError:
                return f"Step {step_number} is malformed."

        for fact in goal_facts:
            table_match = re.match(r"the (\w+) block is on the table", fact)
            if table_match and state["on"].get(table_match.group(1)) != "table":
                return f"Goal not satisfied: {fact}."
            top_match = re.match(r"the (\w+) block is on top of the (\w+) block", fact)
            if top_match:
                block, support = top_match.groups()
                if state["on"].get(block) != support:
                    return f"Goal not satisfied: {fact}."
            clear_match = re.match(r"the (\w+) block is clear", fact)
            if clear_match and clear_match.group(1) not in state["clear"]:
                return f"Goal not satisfied: {fact}."
        return None

    def _validate_mystery_objects(self, question: str, actions: List[str]) -> Optional[str]:
        statement = self._extract_target_statement(question)
        match = re.search(
            r"As initial conditions I have that,\s*(.*?)\.\s*My goal is to have that\s*(.*?)\.",
            statement,
            flags=re.DOTALL,
        )
        if not match:
            return None
        initial_facts = split_fact_list(match.group(1))
        goal_facts = split_fact_list(match.group(2))
        state = {
            "harmony": False,
            "pain": set(),
            "province": set(),
            "planet": set(),
            "craves": set(),
        }
        for fact in initial_facts:
            if fact == "harmony":
                state["harmony"] = True
                continue
            pain_match = re.match(r"pain object (\w+)", fact)
            if pain_match:
                state["pain"].add(pain_match.group(1))
                continue
            province_match = re.match(r"province object (\w+)", fact)
            if province_match:
                state["province"].add(province_match.group(1))
                continue
            planet_match = re.match(r"planet object (\w+)", fact)
            if planet_match:
                state["planet"].add(planet_match.group(1))
                continue
            craves_match = re.match(r"object (\w+) craves object (\w+)", fact)
            if craves_match:
                state["craves"].add(craves_match.groups())

        for step_number, action_text in enumerate(actions, start=1):
            parts = action_text.split()
            verb = parts[0]
            try:
                if verb == "attack":
                    obj = parts[1]
                    if obj not in state["province"] or obj not in state["planet"] or not state["harmony"]:
                        return f"Step {step_number} is invalid: cannot attack {obj}."
                    state["pain"].add(obj)
                    state["province"].discard(obj)
                    state["planet"].discard(obj)
                    state["harmony"] = False
                elif verb == "succumb":
                    obj = parts[1]
                    if obj not in state["pain"]:
                        return f"Step {step_number} is invalid: cannot succumb {obj}."
                    state["pain"].discard(obj)
                    state["province"].add(obj)
                    state["planet"].add(obj)
                    state["harmony"] = True
                elif verb == "overcome":
                    obj, other = parts[1], parts[2]
                    if other not in state["province"] or obj not in state["pain"]:
                        return f"Step {step_number} is invalid: cannot overcome {obj} from {other}."
                    state["harmony"] = True
                    state["province"].add(obj)
                    state["province"].discard(other)
                    state["pain"].discard(obj)
                    state["craves"].add((obj, other))
                elif verb == "feast":
                    obj, other = parts[1], parts[2]
                    if (obj, other) not in state["craves"] or obj not in state["province"] or not state["harmony"]:
                        return f"Step {step_number} is invalid: cannot feast {obj} from {other}."
                    state["pain"].add(obj)
                    state["province"].add(other)
                    state["craves"].discard((obj, other))
                    state["province"].discard(obj)
                    state["harmony"] = False
                else:
                    return f"Step {step_number} uses an unknown action: {verb}."
            except IndexError:
                return f"Step {step_number} is malformed."

        for fact in goal_facts:
            if fact == "harmony" and not state["harmony"]:
                return "Goal not satisfied: harmony."
            pain_match = re.match(r"pain object (\w+)", fact)
            if pain_match and pain_match.group(1) not in state["pain"]:
                return f"Goal not satisfied: {fact}."
            province_match = re.match(r"province object (\w+)", fact)
            if province_match and province_match.group(1) not in state["province"]:
                return f"Goal not satisfied: {fact}."
            planet_match = re.match(r"planet object (\w+)", fact)
            if planet_match and planet_match.group(1) not in state["planet"]:
                return f"Goal not satisfied: {fact}."
            craves_match = re.match(r"object (\w+) craves object (\w+)", fact)
            if craves_match and craves_match.groups() not in state["craves"]:
                return f"Goal not satisfied: {fact}."
        return None

class AnswerNormalizer:
    def normalize(self, answer: str, parsed: ParsedQuestion, domain: str) -> str:
        if domain == "coding":
            return self._normalize_coding_answer(answer, parsed)
        if domain == "future_prediction":
            return self._normalize_future_answer(answer, parsed)
        if domain == "planning":
            return self._normalize_plan(answer)
        return self._normalize_short_answer(answer, parsed)

    def _normalize_coding_answer(self, answer: str, parsed: ParsedQuestion) -> str:
        answer = strip_code_fences(answer)
        answer = answer.replace("\r\n", "\n").strip("\n")

        if parsed.code_prefix and parsed.code_prefix in answer:
            answer = answer.split(parsed.code_prefix, 1)[1].lstrip("\n")

        if "def task_func" in answer:
            lines = answer.splitlines()
            body_lines = []
            seen_signature = False
            for line in lines:
                if line.lstrip().startswith("def task_func"):
                    seen_signature = True
                    continue
                if seen_signature:
                    body_lines.append(line)
            answer = "\n".join(body_lines).strip("\n")

        cleaned_lines = []
        for line in answer.splitlines():
            if not line.strip():
                cleaned_lines.append("")
                continue
            if line.startswith("    ") or line.startswith("\t"):
                cleaned_lines.append(line.replace("\t", "    "))
            else:
                cleaned_lines.append("    " + line.lstrip())
        return "\n".join(cleaned_lines).rstrip() or "    pass"

    def _normalize_future_answer(self, answer: str, parsed: ParsedQuestion) -> str:
        answer = strip_code_fences(answer)
        content = extract_last_boxed(answer) or normalize_whitespace(answer)
        content = self._coerce_option_content(content, parsed.options, keep_letters=True)
        content = content.strip()
        return f"\\boxed{{{content}}}"

    def _normalize_plan(self, answer: str) -> str:
        actions = re.findall(r"\([^()]+\)", answer or "")
        if actions:
            return "\n".join(action.strip() for action in actions)
        return normalize_whitespace(answer)

    def _normalize_short_answer(self, answer: str, parsed: ParsedQuestion) -> str:
        answer = strip_code_fences(answer)
        answer = answer.strip()

        if parsed.expects_boolean:
            lowered = answer.lower()
            if lowered.startswith("true") or lowered == "yes":
                return "True"
            if lowered.startswith("false") or lowered == "no":
                return "False"
        if parsed.options:
            mapped = self._coerce_option_content(answer, parsed.options, keep_letters=False)
            if mapped:
                answer = mapped

        if parsed.expects_number:
            number = extract_last_number(answer)
            if number and len(answer) > len(number) + 8:
                return number

        return answer.strip()
    def _coerce_option_content(
        self,
        answer: str,
        options: Dict[str, str],
        keep_letters: bool,
    ) -> str:
        answer = normalize_whitespace(answer)
        if not options:
            return answer

        raw_parts = [part.strip() for part in answer.split(",") if part.strip()]
        if not raw_parts:
            raw_parts = [answer]

        picked: List[str] = []
        lowered_options = {key: value.lower() for key, value in options.items()}

        for part in raw_parts:
            letter_match = re.fullmatch(r"[A-Z]", part.upper())
            if letter_match:
                key = letter_match.group(0)
                picked.append(key if keep_letters else options.get(key, part))
                continue

            lowered = part.lower()
            for key, option_text in lowered_options.items():
                if lowered == option_text or lowered in option_text or option_text in lowered:
                    picked.append(key if keep_letters else options[key])
                    break

        if picked:
            unique_picked = []
            seen = set()
            for item in picked:
                if item not in seen:
                    unique_picked.append(item)
                    seen.add(item)
            return ", ".join(unique_picked)
        return answer

class NearestNeighborBaseline:
    def __init__(self, index: ExampleIndex):
        self.index = index

    def solve(self, question: str, domain: str) -> str:
        hits = self.index.search(question, domain=domain, limit=1)
        if hits:
            return hits[0].output_text
        hits = self.index.search(question, limit=1)
        if hits:
            return hits[0].output_text
        return "Unknown"

class ReasoningAgent:
    def __init__(
        self,
        index: ExampleIndex,
        parser: QuestionParser,
        router: DomainRouter,
        normalizer: AnswerNormalizer,
        model_client: Optional[APIModelClient] = None,
        backend: str = "api",
        max_calls_per_question: int = 6,
        seed: int = 7,
    ):
        self.index = index
        self.parser = parser
        self.router = router
        self.normalizer = normalizer
        self.model_client = model_client
        self.backend = backend
        self.max_calls_per_question = max_calls_per_question
        self.random = random.Random(seed)
        self.validator = PlanningValidator()
        self.baseline = NearestNeighborBaseline(index)

    def solve(self, question: str) -> SolveResult:
        parsed = self.parser.parse(question)
        domain = self.router.route(question)
        notes: List[str] = []
        calls_used = 0

        if self.backend == "nearest":
            answer = self.baseline.solve(question, domain)
            answer = self.normalizer.normalize(answer, parsed, domain)
            return SolveResult(answer=answer, domain=domain, calls_used=0, notes=["nearest-neighbor baseline"])

        if not self.model_client:
            raise RuntimeError("API backend requested, but no model client is configured.")
        if domain == "coding":
            answer, calls_used, notes = self._solve_coding(question, parsed)
        elif domain == "planning":
            answer, calls_used, notes = self._solve_planning(question, parsed)
        elif domain == "future_prediction":
            answer, calls_used, notes = self._solve_future(question, parsed)
        elif domain == "math":
            answer, calls_used, notes = self._solve_math(question, parsed)
        else:
            answer, calls_used, notes = self._solve_common_sense(question, parsed)

        answer = self.normalizer.normalize(answer, parsed, domain)
        return SolveResult(answer=answer, domain=domain, calls_used=calls_used, notes=notes)

    def _call_model(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        calls_used: int,
    ) -> Tuple[str, int]:
        if calls_used >= self.max_calls_per_question:
            raise RuntimeError("Per-question call budget exceeded.")
        text = self.model_client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return text, calls_used + 1

    def _select_examples(self, question: str, domain: str, limit: int = 2) -> List[Example]:
        if len(question) > 4500:
            limit = 1
        return self.index.search(question, domain=domain, limit=limit)

    def _format_examples(self, examples: List[Example], input_limit: int = 450, output_limit: int = 250) -> str:
        if not examples:
            return "No retrieved examples.\n"
        blocks = []
        for idx, example in enumerate(examples, start=1):
            blocks.append(
                f"Example {idx}\n"
                f"Question:\n{truncate(example.input_text, input_limit)}\n\n"
                f"Answer:\n{truncate(example.output_text, output_limit)}"
            )
        return "\n\n".join(blocks) + "\n"

    def _build_common_rules(self, parsed: ParsedQuestion, domain: str) -> List[str]:
        rules = [
            "Work through the reasoning privately.",
            "Return only the final answer.",
        ]
        if parsed.context:
            rules.append("Use the provided context when it is enough to answer the question.")
        if parsed.options and domain != "future_prediction":
            rules.append("If the question has labeled choices, return the exact option text, not the letter.")
        if parsed.expects_boolean and domain != "future_prediction":
            rules.append("Return exactly True or False.")
        if parsed.expects_boxed:
            rules.append("Preserve the exact boxed format requested by the question.")
        if parsed.expects_number and domain in {"math", "common_sense"}:
            rules.append("If the answer is a single numeric value, keep the final answer short.")
        return rules

    def _question_complexity(self, question: str, parsed: ParsedQuestion) -> int:
        score = 0
        if len(question) > 1200:
            score += 2
        if parsed.context:
            score += 1
        if parsed.options:
            score += 1
        if len(re.findall(r"\d", question)) >= 8:
            score += 2
        if any(symbol in question for symbol in ["sqrt", "^", "∠", "triangle", "$"]):
            score += 2
        return score
    def _solve_math(self, question: str, parsed: ParsedQuestion) -> Tuple[str, int, List[str]]:
        calls_used = 0
        notes = ["domain routing", "retrieval few-shot", "self-consistency", "answer verification"]
        examples = self._select_examples(question, "math", limit=2)
        example_block = self._format_examples(examples)
        rules = "\n".join(f"- {rule}" for rule in self._build_common_rules(parsed, "math"))

        system_prompt = (
            "You are a careful mathematical reasoning agent. "
            "Reason step by step in private and give only the final answer."
        )
        base_prompt = (
            f"Solve the math problem.\n\nRules:\n{rules}\n\n"
            f"Retrieved examples:\n{example_block}\n"
            f"Problem:\n{question}\n\nFinal answer only:"
        )

        first_answer, calls_used = self._call_model(
            system_prompt=system_prompt,
            user_prompt=base_prompt,
            max_tokens=220,
            temperature=0.0,
            calls_used=calls_used,
        )
        normalized_first = self.normalizer.normalize(first_answer, parsed, "math")
        if self._question_complexity(question, parsed) < 3:
            return normalized_first, calls_used, notes

        second_prompt = (
            f"Solve the same problem again, but use a different line of attack internally.\n\n"
            f"Rules:\n{rules}\n\nProblem:\n{question}\n\nFinal answer only:"
        )
        second_answer, calls_used = self._call_model(
            system_prompt=system_prompt,
            user_prompt=second_prompt,
            max_tokens=220,
            temperature=0.3,
            calls_used=calls_used,
        )
        normalized_second = self.normalizer.normalize(second_answer, parsed, "math")

        if self._answers_agree(normalized_first, normalized_second, parsed, "math"):
            return normalized_first, calls_used, notes

        judge_prompt = (
            "Pick the better final answer for the problem below.\n"
            "Return only one final answer with no explanation.\n\n"
            f"Problem:\n{question}\n\n"
            f"Candidate A:\n{normalized_first}\n\n"
            f"Candidate B:\n{normalized_second}\n"
        )
        judged_answer, calls_used = self._call_model(
            system_prompt="You are a strict verifier. Return only the better final answer.",
            user_prompt=judge_prompt,
            max_tokens=80,
            temperature=0.0,
            calls_used=calls_used,
        )
        return judged_answer, calls_used, notes

    def _solve_common_sense(self, question: str, parsed: ParsedQuestion) -> Tuple[str, int, List[str]]:
        calls_used = 0
        notes = ["domain routing", "retrieval few-shot", "context-aware prompting", "format control"]
        examples = self._select_examples(question, "common_sense", limit=2)
        example_block = self._format_examples(examples)
        rules = "\n".join(f"- {rule}" for rule in self._build_common_rules(parsed, "common_sense"))
        system_prompt = (
            "You are a careful question-answering agent. "
            "Use context when provided and respond with only the final answer."
        )
        prompt = (
            f"Answer the question.\n\nRules:\n{rules}\n\n"
            f"Retrieved examples:\n{example_block}\n"
            f"Question:\n{question}\n\nFinal answer only:"
        )
        answer, calls_used = self._call_model(
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_tokens=140,
            temperature=0.0,
            calls_used=calls_used,
        )

        normalized = self.normalizer.normalize(answer, parsed, "common_sense")
        if self._looks_well_formed(normalized, parsed, "common_sense"):
            return normalized, calls_used, notes

        repair_prompt = (
            "Rewrite the draft answer so that it matches the requested format exactly.\n"
            "Return only the corrected final answer.\n\n"
            f"Question:\n{question}\n\n"
            f"Draft answer:\n{answer}\n"
        )
        repaired, calls_used = self._call_model(
            system_prompt="You are fixing answer formatting.",
            user_prompt=repair_prompt,
            max_tokens=80,
            temperature=0.0,
            calls_used=calls_used,
        )
        return repaired, calls_used, notes

    def _solve_coding(self, question: str, parsed: ParsedQuestion) -> Tuple[str, int, List[str]]:
        calls_used = 0
        notes = ["domain routing", "retrieval few-shot", "tool validation", "self-refinement"]
        examples = self._select_examples(question, "coding", limit=2)
        example_block = self._format_examples(examples, input_limit=360, output_limit=420)
        scaffold = parsed.code_prefix or "def task_func(...):"
        system_prompt = (
            "You write clean Python function bodies. "
            "Return only the code that belongs inside the function body."
        )
        prompt = (
            "Complete the function described below.\n\n"
            "Rules:\n"
            "- Return only the function body after the provided header.\n"
            "- Do not repeat imports, the function signature, or Markdown fences.\n"
            "- Keep the code self-contained and practical.\n\n"
            f"Retrieved examples:\n{example_block}\n"
            f"Function header:\n```python\n{scaffold}\n```\n\n"
            f"Task description:\n{question}\n\nFunction body only:"
        )
        # Get the first draft from the model.
        draft, calls_used = self._call_model(
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_tokens=900,
            temperature=0.0,
            calls_used=calls_used,
        )
        normalized = self.normalizer.normalize(draft, parsed, "coding")
        validation_error = self._validate_python_answer(parsed.code_prefix, normalized)
        if validation_error is None:
            return normalized, calls_used, notes

        repair_prompt = (
            "Fix the Python function body below.\n"
            "Return only the corrected function body.\n\n"
            f"Function header:\n```python\n{scaffold}\n```\n\n"
            f"Original task:\n{question}\n\n"
            f"Current draft:\n```python\n{normalized}\n```\n\n"
            f"Validation error:\n{validation_error}\n"
        )
        repaired, calls_used = self._call_model(
            system_prompt="You repair Python code while keeping the task requirements intact.",
            user_prompt=repair_prompt,
            max_tokens=900,
            temperature=0.0,
            calls_used=calls_used,
        )
        return repaired, calls_used, notes

    def _solve_planning(self, question: str, parsed: ParsedQuestion) -> Tuple[str, int, List[str]]:
        calls_used = 0
        notes = ["domain routing", "decomposition", "tool validation", "self-refinement"]
        examples = self._select_examples(question, "planning", limit=2)
        example_block = self._format_examples(examples, input_limit=600, output_limit=300)
        system_prompt = (
            "You solve planning problems. "
            "Return only the final plan as parenthesized actions, one per line."
        )
        prompt = (
            "Produce the plan for the final empty [PLAN] block in the problem below.\n\n"
            "Rules:\n"
            "- Use the exact action names from the problem.\n"
            "- Return only parenthesized actions like (pick-up blue).\n"
            "- One action per line is preferred.\n\n"
            f"Retrieved examples:\n{example_block}\n"
            f"Problem:\n{question}\n\nPlan only:"
        )
        draft, calls_used = self._call_model(
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_tokens=900,
            temperature=0.0,
            calls_used=calls_used,
        )
        normalized = self.normalizer.normalize(draft, parsed, "planning")
        validation_error = self.validator.validate(question, normalized)
        if validation_error is None:
            return normalized, calls_used, notes

        repair_prompt = (
            "Repair the plan below so that it satisfies the problem constraints.\n"
            "Return only the corrected plan as parenthesized actions.\n\n"
            f"Problem:\n{question}\n\n"
            f"Current draft:\n{normalized}\n\n"
            f"Validation feedback:\n{validation_error}\n"
        )
        repaired, calls_used = self._call_model(
            system_prompt="You fix planning outputs while preserving the target goal.",
            user_prompt=repair_prompt,
            max_tokens=900,
            temperature=0.0,
            calls_used=calls_used,
        )
        return repaired, calls_used, notes
    
    def _solve_future(self, question: str, parsed: ParsedQuestion) -> Tuple[str, int, List[str]]:
        calls_used = 0
        notes = ["domain routing", "forecasting prompt", "format control", "retrieval few-shot"]
        examples = self._select_examples(question, "future_prediction", limit=2)
        example_block = self._format_examples(examples, input_limit=360, output_limit=120)
        system_prompt = (
            "You are a calibrated forecaster. "
            "Think privately about base rates, current strength, and uncertainty. "
            "Return only the boxed final prediction."
        )
        prompt = (
            "Make the best prediction you can.\n\n"
            "Rules:\n"
            "- Match the exact boxed format requested in the question.\n"
            "- Do not add explanation.\n"
            "- If the question provides labeled options, keep the answer consistent with those labels.\n\n"
            f"Retrieved examples:\n{example_block}\n"
            f"Prediction task:\n{question}\n\nBoxed final answer only:"
        )
        answer, calls_used = self._call_model(
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_tokens=120,
            temperature=0.2,
            calls_used=calls_used,
        )
        return answer, calls_used, notes

    def _answers_agree(self, first: str, second: str, parsed: ParsedQuestion, domain: str) -> bool:
        first_norm = self.normalizer.normalize(first, parsed, domain).lower()
        second_norm = self.normalizer.normalize(second, parsed, domain).lower()
        if first_norm == second_norm:
            return True

        first_num = extract_last_number(first_norm)
        second_num = extract_last_number(second_norm)
        if parsed.expects_number and first_num and second_num and first_num == second_num:
            return True

        if domain == "future_prediction":
            return extract_last_boxed(first) == extract_last_boxed(second)
        return False

    def _looks_well_formed(self, answer: str, parsed: ParsedQuestion, domain: str) -> bool:
        if not answer:
            return False
        if parsed.expects_boolean and answer not in {"True", "False"}:
            return False
        if parsed.expects_boxed and not re.search(r"\\boxed\{.+\}", answer):
            return False
        if parsed.options and domain != "future_prediction":
            option_values = {normalize_whitespace(text).lower() for text in parsed.options.values()}
            if len(answer.split()) <= 4 and answer.lower() not in option_values and answer.upper() in parsed.options:
                return False
        return len(answer) < 5000

    def _validate_python_answer(self, code_prefix: str, body: str) -> Optional[str]:
        prefix = code_prefix.strip("\n") or "def task_func():"
        full_code = prefix + "\n" + body.rstrip() + "\n"
        try:
            ast.parse(full_code)
        except SyntaxError as exc:
            return f"SyntaxError on line {exc.lineno}: {exc.msg}"
        return None
def find_dev_data_path(explicit_path: Optional[str] = None) -> Path:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    env_path = os.getenv("DEV_DATA_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    project_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            project_dir / "cse476_final_project_dev_data.json",
            project_dir.parent / "final_project_tutorial_and_dev_data" / "cse476_final_project_dev_data.json",
            Path("/Users/marcocastro/Downloads/final_project_tutorial_and_dev_data/cse476_final_project_dev_data.json"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find cse476_final_project_dev_data.json. "
        "Set DEV_DATA_PATH or place the file next to this project."
    )

def build_agent(
    *,
    dev_data_path: Optional[str] = None,
    backend: str = "api",
    max_calls_per_question: int = 6,
) -> ReasoningAgent:
    project_dir = Path(__file__).resolve().parent
    load_env_file(project_dir / ".env")
    index = ExampleIndex.from_json(find_dev_data_path(dev_data_path))
    parser = QuestionParser()
    router = DomainRouter(index)
    normalizer = AnswerNormalizer()

    if backend == "nearest":
        return ReasoningAgent(
            index=index,
            parser=parser,
            router=router,
            normalizer=normalizer,
            model_client=None,
            backend="nearest",
            max_calls_per_question=max_calls_per_question,
        )

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Create a .env file or export the variable before running."
        )
    model_client = APIModelClient(
        api_key=api_key,
        api_base=os.getenv("API_BASE", API_BASE_DEFAULT),
        model_name=os.getenv("MODEL_NAME", MODEL_DEFAULT),
    )
    return ReasoningAgent(
        index=index,
        parser=parser,
        router=router,
        normalizer=normalizer,
        model_client=model_client,
        backend="api",
        max_calls_per_question=max_calls_per_question,
    )
