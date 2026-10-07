"""JSON/JSONL and legacy field adapters, with stable IDs and original records."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import math


TASKS = ("math", "coding", "stem", "humanities", "reasoning", "writing", "roleplay", "extraction")
ALIASES = {"code": "coding", "mathematics": "math", "social_science": "humanities",
           "social science": "humanities", "role-play": "roleplay", "role_play": "roleplay"}


def canonical_task(value):
    if not isinstance(value, str):
        return "unknown"
    value = value.strip().lower()
    value = ALIASES.get(value, value)
    return value if value in TASKS else "unknown"


def nested(record, path):
    current = record
    for key in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def first_present(record, paths):
    for path in paths:
        value = nested(record, path)
        if value is not None:
            return value
    return None


def finite_mean(value):
    if value is None:
        return None
    values = value if isinstance(value, list) else [value]
    if not values or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise ValueError("Metrics must be nonempty finite numeric scalars/lists")
    return sum(values) / len(values)


@dataclass
class Sample:
    id: str
    instruction: str
    response: str
    text: str
    raw: dict = field(repr=False)
    outer_label: str | None = None
    inner_label: str | None = None
    task: str = "unknown"
    embedding: list | None = field(default=None, repr=False)
    metrics: dict = field(default_factory=dict)
    task_is_provided: bool = False


def adapt_record(raw, index):
    if not isinstance(raw, dict):
        raise ValueError(f"Record {index} must be a JSON object")
    identifier = first_present(raw, ("id", "uuid", "sample_id"))
    if identifier is not None and (isinstance(identifier, bool) or not isinstance(identifier, (int, str))):
        raise ValueError(f"Record {index}: id must be a string or integer")
    identifier = str(identifier) if identifier is not None else f"row-{index:08d}"
    dialogs = first_present(raw, ("dialogs", "conversations", "messages"))
    if dialogs is not None:
        if not isinstance(dialogs, list):
            raise ValueError(f"{identifier}: dialogs must be a list")
        turns = []
        for turn in dialogs:
            if not isinstance(turn, dict):
                raise ValueError(f"{identifier}: invalid dialogue turn")
            role = turn.get("role", turn.get("from", ""))
            content = turn.get("content", turn.get("value", ""))
            if not isinstance(content, str):
                raise ValueError(f"{identifier}: dialogue content must be text")
            turns.append((role, content))
        instruction = "\n".join(t for r, t in turns if r in ("human", "user"))
        response = "\n".join(t for r, t in turns if r in ("gpt", "assistant"))
        text = "\n".join(t for _, t in turns)
    else:
        parts = [raw.get("instruction", raw.get("prompt", "")), raw.get("input", "")]
        response = raw.get("output", raw.get("response", ""))
        if not all(isinstance(t, str) for t in parts + [response]):
            raise ValueError(f"{identifier}: instruction/input/output must be text")
        instruction = "\n".join(t for t in parts if t)
        text = instruction + "\n" + response
    if not instruction.strip():
        raise ValueError(f"{identifier}: no user instruction found")
    task = "unknown"
    task_is_provided = False
    for path in ("task_label", "category", "alpaca_prediction.category", "gpt_classification.category", "mtbench_category", "classification.category"):
        value = nested(raw, path)
        task = canonical_task(value)
        if task in TASKS:
            task_is_provided = True
            break
    metrics = {
        "complexity": finite_mean(first_present(raw, ("metrics.complexity", "complexity", "annotation.deita.complexity_scores"))),
        "quality": finite_mean(first_present(raw, ("metrics.quality", "quality", "annotation.deita.quality_scores"))),
        "ifd": finite_mean(first_present(raw, ("metrics.ifd", "ifd", "ifd_score.content"))),
    }
    if metrics["ifd"] is None:
        pc = finite_mean(nested(raw, "ppl_conditional"))
        pu = finite_mean(nested(raw, "ppl_unconditional"))
        if pc is not None and pu is not None:
            if pc <= 0 or pu <= 0:
                raise ValueError(f"{identifier}: perplexity must be positive")
            metrics["ifd"] = pc / pu
    outer = first_present(raw, ("outer_label", "outer_task_label"))
    inner = first_present(raw, ("inner_label", "sub_label"))
    return Sample(identifier, instruction, response, text, raw,
                  None if outer is None else str(outer), None if inner is None else str(inner),
                  task, first_present(raw, ("emb", "embedding", "features")), metrics, task_is_provided)


def load_samples(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig") as stream:
        first = ""
        while not first:
            char = stream.read(1)
            if not char:
                raise ValueError("Input data is empty")
            first = char.strip()
        stream.seek(0)
        if first == "[":
            records = json.load(stream)
        else:
            records = []
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
    samples = [adapt_record(record, index) for index, record in enumerate(records)]
    if not samples:
        raise ValueError("Input data is empty")
    seen = set()
    for sample in samples:
        if sample.id in seen:
            raise ValueError(f"Duplicate sample ID: {sample.id}")
        seen.add(sample.id)
    return samples


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
