"""Classifier inputs match the instruction in the supplied training dataset."""
from .data import first_present, nested

CLASSIFIER_INSTRUCTION = (
    "Classify the following conversation into one of these categories: writing, roleplay, "
    "extraction, reasoning, math, coding, STEM, or social_science. "
    "Consider both the conversation content and any existing tags provided."
)


def classifier_input(sample):
    turns = first_present(sample.raw, ("dialogs", "conversations", "messages"))
    if turns is None:
        turns = [{"role": "user", "content": sample.instruction},
                 {"role": "assistant", "content": sample.response}]
    parts = []
    for turn in turns[:5]:
        role = turn.get("role", turn.get("from", "unknown"))
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        content = turn.get("content", turn.get("value", ""))
        if len(content) > 3000:
            content = content[:3000] + "..."
        if content.strip():
            parts.append(f"{role.capitalize()}: {content}")
    tags = nested(sample.raw, "annotation.instag.content")
    tags_text = ", ".join(str(t) for t in tags[:5]) if isinstance(tags, list) and tags else "None"
    return "Conversation:\n" + "\n".join(parts) + f"\n\nExisting Tags: {tags_text}"


def classifier_query(sample):
    # LLaMA-Factory's Alpaca dataset loader joins instruction and input by a newline.
    return CLASSIFIER_INSTRUCTION + "\n" + classifier_input(sample)


def alpaca_prompt(sample):
    return ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{CLASSIFIER_INSTRUCTION}\n\n### Input:\n{classifier_input(sample)}\n\n### Response:\n")
