"""Inner task annotation, independent of coarse/fine cluster IDs."""
import re
from .data import TASKS, canonical_task


# English-only synthetic-demo backend; production uses the Qwen classifier.
RULES = {
    "coding": r"\b(python|javascript|sql|debug|program|function|algorithm|code|compiler)\b",
    "math": r"\b(equation|integral|calculate|algebra|geometry|probability|solve|derivative)\b",
    "extraction": r"\b(extract|entities|csv|json|schema|fields)\b",
    "roleplay": r"\b(pretend|roleplay|act as|role of|stay in character)\b",
    "stem": r"\b(physics|chemistry|biology|quantum|molecule|photosynthesis|neuron|gravity)\b",
    "humanities": r"\b(history|philosophy|economics|renaissance|ethics|democracy|society)\b",
    "reasoning": r"\b(reasoning|logic|riddle|puzzle|deduce|infer|pros and cons)\b",
    "writing": r"\b(write|draft|poem|story|email|rewrite|compose|essay)\b",
}


def rule_label(text):
    scores = {task: len(re.findall(pattern, text, re.IGNORECASE)) for task, pattern in RULES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


def label_tasks(samples, config, predictor=None):
    """Optional predictor: callable(list[str]) -> list[str], e.g. your Qwen model."""
    indices = [i for i, s in enumerate(samples) if config.relabel_tasks or not s.task_is_provided]
    if not indices:
        return {"backend": "existing", "labeled": 0, "unknown": sum(s.task == "unknown" for s in samples)}
    texts = [samples[i].instruction for i in indices]
    if predictor is not None:
        labels = predictor(texts)
    elif config.task_backend == "rules":
        labels = [rule_label(text) for text in texts]
    elif config.task_backend == "transformers":
        from .models import TransformersTaskLabeler, release_model
        adapter = TransformersTaskLabeler(config)
        try:
            labels = adapter.predict_samples([samples[i] for i in indices])
        finally:
            release_model(adapter)
    else:
        raise ValueError(f"{samples[indices[0]].id}: missing task label; supply task_label/alpaca_prediction.category or configure a task backend")
    if len(labels) != len(indices):
        raise ValueError("Task predictor returned the wrong number of labels")
    for i, label in zip(indices, labels):
        normalized = canonical_task(label)
        if normalized not in TASKS:
            raise ValueError(f"Invalid task prediction {label!r}; allowed: {TASKS}")
        samples[i].task = normalized
        samples[i].task_is_provided = True
    return {"backend": "callback" if predictor is not None else config.task_backend,
            "labeled": len(indices), "unknown": sum(s.task == "unknown" for s in samples)}
