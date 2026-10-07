"""Deterministic synthetic fixtures; none of these scores are model evaluations."""
import numpy as np

PROMPTS = {
    "math": "Solve the algebra equation {n} * x + 4 = 20 and calculate x.",
    "coding": "Write a Python function to sort {n} records and explain the algorithm.",
    "stem": "Explain photosynthesis in biology and give {n} observations about molecules.",
    "humanities": "Discuss history and philosophy during the Renaissance in {n} points.",
    "reasoning": "Solve this logic puzzle using deductive reasoning with {n} clues.",
    "writing": "Compose a poem and a short story about {n} travelers.",
    "roleplay": "Pretend to be a guide and stay in character for {n} turns.",
    "extraction": "Extract {n} named entities and return only JSON with the requested fields.",
}


def make_records(per_task=8, annotated=False):
    rng = np.random.default_rng(42)
    rows = []
    for task_index, (task, prompt) in enumerate(PROMPTS.items()):
        for j in range(per_task):
            row = {"id": f"demo-{task}-{j:02d}", "synthetic_fixture": True,
                   "dialogs": [{"role": "user", "content": prompt.format(n=j+2) + " " + "Please explain carefully. " * (j % 3)},
                               {"role": "assistant", "content": "Here is an illustrative response. " + "An example explains the answer in detail. " * (j+1)}]}
            if annotated:
                vector = np.zeros(12)
                vector[task_index] = 1.0
                vector += rng.normal(0, 0.03, 12)
                row.update({"emb": vector.tolist(),
                            "alpaca_prediction": {"category": task},
                            "annotation": {"deita": {"complexity_scores": [2+j*0.2], "quality_scores": [3+(j%3)*0.3]}},
                            "ifd_score": {"content": 0.2 + j*0.08}})
            rows.append(row)
    return rows


def synthetic_evaluator(records, round_index):
    # Fixed synthetic rewards exercise positive and negative feedback paths only.
    # They intentionally make no claim about trained-model accuracy.
    return {task: 0.5 + (round_index+1) * (0.02 if i % 2 == 0 else -0.01)
            for i, task in enumerate(PROMPTS)}
