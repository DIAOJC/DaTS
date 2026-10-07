"""Optional feedback, implementing Eq. (9) without owning a training framework."""
from collections import defaultdict
import numpy as np
from .aggregation import resolve_weights
from .metrics import METRICS


def project_simplex(vector):
    """Euclidean projection onto nonnegative weights summing to one."""
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (4,) or not np.isfinite(vector).all():
        raise ValueError("Projection expects four finite values")
    ordered = np.sort(vector)[::-1]
    offsets = (np.cumsum(ordered) - 1) / np.arange(1, 5)
    rho = np.flatnonzero(ordered - offsets > 0)[-1]
    return np.maximum(vector - offsets[rho], 0)


def update_weights(weights, trace, before, after, learning_rate=0.1):
    """Require actual evaluation numbers from the caller, never invent rewards.

    Uses the normalized features saved at each selection decision, before adding
    the chosen sample to S; this prevents its diversity from collapsing to zero.
    Missing evaluations or empty selected categories leave their weights unchanged.
    """
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    for scores in (before, after):
        if not isinstance(scores, dict) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in scores.values()):
            raise ValueError("Evaluation scores must map task names to finite numbers")
    resolved = resolve_weights(weights.keys(), weights)
    by_task = defaultdict(list)
    for entry in trace:
        vector = [entry["normalized_metrics"][key] for key in METRICS]
        if not np.isfinite(vector).all():
            raise ValueError("Feedback features must be finite")
        by_task[entry["task"]].append(vector)
    unknown = set(by_task) - set(resolved)
    if unknown:
        raise ValueError(f"Missing weights for selected task(s): {sorted(unknown)}")
    updated, details = {}, {}
    for task, current in resolved.items():
        if not by_task[task] or task not in before or task not in after:
            updated[task] = current.tolist()
            details[task] = {"updated": False, "reason": "no selected samples or missing evaluation"}
            continue
        mean = np.mean(by_task[task], axis=0)
        delta = float(after[task] - before[task])
        next_weights = project_simplex(current + learning_rate * delta * mean)
        updated[task] = next_weights.tolist()
        details[task] = {"updated": True, "delta": delta, "selected_count": len(by_task[task]),
                         "mean_normalized_metrics": mean.tolist(), "before": current.tolist(), "after": next_weights.tolist()}
    return updated, details


def feedback_loop(samples, embeddings, metrics, config, evaluator, baseline, rounds=6, learning_rate=0.1):
    """Select -> user training/evaluation callback -> update -> repeat.

    evaluator(selected_original_records, round_index) -> {task: score}.
    The callback owns model/checkpoint state. After the last update we reselect
    once using the final weights, but do not claim that final set was evaluated.
    """
    from .selection import select_samples
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    weights = config.weights
    previous = baseline
    history = []
    for iteration in range(rounds):
        result = select_samples(samples, embeddings, metrics, config, weights)
        scores = evaluator([samples[i].raw for i in result.indices], iteration)
        weights, update = update_weights(result.weights, result.trace, previous, scores, learning_rate)
        history.append({"round": iteration + 1, "selected_ids": [samples[i].id for i in result.indices],
                        "before": previous, "after": scores, "update": update, "weights": weights})
        previous = scores
    final = select_samples(samples, embeddings, metrics, config, weights)
    return final, history
