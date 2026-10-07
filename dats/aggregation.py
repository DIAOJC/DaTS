"""Hierarchical metric aggregation, algebraically identical to paper Eq. (8)."""
import numpy as np
from .data import canonical_task
from .metrics import METRICS


def weight_vector(value):
    if isinstance(value, dict):
        if set(value) != set(METRICS):
            raise ValueError(f"Weight keys must be exactly {METRICS}")
        value = [value[key] for key in METRICS]
    array = np.asarray(value, dtype=float)
    if array.shape != (4,) or not np.isfinite(array).all() or np.any(array < 0) or array.sum() <= 0:
        raise ValueError("Weights must be four finite nonnegative numbers with positive sum")
    return array / array.sum()


def resolve_weights(tasks, configured=None):
    configured = configured or {}
    normalized = {}
    for key, value in configured.items():
        task = key if key == "default" else canonical_task(key)
        if task == "unknown" and key != "unknown":
            raise ValueError(f"Unknown task in weights: {key}")
        if task in normalized:
            raise ValueError(f"Duplicate weight category: {task}")
        normalized[task] = weight_vector(value)
    default = normalized.get("default", np.full(4, 0.25))
    return {str(task): normalized.get(str(task), default).copy() for task in sorted(set(tasks))}


def aggregate(normalized_metrics, weights):
    """C/Q -> sample quality; V -> differences; F -> model difficulty -> total.

    Report weighted contributions rather than applying a second set of weights.
    The three parent contributions therefore sum exactly to w dot x.
    """
    contributions = np.asarray(normalized_metrics) * np.asarray(weights)
    return {
        "sample_quality": contributions[..., 0] + contributions[..., 1],
        "inter_sample_difference": contributions[..., 3],
        "model_difficulty": contributions[..., 2],
        "total": contributions.sum(axis=-1),
    }
