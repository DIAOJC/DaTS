"""Exact two-level budget allocation and dynamic cosine-diversity greedy selection."""
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from .aggregation import aggregate, resolve_weights
from .budget import allocate_proportional
from .clustering import fine_groups
from .metrics import METRICS, TaskNormalizer


@dataclass
class SelectionResult:
    indices: list
    trace: list
    quotas: list
    weights: dict
    normalization: dict
    final_metrics: np.ndarray


def select_samples(samples, embeddings, metrics, config, weights=None):
    if config.budget > len(samples):
        raise ValueError(f"budget={config.budget} exceeds the {len(samples)} available samples")
    tasks = [s.task for s in samples]
    weights = resolve_weights(tasks, config.weights if weights is None else weights)
    row_weights = np.stack([weights[task] for task in tasks])
    groups = fine_groups(samples)
    parents = defaultdict(list)
    for key in groups:
        parents[key[0]].append(key)
    parent_sizes = [sum(len(groups[key]) for key in keys) for keys in parents.values()]
    outer_budgets = allocate_proportional(parent_sizes, config.budget)
    quotas = []
    for (parent, keys), budget in zip(parents.items(), outer_budgets):
        sizes = [len(groups[key]) for key in keys]
        child_budgets = allocate_proportional(sizes, int(budget), config.min_per_fine)
        relaxed = sum(min(size, config.min_per_fine) for size in sizes) > budget
        for key, size, count in zip(keys, sizes, child_budgets):
            quotas.append({"outer_label": parent, "inner_label": key[1], "available": size,
                           "outer_budget": int(budget), "budget": int(count), "minimum_relaxed": bool(relaxed)})
    normalizer = TaskNormalizer(metrics, tasks)
    unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    selected, trace = [], []
    # Stable traversal makes decision-time feedback features reproducible.
    for quota in quotas:
        indices = np.asarray(groups[(quota["outer_label"], quota["inner_label"])], dtype=int)
        available = np.ones(len(indices), dtype=bool)
        max_similarity = np.full(len(indices), -np.inf)
        for _ in range(quota["budget"]):
            normalized = normalizer.normalized(indices)
            contributions = aggregate(normalized, row_weights[indices])
            scores = np.where(available, contributions["total"], -np.inf)
            local = int(np.argmax(scores))
            index = int(indices[local])
            sample = samples[index]
            trace.append({"id": sample.id, "index": index, "task": sample.task,
                          "outer_label": sample.outer_label, "inner_label": sample.inner_label,
                          "raw_metrics": dict(zip(METRICS, normalizer.raw[index].tolist())),
                          "normalized_metrics": dict(zip(METRICS, normalized[local].tolist())),
                          "weights": weights[sample.task].tolist(),
                          "aggregation": {key: float(value[local]) for key, value in contributions.items()}})
            selected.append(index)
            available[local] = False
            similarities = np.clip(unit[indices] @ unit[index], -1.0, 1.0)
            max_similarity = np.maximum(max_similarity, similarities)
            normalizer.update_distinctiveness(indices, 1.0 - max_similarity)
    if len(selected) != config.budget or len(set(selected)) != len(selected):
        raise RuntimeError("Selection invariant violated: incorrect count or duplicate indices")
    return SelectionResult(selected, trace, quotas, {key: value.tolist() for key, value in weights.items()},
                           normalizer.statistics, normalizer.raw.copy())
