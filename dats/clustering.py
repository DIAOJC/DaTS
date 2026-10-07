"""Qwen eight-category groups -> semantic KMeans within each category."""
from collections import defaultdict
import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from .budget import allocate_proportional
from .data import TASKS, canonical_task


def group_indices(labels):
    groups = defaultdict(list)
    for index, label in enumerate(labels):
        groups[str(label)].append(index)
    return dict(sorted(groups.items()))


def kmeans_labels(matrix, count, seed):
    count = min(count, matrix.shape[0])
    if count == 1:
        return np.zeros(matrix.shape[0], dtype=int)
    with warnings.catch_warnings():
        # Identical rows may yield fewer clusters: actual groups are enumerated below.
        warnings.simplefilter("ignore", ConvergenceWarning)
        return KMeans(n_clusters=count, n_init=10, random_state=seed).fit_predict(matrix)


def label_outer(samples, config):
    """Use Qwen task predictions, never fit an additional outer clustering model."""
    for sample in samples:
        if config.reuse_inner_labels:
            existing = canonical_task(sample.outer_label)
            if existing not in TASKS:
                raise ValueError(f"{sample.id}: cached outer_label must be one of the eight task categories")
            sample.outer_label = existing
        else:
            if not sample.task_is_provided or sample.task not in TASKS:
                raise ValueError(f"{sample.id}: an eight-category task prediction is required before grouping")
            sample.outer_label = sample.task
    return {"backend": "task_labels", "categories": list(TASKS),
            "groups": len(set(s.outer_label for s in samples))}


def cluster_inner(samples, embeddings, config):
    groups = group_indices([s.outer_label for s in samples])
    if config.reuse_inner_labels:
        if any(s.inner_label is None for s in samples):
            raise ValueError("reuse_inner_labels requires an inner_label/sub_label for every sample")
        return {"backend": "precomputed", "clusters": len({(s.outer_label, s.inner_label) for s in samples})}
    # At least one child per nonempty outer group; at most one per sample.
    total = min(len(samples), max(config.fine_clusters, len(groups)))
    counts = allocate_proportional([len(g) for g in groups.values()], total, minimum=1)
    actual = 0
    for group_number, ((_, indices), count) in enumerate(zip(groups.items(), counts)):
        labels = kmeans_labels(embeddings[indices], int(count), config.seed + group_number)
        actual += len(set(labels))
        for index, label in zip(indices, labels):
            samples[index].inner_label = str(int(label))
    return {"backend": "kmeans", "clusters": actual, "requested_clusters": config.fine_clusters,
            "allocated_clusters": total}


def fine_groups(samples):
    groups = defaultdict(list)
    for i, sample in enumerate(samples):
        if sample.outer_label is None or sample.inner_label is None:
            raise ValueError("Clustering labels must be populated before selection")
        # The same local inner ID in different parents must never be merged.
        groups[(sample.outer_label, sample.inner_label)].append(i)
    return dict(sorted(groups.items()))
