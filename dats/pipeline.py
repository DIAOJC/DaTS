"""Compose independently callable stages and export auditable artifacts."""
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import numpy as np
from .clustering import label_outer, cluster_inner, fine_groups
from .data import write_json, write_jsonl
from .embeddings import build_embeddings
from .labeling import label_tasks
from .metrics import compute_metrics
from .selection import select_samples


def prepare(samples, config, *, encoder=None, task_predictor=None, scorer=None):
    config.validate()
    if config.budget > len(samples):
        raise ValueError(f"budget={config.budget} exceeds input size {len(samples)}")
    metadata = {}
    metadata["outer_task_labeling"] = label_tasks(samples, config, task_predictor)
    metadata["outer_labeling"] = label_outer(samples, config)
    embeddings, metadata["embedding"] = build_embeddings(samples, config, encoder)
    metadata["inner_clustering"] = cluster_inner(samples, embeddings, config)
    if config.relabel_inner_tasks:
        metadata["inner_task_labeling"] = label_tasks(samples, replace(config, relabel_tasks=True), task_predictor)
    else:
        metadata["inner_task_labeling"] = {"backend": "reuse_outer_task_predictions"}
    metrics, metadata["metrics"] = compute_metrics(samples, config, scorer)
    return embeddings, metrics, metadata


def run(samples, config, **providers):
    embeddings, metrics, metadata = prepare(samples, config, **providers)
    result = select_samples(samples, embeddings, metrics, config)
    return result, metadata


def annotated_record(sample, embedding=None):
    row = dict(sample.raw)
    row["id"] = sample.id
    row["outer_label"] = sample.outer_label
    row["inner_label"] = sample.inner_label
    if sample.task_is_provided:
        row["task_label"] = sample.task
    if embedding is not None:
        row["emb"] = np.asarray(embedding).tolist()
    return row


def build_hierarchy(samples, result):
    selected = set(result.indices)
    quotas = {(row["outer_label"], row["inner_label"]): row for row in result.quotas}
    parents = defaultdict(list)
    for key, indices in fine_groups(samples).items():
        category_indices = defaultdict(list)
        for index in indices:
            category_indices[samples[index].task].append(index)
        categories = []
        for task, rows in sorted(category_indices.items()):
            categories.append({"task": task, "available": len(rows),
                               "selected": sum(i in selected for i in rows),
                               "mean_CQF": result.final_metrics[rows, :3].mean(0).tolist()})
        parents[key[0]].append({"inner_label": key[1], "available": len(indices),
                                "selected": sum(i in selected for i in indices),
                                "budget": quotas[key]["budget"], "tasks": categories})
    return [{"outer_label": parent, "available": sum(c["available"] for c in children),
             "selected": sum(c["selected"] for c in children), "children": children}
            for parent, children in sorted(parents.items())]


def export_result(output_dir, samples, result, config, metadata, history=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "selected.jsonl", (samples[i].raw for i in result.indices))
    write_json(output / "selected_ids.json", [samples[i].id for i in result.indices])
    write_jsonl(output / "selection_trace.jsonl", result.trace)
    selected = set(result.indices)
    write_jsonl(output / "assignments.jsonl", ({"id": s.id, "outer_label": s.outer_label,
                "inner_label": s.inner_label, "task_label": s.task, "selected": i in selected}
                for i, s in enumerate(samples)))
    write_json(output / "hierarchy.json", build_hierarchy(samples, result))
    write_json(output / "quotas.json", result.quotas)
    write_json(output / "weights.json", result.weights)
    write_json(output / "normalization.json", result.normalization)
    write_json(output / "config.json", config.to_dict())
    if history is not None:
        write_json(output / "feedback_history.json", history)
    trace_digest = hashlib.sha256((output / "selection_trace.jsonl").read_bytes()).hexdigest()
    summary = {"input_count": len(samples), "requested_budget": config.budget,
               "selected_count": len(result.indices), "unique_selected_count": len(set(result.indices)),
               "outer_clusters": len(set(s.outer_label for s in samples)), "fine_clusters": len(result.quotas),
               "pool_tasks": dict(Counter(s.task for s in samples)),
               "selected_tasks": dict(Counter(samples[i].task for i in result.indices)),
               "minimum_relaxed_in_outer_groups": sorted({q["outer_label"] for q in result.quotas if q["minimum_relaxed"]}),
               "backends": metadata, "selection_trace_sha256": trace_digest,
               "feedback_rounds": len(history) if history is not None else 0,
               "final_selection_evaluated": False if history is not None else None}
    write_json(output / "summary.json", summary)
    return summary


def read_trace(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
