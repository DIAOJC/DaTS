"""Commands driven exclusively by the root run.py SETTINGS dictionary."""
import argparse
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import sys
from .config import Config
from .data import load_samples, write_json, write_jsonl
from .clustering import label_outer, cluster_inner
from .embeddings import build_embeddings
from .labeling import label_tasks
from .metrics import METRICS, compute_metrics
from .selection import select_samples
from .feedback import update_weights
from .pipeline import prepare, export_result, annotated_record, read_trace


def resolve_path(root, value):
    if not value:
        raise ValueError("A required path is missing from run.py SETTINGS")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_reference(root, value):
    if value and (Path(value).expanduser().is_absolute() or (Path(root) / value).exists()):
        return str(resolve_path(root, value))
    return value


def pipeline_config(settings, root, budget=None, cached=False):
    selection, runtime, models = settings["selection"], settings["runtime"], settings["models"]
    initial = settings["paths"]["initial_weights"]
    config = Config(
        budget=selection["budget"] if budget is None else budget,
        fine_clusters=selection["fine_clusters"], min_per_fine=selection["min_per_fine"],
        task_backend="precomputed" if cached else selection["task_backend"],
        task_model=str(resolve_path(root, settings["paths"]["classifier_checkpoint"])),
        task_prompt_format="chat", relabel_tasks=False if cached else selection["relabel_outer_tasks"],
        relabel_inner_tasks=False if cached else selection["relabel_inner_tasks"],
        embedding_backend="precomputed" if cached else selection["embedding_backend"],
        embedding_model=model_reference(root, models["llama"]),
        metric_backend="precomputed" if cached else selection["metric_backend"],
        ifd_model=model_reference(root, models["llama"]),
        complexity_model=model_reference(root, models["deita_complexity"]),
        quality_model=model_reference(root, models["deita_quality"]),
        device=runtime["device"], precision=runtime["precision"], batch_size=runtime["batch_size"],
        max_length=runtime["max_length"], seed=runtime["seed"],
        deita_max_length=runtime["deita_max_length"],
        reuse_inner_labels=selection["reuse_inner_labels"],
        weights=read_json(resolve_path(root, initial)) if initial else {},
    )
    return config.validate()


def execute(settings, root, args):
    paths = settings["paths"]
    if args.command == "train":
        from .training import launch_training
        return launch_training(settings, root, args.dry_run)
    if args.command == "feedback":
        folder = resolve_path(root, paths["feedback_run_dir"])
        evaluations = read_json(resolve_path(root, paths["feedback_evaluations"]))
        trace_path = folder / "selection_trace.jsonl"
        digest = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        if evaluations.get("selection_trace_sha256", digest) != digest:
            raise ValueError("Evaluation and selected-run checksums do not match")
        updated, details = update_weights(read_json(folder/"weights.json"), read_trace(trace_path),
                                          evaluations["before"], evaluations["after"], settings["feedback"]["learning_rate"])
        write_json(folder/"weights.updated.json", updated)
        write_json(folder/"feedback_update.json", {"selection_trace_sha256": digest, "updates": details})
        return {"weights_file": str(folder/"weights.updated.json"), "updated_tasks": sum(x["updated"] for x in details.values())}
    config = pipeline_config(settings, root, args.budget, cached=args.command == "cached")
    output = resolve_path(root, paths["output_dir"])
    if args.command in ("run", "cached"):
        samples = load_samples(resolve_path(root, paths["pool"]))
        embeddings, metrics, metadata = prepare(samples, config)
        result = select_samples(samples, embeddings, metrics, config)
        return export_result(output, samples, result, config, metadata)
    work = resolve_path(root, paths["work_dir"])
    filenames = ["01_outer.jsonl", "02_embedded.jsonl", "03_clustered.jsonl", "04_labeled.jsonl", "05_scored.jsonl"]
    stages = ["label-outer", "embed", "cluster", "label-inner", "score", "select"]
    index = stages.index(args.command)
    source = resolve_path(root, paths["pool"]) if index == 0 else work/filenames[index-1]
    samples = load_samples(source)
    embeddings = None
    if args.command == "label-outer":
        info = label_tasks(samples, config)
        info["groups"] = label_outer(samples, config)
    elif args.command == "embed":
        embeddings, info = build_embeddings(samples, config)
    elif args.command == "cluster":
        embeddings, _ = build_embeddings(samples, replace(config, embedding_backend="precomputed"))
        info = cluster_inner(samples, embeddings, config)
    elif args.command == "label-inner":
        if config.relabel_inner_tasks:
            info = label_tasks(samples, replace(config, relabel_tasks=True))
        else:
            info = {"backend": "reuse_outer_task_predictions"}
    elif args.command == "score":
        metrics, info = compute_metrics(samples, config)
        write_jsonl(work/filenames[index], (dict(annotated_record(s), metrics=dict(zip(METRICS[:3], values.tolist())))
                                          for s, values in zip(samples, metrics)))
        write_json(work/f"{args.command}.metadata.json", info)
        return info
    else:
        embeddings, emb_info = build_embeddings(samples, replace(config, embedding_backend="precomputed"))
        metrics, metric_info = compute_metrics(samples, replace(config, metric_backend="precomputed"))
        result = select_samples(samples, embeddings, metrics, config)
        return export_result(output, samples, result, config, {"embedding": emb_info, "metrics": metric_info})
    write_jsonl(work/filenames[index], (annotated_record(s, None if embeddings is None else embeddings[i]) for i, s in enumerate(samples)))
    write_json(work/f"{args.command}.metadata.json", info)
    return info


def main(settings, root, argv=None):
    parser = argparse.ArgumentParser(description="DaTS: configure all paths and models in run.py")
    parser.add_argument("command", choices=["cached", "run", "label-outer", "embed", "cluster",
                                           "label-inner", "score", "select", "feedback", "train"])
    parser.add_argument("--budget", type=int, help="Optional selection-budget override")
    parser.add_argument("--dry-run", action="store_true", help="Generate and inspect training configuration without launching GPUs")
    args = parser.parse_args(argv)
    try:
        if args.dry_run and args.command != "train":
            raise ValueError("--dry-run applies only to train")
        result = execute(settings, Path(root), args)
        print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
    except (ValueError, TypeError, KeyError, OSError, ImportError, RuntimeError) as exc:
        print(f"dats: {exc}", file=sys.stderr)
        return 2
    return 0
