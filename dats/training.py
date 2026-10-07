"""Validate classifier data and launch four-GPU full SFT through LLaMA-Factory."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import subprocess
from .data import TASKS, canonical_task, write_json
from .prompts import CLASSIFIER_INSTRUCTION


def read_training_records(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records:
        raise ValueError("Classifier training data is empty")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not all(isinstance(record.get(k), str) for k in ("instruction", "input", "output")):
            raise ValueError(f"Classifier row {index} must have string instruction/input/output fields")
        if record["instruction"] != CLASSIFIER_INSTRUCTION:
            raise ValueError(f"Classifier row {index} uses an instruction different from inference; update dats/prompts.py consistently")
        if not record["input"].strip() or canonical_task(record["output"]) not in TASKS:
            raise ValueError(f"Classifier row {index} has an empty input or invalid category")
    return records


def training_stats(path, records, max_samples=6000):
    return {"file_name": Path(path).name, "actual_records": len(records),
            "configured_max_samples": max_samples, "records_before_tokenization": min(len(records), max_samples),
            "label_counts": dict(sorted(Counter(record["output"] for record in records).items())),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def prepare_training(settings, root):
    from .launcher import resolve_path
    paths, training = settings["paths"], settings["training"]
    source = resolve_path(root, paths["classifier_data"])
    records = read_training_records(source)
    stats = training_stats(source, records, training["max_samples"])
    devices = training["visible_devices"]
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("Training devices must be a nonempty list of unique GPU indices")
    global_batch = len(devices) * training["per_device_train_batch_size"] * training["gradient_accumulation_steps"]
    if global_batch != training["global_batch_size"]:
        raise ValueError("Global batch size must equal GPUs * per-device batch * accumulation steps")
    destination = resolve_path(root, paths["training_work_dir"])
    dataset_dir = destination / "data"
    write_json(dataset_dir / "classifier_train.json", records)
    write_json(dataset_dir / "dataset_info.json", {
        "task_classifier": {"file_name": "classifier_train.json", "formatting": "alpaca",
                            "columns": {"prompt": "instruction", "query": "input", "response": "output"}}
    })
    options = {
        "model_name_or_path": settings["models"]["qwen_base"] or "SET_QWEN_BASE_MODEL_IN_RUN_PY",
        "stage": "sft", "do_train": True, "finetuning_type": "full",
        "dataset": "task_classifier", "dataset_dir": str(dataset_dir),
        "template": training["template"], "cutoff_len": training["max_length"],
        "max_samples": training["max_samples"],
        "output_dir": str(resolve_path(root, paths["classifier_checkpoint"])),
        "per_device_train_batch_size": training["per_device_train_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "learning_rate": training["learning_rate"], "num_train_epochs": training["epochs"],
        "lr_scheduler_type": "linear", "warmup_ratio": training["warmup_ratio"],
        "bf16": True, "gradient_checkpointing": True, "ddp_timeout": 180000,
        "logging_steps": 10, "save_strategy": "epoch", "save_only_model": True,
        "report_to": "none", "overwrite_output_dir": False,
        "deepspeed": str(resolve_path(root, paths["deepspeed_config"])),
    }
    import yaml
    config_path = destination / "qwen_full_sft.yaml"
    config_path.write_text(yaml.safe_dump(options, sort_keys=False), encoding="utf-8")
    stats.update({"finetuning_type": "full", "gpu_count": len(devices), "global_batch_size": global_batch,
                  "learning_rate": training["learning_rate"], "configured_epochs": training["epochs"],
                  "max_length": training["max_length"], "training_executed": False})
    write_json(destination / "training_manifest.json", stats)
    trainer = resolve_path(root, paths["llamafactory_dir"]) / "src/train.py"
    command = ["torchrun", "--nnodes=1", f"--nproc_per_node={len(devices)}", "--standalone", str(trainer), str(config_path)]
    return command, options, stats, destination


def launch_training(settings, root, dry_run=False):
    command, options, stats, destination = prepare_training(settings, root)
    if dry_run:
        return {"dry_run": True, "command": command, "training": stats,
                "qwen_base_configured": bool(settings["models"]["qwen_base"])}
    if not settings["models"]["qwen_base"]:
        raise ValueError("Set SETTINGS['models']['qwen_base'] in run.py before training")
    trainer = Path(command[-2])
    if not trainer.is_file():
        raise ValueError("Set paths.llamafactory_dir to an installed LLaMA-Factory checkout")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in settings["training"]["visible_devices"])
    subprocess.run(command, cwd=root, env=environment, check=True)
    stats["training_executed"] = True
    write_json(destination / "training_manifest.json", stats)
    return stats
