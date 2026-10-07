#!/usr/bin/env python3
"""Single configuration and launch entry point. Edit SETTINGS for your machine."""
from pathlib import Path

SETTINGS = {
    "paths": {
        "pool": "data/example_pool.jsonl",
        "classifier_data": "data/private/alpaca_format.jsonl",
        "classifier_checkpoint": "outputs/qwen_classifier",
        "llamafactory_dir": "external/LLaMA-Factory",
        "training_work_dir": "outputs/classifier_training",
        "deepspeed_config": "configs/deepspeed_zero3.json",
        "work_dir": "outputs/work",
        "output_dir": "outputs/selection",
        "demo_dir": "outputs/demo",
        "feedback_evaluations": "data/feedback.example.json",
        "feedback_run_dir": "outputs/selection",
        "initial_weights": None,
    },
    "models": {
        # Required for classifier training. Use the exact Qwen base you trained.
        "qwen_base": None,
        "llama": "meta-llama/Llama-3.1-8B",  # Shared by embeddings and IFD.
        "deita_complexity": "hkust-nlp/deita-complexity-scorer",
        "deita_quality": "hkust-nlp/deita-quality-scorer",
    },
    "runtime": {
        "device": "cuda:0", "precision": "bfloat16", "batch_size": 4,
        "max_length": 4096, "deita_max_length": 512, "seed": 42,
    },
    "selection": {
        "budget": 50000, "fine_clusters": 1000, "min_per_fine": 2,
        "task_backend": "transformers", "embedding_backend": "transformers",
        "metric_backend": "transformers",
        "relabel_outer_tasks": True, "relabel_inner_tasks": True,
        "reuse_inner_labels": False,
    },
    "training": {
        # Match this to the selected Qwen family, e.g. qwen3_nothink for Qwen3.
        "template": "qwen",
        "visible_devices": [0, 1, 2, 3],
        "global_batch_size": 128, "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "learning_rate": 5e-6, "epochs": 3.0,
        "max_length": 4096, "max_samples": 6000, "warmup_ratio": 0.03,
    },
    "feedback": {"learning_rate": 0.1},
}


if __name__ == "__main__":
    from dats.launcher import main
    raise SystemExit(main(SETTINGS, Path(__file__).resolve().parent))
