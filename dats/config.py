"""Serializable configuration; real-data runs fail on missing required annotations."""
from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    budget: int = 50000
    seed: int = 42
    fine_clusters: int = 1000  # Total over all outer groups, not per group.
    min_per_fine: int = 2
    embedding_backend: str = "precomputed"  # precomputed | transformers
    embedding_model: str = "meta-llama/Llama-3.1-8B"
    task_backend: str = "precomputed"  # precomputed | rules | transformers
    task_model: str | None = None
    task_prompt_format: str = "chat"  # chat | alpaca
    relabel_tasks: bool = False
    relabel_inner_tasks: bool = False
    metric_backend: str = "precomputed"  # precomputed | transformers
    complexity_model: str = "hkust-nlp/deita-complexity-scorer"
    quality_model: str = "hkust-nlp/deita-quality-scorer"
    ifd_model: str = "meta-llama/Llama-3.1-8B"
    reuse_inner_labels: bool = False
    device: str = "cpu"
    precision: str = "float32"
    batch_size: int = 8
    max_length: int = 4096
    deita_max_length: int = 512
    weights: dict = field(default_factory=dict)

    def validate(self):
        for key in ("budget", "seed", "min_per_fine"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        for key in ("fine_clusters", "batch_size", "max_length", "deita_max_length"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        choices = {
            "embedding_backend": ("precomputed", "transformers"),
            "task_backend": ("precomputed", "rules", "transformers"),
            "task_prompt_format": ("alpaca", "chat"),
            "metric_backend": ("precomputed", "transformers"),
            "precision": ("float32", "float16", "bfloat16"),
        }
        for key, allowed in choices.items():
            if getattr(self, key) not in allowed:
                raise ValueError(f"{key} must be one of {allowed}")
        for backend, model in ((self.embedding_backend, self.embedding_model), (self.task_backend, self.task_model)):
            if backend == "transformers" and not model:
                raise ValueError("A model path is required for the transformers backend")
        if self.max_length < 8:
            raise ValueError("max_length must leave room for both context and response tokens")
        return self

    def to_dict(self):
        return asdict(self)
