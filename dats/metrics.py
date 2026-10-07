"""C/Q/F inputs and task-wise normalization, including dynamic distinctiveness V."""
import numpy as np

METRICS = ("complexity", "quality", "ifd", "distinctiveness")


def compute_metrics(samples, config, scorer=None):
    """Use cached DEITA/IFD, or inject scorer(samples) -> Nx3 [C,Q,F]."""
    if scorer is not None:
        values = np.asarray(scorer(samples), dtype=float)
        backend = "callback"
    elif config.metric_backend == "transformers":
        from .scoring import score_with_models
        values = score_with_models(samples, config)
        backend = "deita_and_llama_ifd"
    else:
        values = []
        for s in samples:
            missing = [key for key in METRICS[:3] if s.metrics.get(key) is None]
            if missing:
                raise ValueError(f"{s.id}: missing {missing}; supply DEITA/IFD scores or a scorer callback")
            values.append([s.metrics[key] for key in METRICS[:3]])
        values = np.asarray(values, dtype=float)
        backend = "precomputed"
    if values.shape != (len(samples), 3) or not np.isfinite(values).all():
        raise ValueError("Metrics must be a finite N x 3 matrix in [complexity, quality, ifd] order")
    if np.any(values[:, 2] < 0):
        raise ValueError("IFD is a perplexity ratio and must be nonnegative")
    return values, {"backend": backend, "metric_order": list(METRICS),
                    "distinctiveness": "1 - max cosine to selected items in the same fine cluster"}


class TaskNormalizer:
    """Eq. (7). C/Q/F statistics are fixed; V moments track all samples by task."""
    def __init__(self, values, tasks):
        self.tasks, self.codes = np.unique(tasks, return_inverse=True)
        self.counts = np.bincount(self.codes).astype(float)
        self.raw = np.column_stack([values, np.ones(len(values))])
        self.static = np.zeros_like(values)
        self.statistics = {}
        for code, task in enumerate(self.tasks):
            mask = self.codes == code
            mean, std = values[mask].mean(0), values[mask].std(0)
            scale = np.where(std > 1e-12, std, 1.0)
            self.static[mask] = (values[mask] - mean) / scale
            self.statistics[str(task)] = {"mean_CQF": mean.tolist(), "std_CQF": std.tolist(), "count": int(mask.sum())}
        self.vsum = self.counts.copy()
        self.vsqsum = self.counts.copy()

    def normalized(self, indices):
        indices = np.asarray(indices, dtype=int)
        codes = self.codes[indices]
        mean = self.vsum[codes] / self.counts[codes]
        variance = np.maximum(0.0, self.vsqsum[codes] / self.counts[codes] - mean**2)
        std = np.sqrt(variance)
        # Numerically constant V columns produce exactly zero normalized V.
        z = np.divide(self.raw[indices, 3] - mean, std, out=np.zeros(len(indices)), where=std > 1e-7)
        return np.column_stack([self.static[indices], z])

    def update_distinctiveness(self, indices, values):
        indices = np.asarray(indices, dtype=int)
        values = np.asarray(values, dtype=float)
        old = self.raw[indices, 3]
        codes = self.codes[indices]
        self.vsum += np.bincount(codes, weights=values-old, minlength=len(self.tasks))
        self.vsqsum += np.bincount(codes, weights=values**2-old**2, minlength=len(self.tasks))
        self.raw[indices, 3] = values
