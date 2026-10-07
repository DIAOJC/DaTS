"""Sentence representations with explicit real-data and offline-demo backends."""
import numpy as np


def validate_embeddings(values, count):
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Embeddings must be numeric vectors of equal dimension") from exc
    if array.ndim != 2 or array.shape[0] != count or array.shape[1] == 0 or not np.isfinite(array).all():
        raise ValueError("Embeddings must be a finite N x D matrix with D > 0")
    if np.any(np.linalg.norm(array, axis=1) == 0):
        raise ValueError("Zero embeddings have undefined cosine similarity; recompute these embeddings")
    return array


def build_embeddings(samples, config, encoder=None):
    if encoder is not None:
        values = encoder([s.text for s in samples])
        backend = "callback"
    elif config.embedding_backend == "precomputed":
        for s in samples:
            if s.embedding is None:
                raise ValueError(f"{s.id}: missing emb/embedding; supply embeddings or set embedding_backend")
        values = [s.embedding for s in samples]
        backend = "precomputed"
    elif config.embedding_backend == "transformers":
        from .models import TransformersEncoder, release_model
        adapter = TransformersEncoder(config)
        try:
            values = adapter([s.text for s in samples])
        finally:
            release_model(adapter)
        backend = "transformers"
    else:
        raise ValueError("Unknown embedding backend")
    values = validate_embeddings(values, len(samples))
    return values, {"backend": backend, "dimensions": values.shape[1]}
