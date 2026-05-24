from __future__ import annotations

import numpy as np

from omniguard.embedder import Embedder


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm)


class Detector:
    """Centroid-based anomaly detector."""

    def __init__(self, embedder: Embedder, threshold: float = 0.35) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self.baseline_vectors: np.ndarray | None = None
        self.centroid: np.ndarray | None = None

    def add_baseline(self, texts: list[str]) -> None:
        """Embed baseline texts, store them, and compute the centroid."""
        if not texts:
            raise ValueError("baseline must contain at least one text")
        self.baseline_vectors = self.embedder.embed_batch(texts)
        self.centroid = np.mean(self.baseline_vectors, axis=0)

    def check(self, text: str) -> dict:
        """Score a new text. Returns {'is_anomaly': bool, 'score': float}."""
        if self.centroid is None:
            raise RuntimeError("call add_baseline() before check()")
        vec = self.embedder.embed(text)
        score = 1.0 - _cosine(vec, self.centroid)
        return {"is_anomaly": score > self.threshold, "score": score}
