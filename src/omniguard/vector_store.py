from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


class VectorStore:
    """In- memory store of vectors with KNN search and pickle persistence".

    Internally:
    - 'self.vectors' is a NumPy array of shape(N, dim) - all vectors stacked.
    - 'self.metadatas' is a parallel list of dicts - one per vector.
    - Index i in vectors matches index i in metadatas.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self.metadatas: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.metadatas)

    def add(self, vector: np.ndarray, metadata: dict[str, Any]) -> None:
        """Add one vector + its metadata."""
        if vector.shape != (self.dim,):
            raise ValueError(f"vector must have shape ({self.dim},), got {vector.shape}")
        self.vectors = np.vstack([self.vectors, vector.astype(np.float32)])
        self.metadatas.append(metadata)

    def add_batch(self, vectors: np.ndarray, metadatas: list[dict[str, Any]]) -> None:
        """Add many vectors at once (much faster than calling add() in a loop)."""
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"vectors must have shape (N, {self.dim}), got {vectors.shape}")
        if len(vectors) != len(metadatas):
            raise ValueError("vectors and metadatas must have the same length")
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
        self.metadatas.extend(metadatas)

    def search(self, query: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k closest entries to query, sorted by distance ascending.

        Each result dict: {"distance": float, "metadata": {...}, "index": int}.
        Distance = 1 - cosine_similarity, so smaller = more similar.
        """
        if len(self) == 0:
            return []
        if query.shape != (self.dim,):
            raise ValueError(f"query must have shape ({self.dim},), got {query.shape}")

        # Cosine similarity for all stored vectors at once.
        query_norm = np.linalg.norm(query)
        store_norms = np.linalg.norm(self.vectors, axis=1)
        dots = self.vectors @ query  # shape (N,)
        cosines = dots / (store_norms * query_norm + 1e-12)
        distances = 1.0 - cosines

        # Top-k smallest distances.
        k = min(k, len(self))
        top_idx = np.argsort(distances)[:k]

        return [
            {"distance": float(distances[i]), "metadata": self.metadatas[i], "index": int(i)}
            for i in top_idx
        ]

    def save(self, path: str | Path) -> None:
        """Pickle the whole store to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"dim": self.dim, "vectors": self.vectors, "metadatas": self.metadatas}, f)

    @classmethod
    def load(cls, path: str | Path) -> VectorStore:
        """Load a previously saved store from disk."""
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        store = cls(dim=data["dim"])
        store.vectors = data["vectors"]
        store.metadatas = data["metadatas"]
        return store
