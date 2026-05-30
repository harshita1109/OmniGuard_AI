from __future__ import annotations

import numpy as np

from omniguard.embedder import Embedder
from omniguard.vector_store import VectorStore


class Detector:
    """Anomaly detector that uses kNN search over a VectorStore.


    Score = mean distance to the top-k nearest baseline vectors.
    Higher score = farther from anything we've seen = more anomalous.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore | None = None,
        threshold: float = 0.35,
        k: int = 5,
    ) -> None:
        self.embedder = embedder
        self.store = store if store is not None else VectorStore(dim=384)
        self.threshold = threshold
        self.k = k

    def add_baseline(self, texts: list[str]) -> None:
        """Embed baseline texts and add them to the vector store."""
        if not texts:
            raise ValueError("baseline must contain at least one text")
        vectors = self.embedder.embed_batch(texts)
        metadatas = [{"text": t} for t in texts]
        self.store.add_batch(vectors, metadatas)

    def check(self, text: str) -> dict:
        """Score a new text via kNN. Returns is_anomaly, score, neighbors."""
        if len(self.store) == 0:
            raise RuntimeError("call add_baseline() before check()")

        query = self.embedder.embed(text)
        results = self.store.search(query, k=self.k)

        score = float(np.mean([r["distance"] for r in results]))
        neighbors = [r["metadata"]["text"] for r in results]

        return {
            "is_anomaly": score > self.threshold,
            "score": score,
            "neighbors": neighbors,
        }
