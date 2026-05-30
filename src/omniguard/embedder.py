from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Converts text into 384-dim vectors using all-MiniLM-L6-v2.

    The model is loaded lazily on first use so importing this module is cheap,
    and tests that mock the embedder pay zero startup cost.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None  # lazy

    def _load(self) -> SentenceTransformer:
        """Load the model the first time it's needed, then reuse it."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string. Returns shape (384,)."""
        model = self._load()
        return model.encode(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed many strings at once. Returns shape (len(texts),384)."""
        model = self._load()
        return model.encode(texts)
