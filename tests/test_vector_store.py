# tests/test_vector_store.py
# Phase 3.3 — tests for the custom VectorStore.

from pathlib import Path

import numpy as np

from omniguard.vector_store import VectorStore


def _unit_vec(values: list[float]) -> np.ndarray:
    """Helper: make a small float32 vector for tests."""
    return np.array(values, dtype=np.float32)


def test_add_and_search() -> None:
    """Adding 3 vectors and searching with one of them returns it first."""
    vs = VectorStore(dim=4)
    vs.add(_unit_vec([1.0, 0.0, 0.0, 0.0]), {"text": "a"})
    vs.add(_unit_vec([0.9, 0.1, 0.0, 0.0]), {"text": "b"})  # close to a
    vs.add(_unit_vec([0.0, 0.0, 1.0, 0.0]), {"text": "c"})  # far from a

    results = vs.search(_unit_vec([1.0, 0.0, 0.0, 0.0]), k=3)

    assert len(results) == 3
    assert results[0]["metadata"]["text"] == "a"
    assert results[0]["distance"] < 0.01
    assert results[1]["metadata"]["text"] == "b"
    assert results[2]["metadata"]["text"] == "c"


def test_batch_add() -> None:
    """add_batch adds many vectors in one call; len reflects total count."""
    vs = VectorStore(dim=3)
    vectors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    metadatas = [{"i": 0}, {"i": 1}, {"i": 2}]
    vs.add_batch(vectors, metadatas)

    assert len(vs) == 3


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Save a store to disk, load it back, search produces the same result."""
    vs = VectorStore(dim=4)
    vs.add(_unit_vec([1.0, 0.0, 0.0, 0.0]), {"text": "a"})
    vs.add(_unit_vec([0.0, 1.0, 0.0, 0.0]), {"text": "b"})

    path = tmp_path / "store.pkl"
    vs.save(path)

    loaded = VectorStore.load(path)
    assert len(loaded) == 2

    results = loaded.search(_unit_vec([1.0, 0.0, 0.0, 0.0]), k=1)
    assert results[0]["metadata"]["text"] == "a"


def test_empty_store_search_returns_empty_list() -> None:
    """Searching an empty store does not crash; returns []."""
    vs = VectorStore(dim=4)

    results = vs.search(_unit_vec([1.0, 0.0, 0.0, 0.0]), k=5)

    assert results == []
