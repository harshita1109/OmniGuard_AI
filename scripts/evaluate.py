# scripts/evaluate.py
# Phase 4.2 — train/test split, build baseline store, sweep thresholds,
# print precision/recall/F1, save the chosen store to disk.
#
# Run:  python scripts/evaluate.py

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from omniguard.detector import Detector
from omniguard.embedder import Embedder
from omniguard.vector_store import VectorStore

DATA_PATH = Path("data/raw/logs.jsonl")
STORE_PATH = Path("data/store.pkl")
SEED = 42
TEST_FRACTION = 0.20
K = 5
THRESHOLDS = [round(0.05 * i, 2) for i in range(2, 20)]  # 0.10, 0.15, ..., 0.95


def load_records(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def train_test_split(
    records: list[dict[str, str]], test_fraction: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    n_test = int(len(shuffled) * test_fraction)
    return shuffled[n_test:], shuffled[:n_test]


def confusion(predictions: list[bool], truth: list[bool]) -> tuple[int, int, int, int]:
    """Return (TP, FP, TN, FN)."""
    tp = sum(1 for p, t in zip(predictions, truth, strict=False) if p and t)
    fp = sum(1 for p, t in zip(predictions, truth, strict=False) if p and not t)
    tn = sum(1 for p, t in zip(predictions, truth, strict=False) if not p and not t)
    fn = sum(1 for p, t in zip(predictions, truth, strict=False) if not p and t)
    return tp, fp, tn, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main() -> None:
    print(f"Loading {DATA_PATH}")
    records = load_records(DATA_PATH)
    print(f"  total records: {len(records)}")

    train, test = train_test_split(records, TEST_FRACTION, SEED)
    print(f"  train: {len(train)}  test: {len(test)}")

    train_normal = [r["text"] for r in train if r["label"] == "normal"]
    print(f"  train normal (baseline): {len(train_normal)}")

    embedder = Embedder()
    store = VectorStore(dim=384)
    detector = Detector(embedder, store=store, threshold=0.5, k=K)
    detector.add_baseline(train_normal)

    # Score every test example ONCE — we'll vary the threshold afterwards.
    print(f"Scoring {len(test)} test examples...")
    scores: list[float] = []
    truth: list[bool] = []  # True means actually anomaly
    for rec in test:
        result = detector.check(rec["text"])
        scores.append(result["score"])
        truth.append(rec["label"] == "anomaly")

    # Sweep thresholds.
    print()
    print(
        f"{'threshold':>10} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} "
        f"{'precision':>10} {'recall':>8} {'F1':>6}"
    )
    print("-" * 64)
    best_f1 = -1.0
    best_threshold = 0.0
    for t in THRESHOLDS:
        predictions = [s > t for s in scores]
        tp, fp, tn, fn = confusion(predictions, truth)
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        print(
            f"{t:>10.2f} {tp:>4d} {fp:>4d} {tn:>4d} {fn:>4d} "
            f"{precision:>10.3f} {recall:>8.3f} {f1:>6.3f}"
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print()
    print(f"BEST: F1 = {best_f1:.3f} at threshold = {best_threshold:.2f}")
    print(
        f"  score range: min={min(scores):.3f}  max={max(scores):.3f}  "
        f"mean={float(np.mean(scores)):.3f}"
    )

    # Save the store for Phase 5 to load at API startup.
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    store.save(STORE_PATH)
    print(f"Saved store ({len(store)} baseline vectors) to {STORE_PATH}")


if __name__ == "__main__":
    main()
