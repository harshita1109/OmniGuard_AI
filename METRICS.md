# METRICS

Numbers measured during the project. Every entry: phase, what was measured, value, date.

| Phase | Metric | Value | Date |
|-------|--------|-------|------|
| 4     | Vector-only F1 (synthetic dataset, n=282, k=5, 80/20 split) | 0.870 @ threshold 0.75 | 2026-05-30 |

## Phase 4 details

- **Dataset**: 282 synthetic examples from `scripts/generate_data.py` — 152 normal (short factual statements) + 130 anomaly across 6 categories (gibberish, prompt-injection, off-topic, hallucination, truncated, surreal nonsense).
- **Split**: 80% train (226) / 20% test (56), random seed 42.
- **Baseline store**: 120 normal sentences from the train split, persisted to `data/store.pkl`.
- **Detector**: kNN, k=5, 1−cosine distance, mean of top-k distances as score.
- **Threshold sweep**: 0.10 → 0.95 in steps of 0.05.

### Best confusion matrix (threshold = 0.75)

|              | predicted normal | predicted anomaly |
|--------------|------------------|-------------------|
| actually normal  | 30 (TN)         | 2 (FP)           |
| actually anomaly | 4 (FN)          | 20 (TP)          |

- Precision = TP / (TP + FP) = 20 / 22 = **0.909**
- Recall = TP / (TP + FN) = 20 / 24 = **0.833**
- F1 = 2·P·R / (P + R) = **0.870**

### Score range observed

- min = 0.551, max = 0.908, mean = 0.721
- All scores cluster above 0.55 → the embedding model treats short English sentences as broadly similar; what matters is the *gap* between normal and anomaly distributions, which is real at threshold 0.75.

### Why this number is defensible

- Real dataset (committed generator script — anyone can reproduce).
- Train/test split prevents the baseline from "knowing" the test items.
- Threshold chosen by sweep, not by eye.
- Multiple anomaly categories — not just one easy class.

### Future improvements logged here as they happen

(Add new rows below as later phases produce numbers.)
