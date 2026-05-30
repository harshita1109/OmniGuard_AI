# tests/test_detector.py
# Phase 2.5 — first pytest suite for Detector.

import pytest

from omniguard.detector import Detector
from omniguard.embedder import Embedder

BASELINE = [
    "Paris is the capital of France.",
    "Water boils at 100 degrees Celsius.",
    "The Earth orbits the Sun.",
    "Photosynthesis converts sunlight into chemical energy.",
    "Shakespeare wrote Hamlet.",
]


@pytest.fixture(scope="module")
def detector() -> Detector:
    """One Detector for all tests in this file (loads the model only once)."""
    d = Detector(Embedder(), threshold=0.95)
    d.add_baseline(BASELINE)
    return d


def test_anomaly_flagged(detector: Detector) -> None:
    result = detector.check("Asdf qwerty banana lightbulb.")
    assert result["is_anomaly"] is True


def test_normal_not_flagged(detector: Detector) -> None:
    result = detector.check("Mount Everest is the tallest mountain on Earth.")
    assert result["is_anomaly"] is False


def test_score_is_float_in_range(detector: Detector) -> None:
    result = detector.check("Hello world")
    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 1.5
