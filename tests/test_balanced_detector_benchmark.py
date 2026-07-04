import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_balanced_detector_benchmark import _auroc, full_metrics  # noqa: E402


def test_auroc_perfect_separation_is_one():
    labels = [True, True, False, False]
    scores = [0.9, 0.8, 0.2, 0.1]
    assert _auroc(labels, scores) == 1.0


def test_auroc_perfect_anti_separation_is_zero():
    labels = [True, True, False, False]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert _auroc(labels, scores) == 0.0


def test_auroc_random_scores_near_half():
    labels = [True, False, True, False]
    scores = [0.5, 0.5, 0.5, 0.5]
    assert _auroc(labels, scores) == 0.5


def test_full_metrics_matches_confusion_counts():
    labels = [True, True, False, False]
    preds = [True, False, True, False]
    m = full_metrics(labels, preds, [1, 0, 1, 0])
    assert m["tp"] == 1 and m["fn"] == 1 and m["fp"] == 1 and m["tn"] == 1
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["specificity"] == 0.5
    assert m["false_positive_rate"] == 0.5
