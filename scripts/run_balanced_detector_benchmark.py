#!/usr/bin/env python3
"""Post-hoc, exploratory: balanced conflict-detector benchmark (see
ANALYSIS_PLAN_ADDENDUM.md item A3). The locked RQ2 evaluation set contains
only genuine conflicts by construction, so every detector's precision is
mechanically 1.0. This adds five non-conflict categories alongside genuine
contradictions and reports full precision/recall/F1/specificity/FPR/AUROC.
NOT part of ANALYSIS_PLAN.md's confirmatory tests; does not change or
retract RQ2's locked result.
"""
from __future__ import annotations

import json
from pathlib import Path

from branchmem.conflict.embedding_detector import EmbeddingConflictDetector
from branchmem.conflict.llm_judge_detector import LLMJudgeConflictDetector
from branchmem.conflict.nli_detector import NLIConflictDetector
from branchmem.llm.base import build_backend
from branchmem.memory.schemas import MemoryFact

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "balanced_detector_benchmark.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "final"


def _fact(entity, predicate, value, branch_id):
    return MemoryFact(entity=entity, predicate=predicate, value=value, branch_id=branch_id, timestamp=1.0)


def _auroc(labels: list[bool], scores: list[float]) -> float:
    """Rank-based AUROC (Mann-Whitney U), no sklearn dependency."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    all_scores = sorted(scores)
    ranks = {}
    i = 0
    n = len(all_scores)
    while i < n:
        j = i
        while j < n and all_scores[j] == all_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[all_scores[k]] = avg_rank
        i = j
    rank_sum_pos = sum(ranks[s] for s in pos)
    n_pos, n_neg = len(pos), len(neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def full_metrics(labels: list[bool], preds: list[bool], scores: list[float]) -> dict:
    tp = sum(1 for y, p in zip(labels, preds) if y and p)
    fp = sum(1 for y, p in zip(labels, preds) if not y and p)
    fn = sum(1 for y, p in zip(labels, preds) if y and not p)
    tn = sum(1 for y, p in zip(labels, preds) if not y and not p)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) and precision == precision and recall == recall else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "specificity": specificity, "false_positive_rate": fpr,
        "auroc": _auroc(labels, scores),
    }


def per_category_recall(pairs: list[dict], preds: list[bool]) -> dict:
    by_cat: dict[str, list[bool]] = {}
    for pair, pred in zip(pairs, preds):
        by_cat.setdefault(pair["category"], []).append(pred == pair["is_conflict"])
    return {cat: sum(v) / len(v) for cat, v in by_cat.items()}


def main() -> None:
    pairs = json.loads(DATA_PATH.read_text())["pairs"]
    labels = [p["is_conflict"] for p in pairs]

    llm_config = {"backend": "openai_compatible", "model": "gpt-5.4-nano", "cache_dir": "llm_cache",
                  "temperature": 0.0, "max_tokens": 1024}
    backend = build_backend(llm_config)

    detectors = {
        "embedding_threshold": EmbeddingConflictDetector(threshold=0.80),
        "nli": NLIConflictDetector(contradiction_threshold=0.20),
        "llm_judge": LLMJudgeConflictDetector(backend=backend),
    }

    report = {
        "note": (
            "post-hoc, exploratory balanced conflict-detector benchmark; NOT part of "
            "ANALYSIS_PLAN.md's confirmatory tests (see ANALYSIS_PLAN_ADDENDUM.md item A3). "
            "Does not change or retract RQ2's locked result -- a harder, separate, "
            "descriptive follow-up."
        ),
        "n_pairs": len(pairs),
        "n_conflict": sum(labels),
        "n_non_conflict": len(labels) - sum(labels),
        "categories": sorted(set(p["category"] for p in pairs)),
        "detectors": {},
    }

    for name, detector in detectors.items():
        preds, scores = [], []
        for p in pairs:
            fact_a = _fact(p["entity"], p["predicate"], p["value_a"], "a")
            fact_b = _fact(p["entity"], p["predicate"], p["value_b"], "b")
            judgment = detector.detect(fact_a, fact_b)
            preds.append(judgment.is_conflict)
            scores.append(judgment.score)
        metrics = full_metrics(labels, preds, scores)
        metrics["per_category_accuracy"] = per_category_recall(pairs, preds)
        report["detectors"][name] = metrics

    out_path = OUT_DIR / "balanced_detector_benchmark.json"
    out_path.write_text(json.dumps(report, indent=2))
    print("Wrote", out_path)
    print(json.dumps(report["detectors"], indent=2))


if __name__ == "__main__":
    main()
