"""Downstream accuracy and conflict-detection P/R/F1 scoring."""

from __future__ import annotations

from dataclasses import dataclass

from branchmem.conflict.base import ConflictJudgment
from branchmem.memory.schemas import ConflictPair, MergeResult, Resolution


@dataclass
class DownstreamScore:
    scenario_id: str
    strategy_name: str
    n_total: int
    n_correct: int
    n_correct_by_category: dict[str, int]
    n_total_by_category: dict[str, int]
    detail: list[dict]

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_total if self.n_total else float("nan")

    def accuracy_for(self, category: str) -> float:
        total = self.n_total_by_category.get(category, 0)
        return self.n_correct_by_category.get(category, 0) / total if total else float("nan")


def score_downstream(questions: list, merge_result: MergeResult) -> DownstreamScore:
    """Score a MergeResult's reconciled facts against downstream questions.

    Scoring rule: for a question with expected_answer set (orthogonal /
    resolvable), correct means the reconciled fact set has EXACTLY that value
    for (entity, predicate) and it isn't flagged unresolved. For a question
    with expected_answer=None (ambiguous — no single correct answer), correct
    means the merge strategy flagged the key unresolved rather than
    confidently picking one side; guessing either side's value counts wrong,
    since a confident wrong-feeling guess on a genuinely ambiguous fact is the
    failure mode this category exists to catch.
    """
    current: dict[tuple[str, str], tuple[str, Resolution]] = {}
    for rf in merge_result.resulting_facts:
        if rf.resolution == Resolution.DROPPED:
            continue
        key = rf.fact.key()
        current[key] = (rf.fact.value, rf.resolution)

    n_correct = 0
    n_correct_by_cat: dict[str, int] = {}
    n_total_by_cat: dict[str, int] = {}
    detail = []
    for q in questions:
        key = (q.entity, q.predicate)
        got_value, got_resolution = current.get(key, (None, None))
        if q.expected_answer is None:
            correct = got_resolution == Resolution.FLAGGED_UNRESOLVED
        else:
            correct = got_value == q.expected_answer and got_resolution != Resolution.FLAGGED_UNRESOLVED
        n_correct += int(correct)
        n_total_by_cat[q.category] = n_total_by_cat.get(q.category, 0) + 1
        n_correct_by_cat[q.category] = n_correct_by_cat.get(q.category, 0) + int(correct)
        detail.append(
            {
                "question": q.question,
                "category": q.category,
                "expected": q.expected_answer,
                "got": got_value,
                "got_resolution": got_resolution.value if got_resolution else None,
                "correct": correct,
            }
        )

    return DownstreamScore(
        scenario_id=questions[0].scenario_id if questions else "",
        strategy_name=merge_result.strategy_name,
        n_total=len(questions),
        n_correct=n_correct,
        n_correct_by_category=n_correct_by_cat,
        n_total_by_category=n_total_by_cat,
        detail=detail,
    )


@dataclass
class DetectorScore:
    detector_name: str
    precision: float
    recall: float
    f1: float
    n_pairs: int
    mean_latency_s: float


def score_detector(pairs: list[ConflictPair], judgments: list[ConflictJudgment], detector_name: str) -> DetectorScore:
    tp = fp = fn = tn = 0
    for pair, judgment in zip(pairs, judgments):
        if pair.is_conflict and judgment.is_conflict:
            tp += 1
        elif not pair.is_conflict and judgment.is_conflict:
            fp += 1
        elif pair.is_conflict and not judgment.is_conflict:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) and precision == precision and recall == recall and (precision + recall) > 0 else float("nan")
    mean_latency = sum(j.latency_s for j in judgments) / len(judgments) if judgments else 0.0

    return DetectorScore(
        detector_name=detector_name, precision=precision, recall=recall, f1=f1,
        n_pairs=len(pairs), mean_latency_s=mean_latency,
    )
