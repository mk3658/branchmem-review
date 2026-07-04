"""Abstention-aware metrics: separates calibrated abstention from genuine
commit-time error, since raw accuracy scores both as equally "not correct."

Added post-hoc (see ANALYSIS_PLAN_ADDENDUM.md, item A1). Every function
here is a pure computation over already-scored
per-question detail records (see branchmem.evaluation.metrics.score_downstream's
`detail` list) or simple counts; nothing here makes an LLM call.

A question record is expected to look like:
    {"category": "resolvable"|"ambiguous"|"orthogonal",
     "correct": bool, "got_resolution": str|None}
"COMMITTED" means got_resolution is not "flagged_unresolved" (and not None,
i.e. the key was addressed at all). Orthogonal questions are excluded from
abstention accounting: there is no ambiguity to abstain from, every strategy
either preserves the fact or drops it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

FLAGGED = "flagged_unresolved"


@dataclass
class AbstentionMetrics:
    n_questions: int
    n_committed: int
    n_correct_committed: int
    n_wrong_committed: int
    n_flagged: int
    n_ambiguous_total: int
    n_ambiguous_flagged: int
    n_resolvable_or_orthog_flagged: int  # false-positive flags (should have committed)
    commit_rate: float
    abstention_rate: float
    conditional_on_commit_accuracy: Optional[float]
    wrong_commit_rate: float
    flag_precision: Optional[float]  # of all flags, how many were on genuinely ambiguous items
    flag_recall: Optional[float]  # of all genuinely ambiguous items, how many were flagged
    expected_utility: dict = field(default_factory=dict)  # cost_label -> utility
    coverage: float = 0.0
    risk: Optional[float] = None


def compute_abstention_metrics(
    detail: list[dict],
    categories_in_scope: tuple[str, ...] = ("resolvable", "ambiguous"),
    utility_costs: Optional[dict[str, float]] = None,
) -> AbstentionMetrics:
    """`detail` is a flat list of per-question dicts (category, correct,
    got_resolution) pooled across scenarios/conflicts for one strategy.
    `categories_in_scope` restricts which categories count toward
    commit/abstention accounting (orthogonal is excluded by default: there
    is nothing to abstain from there).
    """
    utility_costs = utility_costs or {"wrong_commit_-2": -2.0, "wrong_commit_-5": -5.0, "wrong_commit_-10": -10.0}
    rows = [d for d in detail if d["category"] in categories_in_scope]
    n_questions = len(rows)

    n_committed = sum(1 for d in rows if d.get("got_resolution") != FLAGGED)
    n_flagged = n_questions - n_committed
    n_correct_committed = sum(1 for d in rows if d.get("got_resolution") != FLAGGED and d["correct"])
    n_wrong_committed = n_committed - n_correct_committed

    ambiguous_rows = [d for d in rows if d["category"] == "ambiguous"]
    n_ambiguous_total = len(ambiguous_rows)
    n_ambiguous_flagged = sum(1 for d in ambiguous_rows if d.get("got_resolution") == FLAGGED)
    non_ambiguous_rows = [d for d in rows if d["category"] != "ambiguous"]
    n_resolvable_or_orthog_flagged = sum(1 for d in non_ambiguous_rows if d.get("got_resolution") == FLAGGED)

    commit_rate = n_committed / n_questions if n_questions else float("nan")
    abstention_rate = n_flagged / n_questions if n_questions else float("nan")
    conditional_on_commit_accuracy = (n_correct_committed / n_committed) if n_committed else None
    wrong_commit_rate = n_wrong_committed / n_questions if n_questions else float("nan")

    flag_precision = (n_ambiguous_flagged / n_flagged) if n_flagged else None
    flag_recall = (n_ambiguous_flagged / n_ambiguous_total) if n_ambiguous_total else None

    expected_utility = {}
    for label, wrong_cost in utility_costs.items():
        utility = (n_correct_committed * 1.0) + (n_wrong_committed * wrong_cost) + (n_flagged * -1.0)
        expected_utility[label] = utility / n_questions if n_questions else float("nan")

    coverage = commit_rate
    risk = (1.0 - conditional_on_commit_accuracy) if conditional_on_commit_accuracy is not None else None

    return AbstentionMetrics(
        n_questions=n_questions,
        n_committed=n_committed,
        n_correct_committed=n_correct_committed,
        n_wrong_committed=n_wrong_committed,
        n_flagged=n_flagged,
        n_ambiguous_total=n_ambiguous_total,
        n_ambiguous_flagged=n_ambiguous_flagged,
        n_resolvable_or_orthog_flagged=n_resolvable_or_orthog_flagged,
        commit_rate=commit_rate,
        abstention_rate=abstention_rate,
        conditional_on_commit_accuracy=conditional_on_commit_accuracy,
        wrong_commit_rate=wrong_commit_rate,
        flag_precision=flag_precision,
        flag_recall=flag_recall,
        expected_utility=expected_utility,
        coverage=coverage,
        risk=risk,
    )


def metrics_from_counts(
    n_questions: int, n_committed: int, n_correct_committed: int,
    utility_costs: Optional[dict[str, float]] = None,
) -> AbstentionMetrics:
    """Fallback for result files that only logged aggregate commit/correct
    counts (no per-question `detail`, e.g. the existing
    abstention_adjusted_metric.json). Ambiguous-specific flag precision/recall
    are unavailable in this path (None) since the source counts don't
    distinguish which category was flagged.
    """
    utility_costs = utility_costs or {"wrong_commit_-2": -2.0, "wrong_commit_-5": -5.0, "wrong_commit_-10": -10.0}
    n_flagged = n_questions - n_committed
    n_wrong_committed = n_committed - n_correct_committed
    commit_rate = n_committed / n_questions if n_questions else float("nan")
    abstention_rate = n_flagged / n_questions if n_questions else float("nan")
    conditional_on_commit_accuracy = (n_correct_committed / n_committed) if n_committed else None
    wrong_commit_rate = n_wrong_committed / n_questions if n_questions else float("nan")

    expected_utility = {}
    for label, wrong_cost in utility_costs.items():
        utility = (n_correct_committed * 1.0) + (n_wrong_committed * wrong_cost) + (n_flagged * -1.0)
        expected_utility[label] = utility / n_questions if n_questions else float("nan")

    risk = (1.0 - conditional_on_commit_accuracy) if conditional_on_commit_accuracy is not None else None

    return AbstentionMetrics(
        n_questions=n_questions, n_committed=n_committed, n_correct_committed=n_correct_committed,
        n_wrong_committed=n_wrong_committed, n_flagged=n_flagged, n_ambiguous_total=0,
        n_ambiguous_flagged=0, n_resolvable_or_orthog_flagged=0, commit_rate=commit_rate,
        abstention_rate=abstention_rate, conditional_on_commit_accuracy=conditional_on_commit_accuracy,
        wrong_commit_rate=wrong_commit_rate, flag_precision=None, flag_recall=None,
        expected_utility=expected_utility, coverage=commit_rate, risk=risk,
    )
