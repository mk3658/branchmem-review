#!/usr/bin/env python3
"""Post-hoc, exploratory: cross-strategy risk-coverage frontier on the locked
synthetic set (pooled resolvable+ambiguous, n=180). NOT part of
ANALYSIS_PLAN.md confirmatory tests.

The synthetic construction exposes only two confidence tiers (0.95 vs 0.55),
so a single strategy cannot be swept across a continuum of coverage the way
selective-prediction risk-coverage curves usually are. Instead we plot each
strategy's *achieved* operating point (coverage = commit rate on the
in-scope categories; risk = 1 - conditional-on-commit accuracy). The
deterministic strategies here make no LLM call, so this whole computation is
zero-API; the ThreeWayLLMMerge point is taken from the already-computed,
cache-backed abstention report (results/final/abstention_metrics_report.json)
rather than recomputed, to avoid any cache/token ambiguity.
"""
from __future__ import annotations

import json
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.scenario_generator import ScenarioConfig, ScenarioGenerator
from branchmem.eval.abstention_metrics import compute_abstention_metrics
from branchmem.evaluation.metrics import score_downstream
from branchmem.merge.branch_discard import BranchDiscard
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.last_writer_wins import LastWriterWins
from branchmem.merge.naive_concat import NaiveConcat
from branchmem.utils.seeding import set_all_seeds

SEED = 2026
SPANS = [4.0, 10.0, 20.0]
N_PER_SPAN = 20
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "final"


def main() -> None:
    set_all_seeds(SEED)
    gen = ScenarioGenerator()
    scenarios = []
    for i, span in enumerate(SPANS):
        scenarios.extend(gen.generate(N_PER_SPAN, ScenarioConfig(divergence_span=span), seed=SEED + i))

    # Deterministic, no-LLM strategies only -> zero API.
    strategies = {
        "last_writer_wins": LastWriterWins(),
        "naive_concat": NaiveConcat(),
        "branch_discard_always_b": BranchDiscard(policy="always_b"),
        "confidence_rule": ConfidenceRuleMerge(),
    }

    points = {}
    for name, strat in strategies.items():
        detail = []
        for scenario in scenarios:
            questions = generate_downstream_questions(scenario)
            merged = strat.merge(scenario.ancestor, scenario.branch_a, scenario.branch_b)
            detail.extend(score_downstream(questions, merged).detail)
        m = compute_abstention_metrics(detail)
        points[name] = {
            "coverage": m.coverage,
            "risk": m.risk,
            "commit_rate": m.commit_rate,
            "conditional_on_commit_accuracy": m.conditional_on_commit_accuracy,
        }

    # ThreeWayLLMMerge point from the already-computed cache-backed report.
    report = json.loads((OUT_DIR / "abstention_metrics_report.json").read_text())
    syn = report["synthetic_resolvable_and_ambiguous"]
    points["three_way_llm"] = {
        "coverage": syn["coverage"], "risk": syn["risk"],
        "commit_rate": syn["commit_rate"],
        "conditional_on_commit_accuracy": syn["conditional_on_commit_accuracy"],
    }

    result = {
        "note": (
            "post-hoc, exploratory cross-strategy risk-coverage frontier on the locked "
            "synthetic set (pooled resolvable+ambiguous, n=180). Deterministic strategies "
            "computed zero-API; ThreeWayLLMMerge point taken from the cache-backed "
            "abstention report. Two confidence tiers only, so these are achieved operating "
            "points, not a swept selective-prediction curve. NOT part of ANALYSIS_PLAN.md "
            "confirmatory tests."
        ),
        "points": points,
    }
    out_path = OUT_DIR / "risk_coverage_frontier.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("Wrote", out_path)
    print(json.dumps(points, indent=2))


if __name__ == "__main__":
    main()
