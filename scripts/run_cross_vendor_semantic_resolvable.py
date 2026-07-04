#!/usr/bin/env python3
"""Post-hoc, exploratory: cross-vendor replication of the semantic_resolvable
category (see ANALYSIS_PLAN_ADDENDUM.md item A2 and
scripts/run_semantic_resolvable_experiment.py). NOT part of ANALYSIS_PLAN.md's
confirmatory tests. Identical 150-scenario construction (same seed=3001) as
the original gpt-5.4-nano run, with a genuinely different model vendor
(Anthropic Claude) for the three LLM-based strategies. Tests whether the
original run's striking result -- every confidence-independent strategy,
including ThreeWayLLMMerge, scores exactly 0.000 -- is a property of the
task construction or specific to gpt-5.4-nano.
"""
from __future__ import annotations

import json
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.semantic_resolvable_generator import generate_semantic_resolvable_scenarios
from branchmem.evaluation.metrics import score_downstream
from branchmem.evaluation.stats import paired_comparison
from branchmem.llm.base import build_backend
from branchmem.merge.branch_discard import BranchDiscard
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.last_writer_wins import LastWriterWins
from branchmem.merge.naive_concat import NaiveConcat
from branchmem.merge.raw_text_llm import RawTextLLMMerge
from branchmem.merge.three_way_llm import ThreeWayLLMMerge
from branchmem.merge.two_way_llm import TwoWayLLMMerge
from branchmem.utils.seeding import set_all_seeds

SEED = 3001  # identical to the original run -- same 150 scenarios, disjoint from 2026/2029
N_SCENARIOS = 150
MODEL = "claude-haiku-4-5-20251001"
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "final"


def main() -> None:
    set_all_seeds(SEED)
    llm_config = {"backend": "anthropic", "model": MODEL, "cache_dir": "llm_cache",
                  "temperature": 0.0, "max_tokens": 8000}
    backend = build_backend(llm_config)

    scenarios = generate_semantic_resolvable_scenarios(N_SCENARIOS, seed=SEED)

    strategies = {
        "last_writer_wins": LastWriterWins(),
        "naive_concat": NaiveConcat(),
        "branch_discard_always_b": BranchDiscard(policy="always_b"),
        "branch_discard_fewer_updates": BranchDiscard(policy="fewer_updates"),
        "confidence_rule": ConfidenceRuleMerge(),
        "two_way_llm": TwoWayLLMMerge(backend=backend),
        "raw_text_llm": RawTextLLMMerge(backend=backend),
        "three_way_llm": ThreeWayLLMMerge(backend=backend),
    }

    per_scenario = []
    accuracy_by_strategy = {name: [] for name in strategies}
    for scenario in scenarios:
        questions = generate_downstream_questions(scenario)
        row = {"scenario_id": scenario.scenario_id}
        for name, strategy in strategies.items():
            merge_result = strategy.merge(scenario.ancestor, scenario.branch_a, scenario.branch_b)
            score = score_downstream(questions, merge_result)
            row[f"{name}_accuracy"] = score.accuracy
            accuracy_by_strategy[name].append(score.accuracy)
        per_scenario.append(row)

    mean_accuracy = {name: sum(vals) / len(vals) for name, vals in accuracy_by_strategy.items()}

    paired_comparisons = {}
    for name in strategies:
        if name == "three_way_llm":
            continue
        test = paired_comparison(
            "three_way_llm", accuracy_by_strategy["three_way_llm"], name, accuracy_by_strategy[name], seed=SEED
        )
        paired_comparisons[f"three_way_vs_{name}"] = {
            "mean_diff": test.mean_diff, "ci_low": test.ci_low, "ci_high": test.ci_high,
            "p_value": test.p_value,
        }

    result = {
        "note": (
            "post-hoc, exploratory: cross-vendor replication of the semantic_resolvable "
            f"category, n={N_SCENARIOS} scenarios/conflict pairs, seed={SEED} (identical "
            "construction to scripts/run_semantic_resolvable_experiment.py's gpt-5.4-nano "
            "run). NOT part of ANALYSIS_PLAN.md's confirmatory tests -- see "
            "ANALYSIS_PLAN_ADDENDUM.md item A2."
        ),
        "run_metadata": {"seed": SEED, "n_scenarios": N_SCENARIOS, "model": MODEL},
        "per_scenario": per_scenario,
        "mean_accuracy": mean_accuracy,
        "paired_comparisons_vs_three_way_llm": paired_comparisons,
    }
    out_path = OUT_DIR / "cross_vendor_semantic_resolvable.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("Wrote", out_path)
    print(json.dumps(mean_accuracy, indent=2))


if __name__ == "__main__":
    main()
