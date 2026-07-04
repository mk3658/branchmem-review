#!/usr/bin/env python3
"""Post-hoc, exploratory: semantic_resolvable category experiment (see
ANALYSIS_PLAN_ADDENDUM.md item A2). NOT part of ANALYSIS_PLAN.md's
confirmatory tests. 150 new scenarios/conflict pairs, real API calls for the
three LLM-based strategies (ThreeWayLLMMerge, TwoWayLLMMerge,
RawTextLLMMerge), cached going forward.
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

SEED = 3001  # disjoint from the locked run's 2026 and the power-expansion's 2029
N_SCENARIOS = 150
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "final"


def main() -> None:
    set_all_seeds(SEED)
    llm_config = {"backend": "openai_compatible", "model": "gpt-5.4-nano", "cache_dir": "llm_cache",
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

    # Paired comparisons vs. three_way_llm, descriptive (this experiment is
    # entirely post-hoc; no Holm-Bonferroni correction is applied since there
    # is no preregistered family of tests here to correct across).
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
            "post-hoc, exploratory: semantic_resolvable category, "
            f"n={N_SCENARIOS} scenarios/conflict pairs, seed={SEED} (disjoint from the "
            "locked run's 2026 and the power-expansion's 2029). Source and confidence "
            "equal on both branches by construction; ground truth recoverable only from "
            "semantic compatibility with the ancestor constraint. NOT part of "
            "ANALYSIS_PLAN.md's confirmatory tests -- see ANALYSIS_PLAN_ADDENDUM.md item A2."
        ),
        "run_metadata": {"seed": SEED, "n_scenarios": N_SCENARIOS, "model": "gpt-5.4-nano"},
        "per_scenario": per_scenario,
        "mean_accuracy": mean_accuracy,
        "paired_comparisons_vs_three_way_llm": paired_comparisons,
    }
    out_path = OUT_DIR / "semantic_resolvable.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("Wrote", out_path)
    print(json.dumps(mean_accuracy, indent=2))
    print(json.dumps(paired_comparisons, indent=2))


if __name__ == "__main__":
    main()
