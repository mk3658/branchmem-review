#!/usr/bin/env python3
"""Post-hoc, exploratory: cross-vendor replication of the real-content
MemoryAgentBench secondary analysis (see ANALYSIS_PLAN_ADDENDUM.md and
results/final/mab_secondary_analysis_expanded.json). NOT part of
ANALYSIS_PLAN.md's confirmatory tests. Identical construction (all 8
Conflict_Resolution rows, max_conflict_keys=50) to the original gpt-5.4-nano
run, with a genuinely different model vendor (Anthropic Claude) for
ThreeWayLLMMerge. Tests whether a different model family also abstains on
every conflict once the engineered source-reliability signal is absent, or
whether that behavior is specific to gpt-5.4-nano.
"""
from __future__ import annotations

import json
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.mab_extension import build_branch_scenario_from_mab_row, load_mab_conflict_resolution
from branchmem.evaluation.metrics import score_downstream
from branchmem.llm.base import build_backend
from branchmem.merge.branch_discard import BranchDiscard
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.last_writer_wins import LastWriterWins
from branchmem.merge.naive_concat import NaiveConcat
from branchmem.merge.three_way_llm import ThreeWayLLMMerge
from branchmem.utils.seeding import set_all_seeds

SEED = 2026
MAX_CONFLICT_KEYS = 50  # matches the expanded (not the original 15-cap) MAB secondary analysis
MODEL = "claude-haiku-4-5-20251001"
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "final"


def main() -> None:
    set_all_seeds(SEED)
    llm_config = {"backend": "anthropic", "model": MODEL, "cache_dir": "llm_cache",
                  "temperature": 0.0, "max_tokens": 8000}
    backend = build_backend(llm_config)

    df = load_mab_conflict_resolution()
    strategies = {
        "last_writer_wins": LastWriterWins(),
        "naive_concat": NaiveConcat(),
        "branch_discard_always_b": BranchDiscard(policy="always_b"),
        "branch_discard_fewer_updates": BranchDiscard(policy="fewer_updates"),
        "confidence_rule": ConfidenceRuleMerge(),
        "three_way_llm": ThreeWayLLMMerge(backend=backend),
    }

    per_row = []
    accuracy_by_strategy = {name: [] for name in strategies}
    total_conflict_pairs = 0
    for i, row in df.iterrows():
        scenario = build_branch_scenario_from_mab_row(
            row, scenario_id=f"mab_{i}", seed=SEED, max_conflict_keys=MAX_CONFLICT_KEYS
        )
        if scenario is None:
            continue
        questions = generate_downstream_questions(scenario)
        row_entry = {
            "row": int(i), "n_questions": len(questions), "n_conflict_pairs": len(scenario.conflict_pairs),
            "accuracy_by_strategy": {},
        }
        total_conflict_pairs += len(scenario.conflict_pairs)
        for name, strategy in strategies.items():
            merge_result = strategy.merge(scenario.ancestor, scenario.branch_a, scenario.branch_b)
            score = score_downstream(questions, merge_result)
            row_entry["accuracy_by_strategy"][name] = score.accuracy
            accuracy_by_strategy[name].append(score.accuracy)
        per_row.append(row_entry)

    mean_accuracy = {
        name: (sum(vals) / len(vals) if vals else None) for name, vals in accuracy_by_strategy.items()
    }

    result = {
        "source": "mab_extension",
        "model": MODEL,
        "max_conflict_keys": MAX_CONFLICT_KEYS,
        "max_tokens": 8000,
        "n_rows": len(per_row),
        "total_conflict_pairs": total_conflict_pairs,
        "per_row": per_row,
        "mean_accuracy_by_strategy": mean_accuracy,
        "note": (
            "post-hoc, exploratory: cross-vendor replication of the real-content "
            "MemoryAgentBench secondary analysis (identical construction to "
            "results/final/mab_secondary_analysis_expanded.json's gpt-5.4-nano run, "
            "same 8 rows, max_conflict_keys=50). NOT part of ANALYSIS_PLAN.md "
            "confirmatory tests."
        ),
    }
    out_path = OUT_DIR / "cross_vendor_mab.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("Wrote", out_path)
    print(json.dumps(mean_accuracy, indent=2))


if __name__ == "__main__":
    main()
