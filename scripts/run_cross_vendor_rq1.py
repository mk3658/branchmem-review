#!/usr/bin/env python3
"""Post-hoc, exploratory: cross-vendor RQ1 replication (see
ANALYSIS_PLAN_ADDENDUM.md). NOT part of ANALYSIS_PLAN.md's confirmatory
tests. Regenerates the identical 20 scenarios used for the locked run's
divergence_span=4.0 group (seed=2026) and for the existing gpt-5.4-mini
robustness check (results/final/robustness_second_model.json), and runs
ThreeWayLLMMerge with a genuinely different model vendor (Anthropic Claude)
instead of a same-provider model swap. Baselines are model-independent and
reused from results/final/results.csv, not rerun.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from branchmem.evaluation.runner import run_full_benchmark
from branchmem.evaluation.stats import paired_comparison
from branchmem.llm.base import build_backend
from branchmem.utils.seeding import set_all_seeds

SEED = 2026
N_SCENARIOS = 20
DIVERGENCE_SPANS = [4.0]  # matches the locked run's divergence_span=4.0 group exactly
MODEL = "claude-haiku-4-5-20251001"
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "final"


def _load_locked_nano_accuracies() -> dict[str, float]:
    """Pull the original gpt-5.4-nano three_way_llm accuracies for the same
    20 scenario IDs, from the already-written locked results.csv (no rerun)."""
    accs = {}
    with open(OUT_DIR / "results.csv") as f:
        for row in csv.DictReader(f):
            if row["scenario_id"].startswith("synth_2026_00") and float(row["divergence_span"]) == 4.0:
                accs[row["scenario_id"]] = float(row["three_way_llm"])
    return accs


def main() -> None:
    set_all_seeds(SEED)
    llm_config = {"backend": "anthropic", "model": MODEL, "cache_dir": "llm_cache",
                  "temperature": 0.0, "max_tokens": 1024}
    backend = build_backend(llm_config)

    results, _, _ = run_full_benchmark(
        backend=backend, n_scenarios=N_SCENARIOS, seed=SEED, divergence_spans=DIVERGENCE_SPANS,
    )
    claude_accs = {r.scenario_id: r.accuracy_by_strategy["three_way_llm"] for r in results}

    nano_accs = _load_locked_nano_accuracies()
    shared_ids = sorted(set(claude_accs) & set(nano_accs))
    assert len(shared_ids) == N_SCENARIOS, f"scenario_id mismatch: {len(shared_ids)} shared of {N_SCENARIOS}"

    claude_vals = [claude_accs[sid] for sid in shared_ids]
    nano_vals = [nano_accs[sid] for sid in shared_ids]
    test = paired_comparison("claude_haiku", claude_vals, "gpt_5_4_nano", nano_vals, seed=SEED)

    mini_path = OUT_DIR / "robustness_second_model.json"
    mini_mean = None
    if mini_path.exists():
        mini_data = json.loads(mini_path.read_text())
        mini_mean = statistics.mean(row["accuracy"] for row in mini_data["per_scenario"])

    result = {
        "note": (
            "post-hoc, exploratory: cross-vendor RQ1 replication on the identical "
            f"{N_SCENARIOS} scenarios (seed={SEED}, divergence_span=4.0) already used for "
            "the gpt-5.4-mini same-provider robustness check. Tests whether the RQ1 "
            "effect is OpenAI-specific or holds across a genuinely different model "
            "vendor. NOT part of ANALYSIS_PLAN.md's confirmatory tests."
        ),
        "run_metadata": {"seed": SEED, "n_scenarios": N_SCENARIOS, "divergence_span": 4.0, "model": MODEL},
        "per_scenario": [{"scenario_id": sid, "accuracy": claude_accs[sid]} for sid in shared_ids],
        "mean_accuracy_claude_haiku": statistics.mean(claude_vals),
        "mean_accuracy_gpt_5_4_nano_same_scenarios": statistics.mean(nano_vals),
        "mean_accuracy_gpt_5_4_mini_same_scenarios": mini_mean,
        "paired_comparison_claude_vs_nano": {
            "mean_diff": test.mean_diff, "ci_low": test.ci_low, "ci_high": test.ci_high,
            "p_value": test.p_value,
        },
    }
    out_path = OUT_DIR / "cross_vendor_rq1.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("Wrote", out_path)
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_scenario",)}, indent=2))


if __name__ == "__main__":
    main()
