#!/usr/bin/env python3
"""Post-hoc: full abstention-aware metrics report for ThreeWayLLMMerge, on
both the synthetic locked scenarios (recomputed via cache, zero new API
cost) and real MemoryAgentBench content (loaded from the existing
abstention_adjusted_metric.json, itself already cache-computed).

Not part of ANALYSIS_PLAN.md's confirmatory analysis (see
ANALYSIS_PLAN_ADDENDUM.md, item A1).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.scenario_generator import ScenarioConfig, ScenarioGenerator
from branchmem.eval.abstention_metrics import compute_abstention_metrics, metrics_from_counts
from branchmem.evaluation.metrics import score_downstream
from branchmem.llm.base import build_backend
from branchmem.merge.three_way_llm import ThreeWayLLMMerge
from branchmem.utils.seeding import set_all_seeds

SEED = 2026
N_SCENARIOS = 60
DIVERGENCE_SPANS = [4.0, 10.0, 20.0]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "final"

UTILITY_COSTS = {"cost_-2": -2.0, "cost_-5": -5.0, "cost_-10": -10.0}


def _asdict_metrics(m):
    d = asdict(m)
    return d


def main() -> None:
    set_all_seeds(SEED)
    llm_config = {"backend": "openai_compatible", "model": "gpt-5.4-nano", "cache_dir": "llm_cache",
                  "temperature": 0.0, "max_tokens": 8000}
    backend = build_backend(llm_config)
    strategy = ThreeWayLLMMerge(backend=backend)

    generator = ScenarioGenerator()
    n_per_span = N_SCENARIOS // len(DIVERGENCE_SPANS)
    remainder = N_SCENARIOS - n_per_span * len(DIVERGENCE_SPANS)
    scenarios = []
    for i, span in enumerate(DIVERGENCE_SPANS):
        n_this_span = n_per_span + (1 if i < remainder else 0)
        config = ScenarioConfig(divergence_span=span)
        scenarios.extend(generator.generate(n_this_span, config, seed=SEED + i))

    all_detail = []
    for scenario in scenarios:
        questions = generate_downstream_questions(scenario)
        merge_result = strategy.merge(scenario.ancestor, scenario.branch_a, scenario.branch_b)
        score = score_downstream(questions, merge_result)
        all_detail.extend(score.detail)

    synthetic_metrics = compute_abstention_metrics(all_detail, utility_costs=UTILITY_COSTS)

    # MAB side: reuse the already-computed aggregate counts (no new API calls;
    # the source result file was itself computed from cached model calls).
    existing = json.loads((RESULTS_DIR / "abstention_adjusted_metric.json").read_text())
    mab = existing["mab_expanded"]
    mab_metrics = metrics_from_counts(
        n_questions=mab["n_questions"], n_committed=mab["n_committed"],
        n_correct_committed=mab["n_correct_when_committed"], utility_costs=UTILITY_COSTS,
    )

    report = {
        "note": (
            "post-hoc, exploratory abstention-aware metrics; NOT part of "
            "ANALYSIS_PLAN.md's confirmatory tests (see ANALYSIS_PLAN_ADDENDUM.md, "
            "item A1). Synthetic side recomputed via branchmem.eval.abstention_metrics "
            "on the identical locked seed=2026 scenarios (cache-backed, zero new API "
            "spend). MAB side derived from the existing abstention_adjusted_metric.json "
            "aggregate counts (also cache-backed)."
        ),
        "utility_cost_regimes": {"correct_commit": 1.0, "abstention": -1.0, **{k: v for k, v in UTILITY_COSTS.items()}},
        "synthetic_resolvable_and_ambiguous": _asdict_metrics(synthetic_metrics),
        "mab_expanded": _asdict_metrics(mab_metrics),
    }
    out_path = RESULTS_DIR / "abstention_metrics_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print("Wrote", out_path)
    print(json.dumps({"synthetic": report["synthetic_resolvable_and_ambiguous"],
                       "mab": report["mab_expanded"]}, indent=2))


if __name__ == "__main__":
    main()
