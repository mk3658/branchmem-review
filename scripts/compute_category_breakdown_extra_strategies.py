#!/usr/bin/env python3
"""Post-hoc: category-level accuracy for ConfidenceRuleMerge, TwoWayLLMMerge,
and RawTextLLMMerge on the exact locked scenarios (same seed=2026, same
config as scripts/run_experiment.py), so Table 2 can show these strategies
promoted out of ablation-only status rather than only overall accuracy.

ConfidenceRuleMerge needs no LLM call at all (deterministic). TwoWayLLMMerge
and RawTextLLMMerge reuse the content-hash cache from the earlier ablation
rounds -- this script makes zero new API calls if that cache is populated.
Not part of ANALYSIS_PLAN.md's confirmatory analysis: purely descriptive,
post-hoc (see ANALYSIS_PLAN_ADDENDUM.md).
"""
from __future__ import annotations

import json
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.scenario_generator import ScenarioConfig, ScenarioGenerator
from branchmem.evaluation.metrics import score_downstream
from branchmem.llm.base import build_backend
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.raw_text_llm import RawTextLLMMerge
from branchmem.merge.two_way_llm import TwoWayLLMMerge
from branchmem.utils.seeding import set_all_seeds

SEED = 2026
N_SCENARIOS = 60
DIVERGENCE_SPANS = [4.0, 10.0, 20.0]
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "final"


def main() -> None:
    set_all_seeds(SEED)
    llm_config = {"backend": "openai_compatible", "model": "gpt-5.4-nano", "cache_dir": "llm_cache",
                  "temperature": 0.0, "max_tokens": 8000}
    backend = build_backend(llm_config)

    generator = ScenarioGenerator()
    n_per_span = N_SCENARIOS // len(DIVERGENCE_SPANS)
    remainder = N_SCENARIOS - n_per_span * len(DIVERGENCE_SPANS)
    scenarios = []
    for i, span in enumerate(DIVERGENCE_SPANS):
        n_this_span = n_per_span + (1 if i < remainder else 0)
        config = ScenarioConfig(divergence_span=span)
        scenarios.extend(generator.generate(n_this_span, config, seed=SEED + i))

    strategies = {
        "confidence_rule": ConfidenceRuleMerge(),
        "two_way_llm": TwoWayLLMMerge(backend=backend),
        "raw_text_llm": RawTextLLMMerge(backend=backend),
    }

    per_scenario = []
    overall = {name: [] for name in strategies}
    by_category = {name: {"orthogonal": [], "resolvable": [], "ambiguous": []} for name in strategies}
    for scenario in scenarios:
        questions = generate_downstream_questions(scenario)
        row = {"scenario_id": scenario.scenario_id, "divergence_span": scenario.metadata.get("divergence_span", 0.0)}
        for name, strategy in strategies.items():
            merge_result = strategy.merge(scenario.ancestor, scenario.branch_a, scenario.branch_b)
            score = score_downstream(questions, merge_result)
            row[f"{name}_accuracy"] = score.accuracy
            overall[name].append(score.accuracy)
            for cat in ("orthogonal", "resolvable", "ambiguous"):
                by_category[name][cat].append(score.accuracy_for(cat))
        per_scenario.append(row)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    result = {
        "note": (
            "post-hoc, exploratory: category-level accuracy for strategies not in the "
            "original build_strategies() locked run, computed on the identical seed=2026 "
            "scenarios. ConfidenceRuleMerge makes zero LLM calls (deterministic); "
            "TwoWayLLMMerge/RawTextLLMMerge reuse the existing content-hash cache. "
            "Not part of ANALYSIS_PLAN.md's confirmatory tests."
        ),
        "per_scenario": per_scenario,
        "mean_accuracy_overall": {name: mean(vals) for name, vals in overall.items()},
        "mean_accuracy_by_category": {
            name: {cat: mean(vals) for cat, vals in cats.items()} for name, cats in by_category.items()
        },
    }
    out_path = OUT_DIR / "category_breakdown_extra_strategies.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("Wrote", out_path)
    print(json.dumps(result["mean_accuracy_by_category"], indent=2))


if __name__ == "__main__":
    main()
