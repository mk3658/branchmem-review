#!/usr/bin/env python3
"""Post-hoc, non-preregistered ablation: RawTextLLMMerge on the same locked
n=60 / 180-conflict-pair scenario set used for `results/final/results.json`.

Added in response to the ACL review's action item #5
(`paper/reviews/acl2027_review.md` Sections 3, 4.8, 9.5), explicitly
APPROVED by the user for real API spend. This does NOT modify
`scripts/run_experiment.py`, `results/final/results.json`, or
`results/final/stats_output.json` — those remain the locked confirmatory
artifact. This script only regenerates the identical scenario set (same
seed=2026, same divergence spans, same ScenarioConfig defaults as
`run_experiment.py`/`run_full_benchmark`) and runs one additional strategy
on it, exactly like the ad hoc script that produced
`results/final/new_baselines.json` (ConfidenceRuleMerge, TwoWayLLMMerge) in
the prior review round.

Writes `results/final/raw_text_ablation.json`: per-scenario accuracy,
mean accuracy, and a paired comparison against `three_way_llm` (loaded from
the locked `results/final/results.json`), using the same paired Wilcoxon +
bootstrap CI machinery as the confirmatory analysis, but reported as
exploratory/post-hoc per project convention.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.scenario_generator import ScenarioConfig, ScenarioGenerator
from branchmem.evaluation.metrics import score_downstream
from branchmem.evaluation.stats import paired_comparison
from branchmem.llm.base import build_backend
from branchmem.merge.raw_text_llm import RawTextLLMMerge
from branchmem.utils.logging import get_logger

logger = get_logger("run_raw_text_ablation")

# Must match scripts/run_experiment.py exactly so the scenario set is identical.
SEED = 2026
N_SCENARIOS = 60
MODEL = "gpt-5.4-nano"
DIVERGENCE_SPANS = [4.0, 10.0, 20.0]

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "final"


def generate_locked_scenarios():
    generator = ScenarioGenerator()
    n_per_span = N_SCENARIOS // len(DIVERGENCE_SPANS)
    remainder = N_SCENARIOS - n_per_span * len(DIVERGENCE_SPANS)
    scenarios = []
    for i, span in enumerate(DIVERGENCE_SPANS):
        n_this_span = n_per_span + (1 if i < remainder else 0)
        config = ScenarioConfig(divergence_span=span)
        span_seed = SEED + i
        scenarios.extend(generator.generate(n_this_span, config, seed=span_seed))
    return scenarios


def main() -> None:
    scenarios = generate_locked_scenarios()
    assert len(scenarios) == N_SCENARIOS

    # Sanity check: scenario IDs must match the locked run's, confirming this
    # is really the same scenario set (not just the same count).
    locked = json.loads((OUT_DIR / "results.json").read_text())
    locked_ids = [r["scenario_id"] for r in locked["per_scenario"]]
    assert [s.scenario_id for s in scenarios] == locked_ids, (
        "Regenerated scenario IDs don't match results.json — scenario "
        "generator config has drifted from the locked run."
    )
    locked_three_way = {
        r["scenario_id"]: r["accuracy_by_strategy"]["three_way_llm"] for r in locked["per_scenario"]
    }

    llm_config = {
        "backend": "openai_compatible", "model": MODEL, "cache_dir": "llm_cache",
        "temperature": 0.0, "max_tokens": 8000,
    }
    backend = build_backend(llm_config)
    strategy = RawTextLLMMerge(backend=backend)

    per_scenario = []
    for idx, scenario in enumerate(scenarios):
        questions = generate_downstream_questions(scenario)
        merge_result = strategy.merge(scenario.ancestor, scenario.branch_a, scenario.branch_b)
        score = score_downstream(questions, merge_result)
        per_scenario.append({
            "scenario_id": scenario.scenario_id,
            "divergence_span": scenario.metadata.get("divergence_span", 0.0),
            "raw_text_llm_accuracy": score.accuracy,
            "category_accuracy": {
                cat: score.accuracy_for(cat) for cat in ("orthogonal", "resolvable", "ambiguous")
            },
        })
        if (idx + 1) % 10 == 0 or idx == len(scenarios) - 1:
            logger.info("Processed %d/%d scenarios", idx + 1, len(scenarios))

    raw_text_accs = [r["raw_text_llm_accuracy"] for r in per_scenario]
    three_way_accs = [locked_three_way[r["scenario_id"]] for r in per_scenario]

    comparison = paired_comparison("three_way_llm", three_way_accs, "raw_text_llm", raw_text_accs, seed=SEED)

    category_means = {}
    for cat in ("orthogonal", "resolvable", "ambiguous"):
        vals = [r["category_accuracy"][cat] for r in per_scenario if r["category_accuracy"][cat] == r["category_accuracy"][cat]]
        category_means[cat] = statistics.mean(vals) if vals else float("nan")

    out = {
        "mean_accuracy": {"raw_text_llm": statistics.mean(raw_text_accs)},
        "mean_accuracy_by_category": category_means,
        "per_scenario": per_scenario,
        "note": (
            "exploratory, post-hoc ablation added in response to peer review "
            "(action item #5: minimal LLM-merge condition with NO structured "
            "metadata -- no ancestor, no source, no confidence, only the two "
            "raw conflicting values); not part of ANALYSIS_PLAN.md confirmatory "
            "tests. Same n=60 locked scenarios as results/final/results.json "
            "(verified by matching scenario_id)."
        ),
        "paired_comparisons": {
            "three_way_vs_raw_text_llm": {
                "mean_diff": comparison.mean_diff,
                "sd_diff": comparison.sd_diff,
                "ci_low": comparison.ci_low,
                "ci_high": comparison.ci_high,
                "wilcoxon_statistic": comparison.wilcoxon_statistic,
                "p_value": comparison.p_value,
            },
            "note": "exploratory, post-hoc paired comparison; not part of ANALYSIS_PLAN.md confirmatory tests",
        },
    }
    (OUT_DIR / "raw_text_ablation.json").write_text(json.dumps(out, indent=2))
    logger.info("Wrote %s", OUT_DIR / "raw_text_ablation.json")
    logger.info(
        "raw_text_llm mean=%.3f vs three_way_llm mean=%.3f, diff=%.3f [%.3f, %.3f] p=%.4f",
        statistics.mean(raw_text_accs), statistics.mean(three_way_accs),
        comparison.mean_diff, comparison.ci_low, comparison.ci_high, comparison.p_value,
    )


if __name__ == "__main__":
    main()
