#!/usr/bin/env python3
"""Post-hoc, non-preregistered statistical-power expansion for the
resolvable-conflict category, explicitly approved for real API spend.

Motivation (fact-checked against the actual locked artifacts: only 60 of
180 locked-run pairs are the resolvable category, which undercounts -- the
locked `ScenarioConfig` used `n_resolvable=2` per scenario x 60 scenarios =
120 resolvable pairs, and `n_ambiguous=1` x 60 = 60 ambiguous pairs, summing
to the reported 180. The underlying concern -- that the ambiguous category
is won by
construction and the orthogonal category is tied, so `resolvable` carries
essentially all of RQ1's non-mechanical evidence -- still holds; there is
just more of it already (120, not 60) than a naive pair count suggests.
This script adds to that 120 for extra statistical power, not to manufacture
power where none existed.

This does NOT modify `scripts/run_experiment.py`, `results/final/results.json`,
or `results/final/stats_output.json` -- those remain the locked confirmatory
artifact. This is an entirely new, independently-seeded scenario sample.

Design: `NEW_SEED = 2029` (base), spans [4.0, 10.0, 20.0] -> per-span
sub-seeds 2029/2030/2031 via the same `span_seed = seed + i` convention
`branchmem/evaluation/runner.py` uses. The locked run's sub-seeds were
2026/2027/2028 (`SEED=2026` in `scripts/run_experiment.py`); 2029-2031 is
disjoint from that range, so this is a genuinely independent scenario draw,
not a re-derivation of already-seen scenarios. Same `ScenarioConfig`
defaults as the locked run (no `scenario_config_kwargs` override, matching
`scripts/run_experiment.py`'s actual call), same three divergence spans,
same `gpt-5.4-nano` model.

Runs all 5 original strategies plus the 3 ablations
(ConfidenceRuleMerge, TwoWayLLMMerge, RawTextLLMMerge) on the new sample,
and reports both the overall accuracy and the resolvable-category-only
accuracy (the category this expansion targets) per strategy, plus paired
Wilcoxon + bootstrap-CI comparisons of `three_way_llm` against every other
strategy restricted to resolvable-category per-scenario accuracy. Reported
as exploratory: no Holm-Bonferroni correction is applied (matching the
convention already used for `results/final/new_baselines.json` and
`results/final/robustness_second_model.json`, which are also post-hoc and
uncorrected).
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
from branchmem.merge.branch_discard import BranchDiscard
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.last_writer_wins import LastWriterWins
from branchmem.merge.naive_concat import NaiveConcat
from branchmem.merge.raw_text_llm import RawTextLLMMerge
from branchmem.merge.three_way_llm import ThreeWayLLMMerge
from branchmem.merge.two_way_llm import TwoWayLLMMerge
from branchmem.utils.logging import get_logger

logger = get_logger("run_resolvable_power_expansion")

NEW_SEED = 2029  # disjoint from the locked run's 2026/2027/2028 span sub-seeds
N_SCENARIOS = 60
MODEL = "gpt-5.4-nano"
DIVERGENCE_SPANS = [4.0, 10.0, 20.0]

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "final"

LLM_STRATEGY_NAMES = {"three_way_llm", "two_way_llm", "raw_text_llm"}


def generate_new_scenarios():
    generator = ScenarioGenerator()
    n_per_span = N_SCENARIOS // len(DIVERGENCE_SPANS)
    remainder = N_SCENARIOS - n_per_span * len(DIVERGENCE_SPANS)
    scenarios = []
    for i, span in enumerate(DIVERGENCE_SPANS):
        n_this_span = n_per_span + (1 if i < remainder else 0)
        config = ScenarioConfig(divergence_span=span)
        span_seed = NEW_SEED + i
        scenarios.extend(generator.generate(n_this_span, config, seed=span_seed))
    return scenarios


def main() -> None:
    scenarios = generate_new_scenarios()
    assert len(scenarios) == N_SCENARIOS

    # Disjointness check against the locked run's scenario IDs.
    locked = json.loads((OUT_DIR / "results.json").read_text())
    locked_ids = {r["scenario_id"] for r in locked["per_scenario"]}
    new_ids = {s.scenario_id for s in scenarios}
    overlap = locked_ids & new_ids
    assert not overlap, f"New scenario IDs overlap with the locked run: {overlap}"

    llm_config = {
        "backend": "openai_compatible", "model": MODEL, "cache_dir": "llm_cache",
        "temperature": 0.0, "max_tokens": 8000,
    }
    backend = build_backend(llm_config)

    strategies = {
        "last_writer_wins": LastWriterWins(),
        "naive_concat": NaiveConcat(),
        "branch_discard_always_b": BranchDiscard(policy="always_b"),
        "branch_discard_fewer_updates": BranchDiscard(policy="fewer_updates"),
        "confidence_rule": ConfidenceRuleMerge(),
        "three_way_llm": ThreeWayLLMMerge(backend=backend),
        "two_way_llm": TwoWayLLMMerge(backend=backend),
        "raw_text_llm": RawTextLLMMerge(backend=backend),
    }

    per_scenario_overall: dict[str, list[float]] = {name: [] for name in strategies}
    per_scenario_resolvable: dict[str, list[float]] = {name: [] for name in strategies}
    per_scenario_records = []

    for idx, scenario in enumerate(scenarios):
        questions = generate_downstream_questions(scenario)
        record = {"scenario_id": scenario.scenario_id, "divergence_span": scenario.metadata.get("divergence_span", 0.0)}
        for name, strategy in strategies.items():
            merge_result = strategy.merge(scenario.ancestor, scenario.branch_a, scenario.branch_b)
            score = score_downstream(questions, merge_result)
            per_scenario_overall[name].append(score.accuracy)
            resolvable_acc = score.accuracy_for("resolvable")
            per_scenario_resolvable[name].append(resolvable_acc)
            record[f"{name}_accuracy"] = score.accuracy
            record[f"{name}_resolvable_accuracy"] = resolvable_acc
        per_scenario_records.append(record)
        if (idx + 1) % 10 == 0 or idx == len(scenarios) - 1:
            logger.info("Processed %d/%d scenarios", idx + 1, len(scenarios))

    mean_accuracy = {name: statistics.mean(accs) for name, accs in per_scenario_overall.items()}
    mean_resolvable_accuracy = {name: statistics.mean(accs) for name, accs in per_scenario_resolvable.items()}

    resolvable_comparisons = {}
    three_way_resolvable = per_scenario_resolvable["three_way_llm"]
    for name, accs in per_scenario_resolvable.items():
        if name == "three_way_llm":
            continue
        cmp = paired_comparison("three_way_llm", three_way_resolvable, name, accs, seed=NEW_SEED)
        resolvable_comparisons[f"three_way_vs_{name}"] = {
            "mean_diff": cmp.mean_diff,
            "sd_diff": cmp.sd_diff,
            "ci_low": cmp.ci_low,
            "ci_high": cmp.ci_high,
            "wilcoxon_statistic": cmp.wilcoxon_statistic,
            "p_value": cmp.p_value,
        }

    out = {
        "run_metadata": {
            "seed_base": NEW_SEED, "n_scenarios": N_SCENARIOS, "model": MODEL,
            "divergence_spans": DIVERGENCE_SPANS,
            "n_resolvable_pairs": N_SCENARIOS * 2,  # ScenarioConfig default n_resolvable=2
        },
        "note": (
            "exploratory, post-hoc statistical-power expansion for the "
            "resolvable-category sample size; NOT part "
            "of ANALYSIS_PLAN.md's confirmatory tests and NOT pooled with "
            "results/final/results.json or results/final/stats_output.json. A "
            "genuinely new, disjoint-seed (2029-2031) scenario sample of the same "
            "size and configuration as the locked run (n=60 scenarios, 120 resolvable "
            "pairs), scored independently."
        ),
        "mean_accuracy": mean_accuracy,
        "mean_resolvable_category_accuracy": mean_resolvable_accuracy,
        "resolvable_category_paired_comparisons_vs_three_way_llm": resolvable_comparisons,
        "per_scenario": per_scenario_records,
    }
    (OUT_DIR / "resolvable_power_expansion.json").write_text(json.dumps(out, indent=2))
    logger.info("Wrote %s", OUT_DIR / "resolvable_power_expansion.json")
    logger.info("Resolvable-category mean accuracy: %s", mean_resolvable_accuracy)


if __name__ == "__main__":
    main()
