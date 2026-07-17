#!/usr/bin/env python3
"""Post-hoc, exploratory: source-signal-noise sensitivity sweep (see
ANALYSIS_PLAN_ADDENDUM.md). NOT part of ANALYSIS_PLAN.md's confirmatory
tests. Regenerates the identical locked-run scenarios (seed 2026, spans
4/10/20, n=60) at several `source_signal_noise` levels and re-scores every
strategy, to show how robust the RQ1 boundary condition and the
ConfidenceRule-vs-LLM ranking are to how informative the reliability signal is.

Key property exploited for zero-cost: `source_signal_noise` only changes each
resolvable conflict's ground-truth assignment (`rng.random() < noise`), which
consumes the same RNG draw at every noise level. Scenario *facts*
(value/source/confidence/timestamp) are therefore byte-identical across noise
levels, and the LLM prompt is built from those facts only (never from ground
truth), so every ThreeWayLLMMerge call is a cache hit against the locked run.
Only the accuracy scoring changes. Verified by watching for cache MISS lines.
"""
from __future__ import annotations

import json
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.scenario_generator import ScenarioConfig, ScenarioGenerator
from branchmem.evaluation.metrics import score_downstream
from branchmem.llm.base import build_backend
from branchmem.merge.branch_discard import BranchDiscard
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.last_writer_wins import LastWriterWins
from branchmem.merge.naive_concat import NaiveConcat
from branchmem.merge.three_way_llm import ThreeWayLLMMerge
from branchmem.utils.seeding import set_all_seeds

LOCKED_SEED = 2026
LOCKED_SPANS = [4.0, 10.0, 20.0]
N_PER_SPAN = 20
MODEL = "gpt-5.4-nano"           # locked-run model → cache hits
MAX_TOKENS = 1024                 # locked-run max_tokens → cache hits
NOISE_LEVELS = [0.0, 0.05, 0.15, 0.30, 0.50]  # 0.15 is the locked value
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "final"


def build_scenarios(noise: float):
    gen = ScenarioGenerator()
    scenarios = []
    for i, span in enumerate(LOCKED_SPANS):
        cfg = ScenarioConfig(divergence_span=span, source_signal_noise=noise)
        scenarios.extend(gen.generate(N_PER_SPAN, cfg, seed=LOCKED_SEED + i))
    return scenarios


def main() -> None:
    set_all_seeds(LOCKED_SEED)
    backend = build_backend({"backend": "openai_compatible", "model": MODEL,
                             "cache_dir": "llm_cache", "temperature": 0.0, "max_tokens": MAX_TOKENS})
    strategies = {
        "last_writer_wins": LastWriterWins(),
        "naive_concat": NaiveConcat(),
        "branch_discard_always_b": BranchDiscard(policy="always_b"),
        "confidence_rule": ConfidenceRuleMerge(),
        "three_way_llm": ThreeWayLLMMerge(backend=backend),
    }

    per_noise = []
    for noise in NOISE_LEVELS:
        scenarios = build_scenarios(noise)
        overall = {name: [] for name in strategies}
        resolvable = {name: [] for name in strategies}
        for scenario in scenarios:
            questions = generate_downstream_questions(scenario)
            for name, strat in strategies.items():
                merged = strat.merge(scenario.ancestor, scenario.branch_a, scenario.branch_b)
                score = score_downstream(questions, merged)
                overall[name].append(score.accuracy)
                resolvable[name].append(score.accuracy_for("resolvable"))
        row = {
            "noise": noise,
            "overall_mean_accuracy": {n: sum(v) / len(v) for n, v in overall.items()},
            "resolvable_mean_accuracy": {
                n: (sum(x for x in v if x == x) / max(1, sum(1 for x in v if x == x)))
                for n, v in resolvable.items()
            },
        }
        per_noise.append(row)
        print(f"noise={noise}: resolvable "
              f"3way={row['resolvable_mean_accuracy']['three_way_llm']:.3f} "
              f"rule={row['resolvable_mean_accuracy']['confidence_rule']:.3f} "
              f"lww={row['resolvable_mean_accuracy']['last_writer_wins']:.3f}")

    result = {
        "note": (
            "post-hoc, exploratory: source-signal-noise sensitivity sweep over the "
            "identical locked-run scenarios (seed 2026, n=60). ThreeWayLLMMerge calls "
            "are cache hits against the locked run (facts unchanged; only ground-truth "
            "scoring varies with noise). NOT part of ANALYSIS_PLAN.md confirmatory tests."
        ),
        "run_metadata": {"seed": LOCKED_SEED, "spans": LOCKED_SPANS, "n_per_span": N_PER_SPAN,
                         "model": MODEL, "noise_levels": NOISE_LEVELS, "locked_noise": 0.15},
        "per_noise": per_noise,
    }
    out_path = OUT_DIR / "noise_sensitivity.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
