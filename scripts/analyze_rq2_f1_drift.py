#!/usr/bin/env python3
"""Investigates the RQ2 judge F1 drift the review flagged (action item #6,
`paper/reviews/acl2027_review.md` detailed comment #5): LLM-judge F1 = 0.966
on the pilot set (20 scenarios, 60 pairs) vs 0.986 on the locked set (60
scenarios, 180 pairs).

No new API calls: regenerates the exact pilot and locked scenario sets
(same seeds/configs as `scripts/run_pilot.py` and `scripts/run_experiment.py`)
and re-runs all three detectors, which hits `llm_cache/` 100% (every one of
these calls was already made for the original pilot/locked runs) plus the
two offline models (embedding, NLI), which are deterministic given fixed
weights and need no network/API access.

Two things this script establishes, both reported honestly regardless of
which way they cut:

1. Whether every scored pair is a true conflict by construction (no
   orthogonal key is ever shared between the two branches in
   `scenario_generator.py` -- see `BranchSimulator.diverge`), which would
   mean precision=1.0 for every detector at every threshold is structural,
   not evidence of detector specificity.
2. A bootstrap 95% CI (2000 resamples over pairs, seeded) on each
   detector's F1 and on the |F1_detector - F1_judge| gap, for both the
   pilot and locked sets, to see whether the pilot-to-locked judge F1 move
   (0.966 -> 0.986) is within ordinary resampling noise for the *larger*
   set's own size, or a larger shift than that.

Writes `results/final/rq2_f1_bootstrap.json`.
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

from branchmem.benchmark.scenario_generator import ScenarioConfig, ScenarioGenerator
from branchmem.conflict.embedding_detector import EmbeddingConflictDetector
from branchmem.conflict.llm_judge_detector import LLMJudgeConflictDetector
from branchmem.conflict.nli_detector import NLIConflictDetector
from branchmem.evaluation.metrics import score_detector
from branchmem.llm.base import build_backend

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "final"

PILOT_SEED = 2026
LOCKED_SEED = 2026
LOCKED_SPANS = [4.0, 10.0, 20.0]
MODEL = "gpt-5.4-nano"


def build_pilot_pairs():
    generator = ScenarioGenerator()
    config = ScenarioConfig(n_orthogonal_a=1, n_orthogonal_b=1, n_resolvable=2, n_ambiguous=1, divergence_span=8.0)
    scenarios = generator.generate(20, config, seed=PILOT_SEED)
    fact_lookup = {}
    for s in scenarios:
        for f in s.branch_a.facts + s.branch_b.facts:
            fact_lookup[f.fact_id] = f
    pairs = [p for s in scenarios for p in s.conflict_pairs]
    return pairs, fact_lookup


def build_locked_pairs():
    generator = ScenarioGenerator()
    n_per_span = 60 // len(LOCKED_SPANS)
    scenarios = []
    for i, span in enumerate(LOCKED_SPANS):
        # No scenario_config_kwargs override, matching scripts/run_experiment.py's
        # actual call (run_full_benchmark(..., scenario_config_kwargs=None)) --
        # i.e. ScenarioConfig defaults (n_orthogonal_a=n_orthogonal_b=2), NOT the
        # n_orthogonal_a=n_orthogonal_b=1 written in ANALYSIS_PLAN.md sec.1 (a
        # documentation discrepancy, disclosed in paper/acl_latex.tex's
        # "Locked Scenario Configuration" appendix). Reproducing the code's
        # actual behavior here, not the plan document's prose, is what makes
        # this a faithful replay.
        config = ScenarioConfig(divergence_span=span)
        scenarios.extend(generator.generate(n_per_span, config, seed=LOCKED_SEED + i))
    fact_lookup = {}
    for s in scenarios:
        for f in s.branch_a.facts + s.branch_b.facts:
            fact_lookup[f.fact_id] = f
    pairs = [p for s in scenarios for p in s.conflict_pairs]
    return pairs, fact_lookup


def bootstrap_f1(pairs, judgments, n_resamples=2000, seed=2026):
    """Bootstrap CI on F1 by resampling (pair, judgment) rows with replacement."""
    rng = random.Random(seed)
    n = len(pairs)
    idxs = list(range(n))
    f1s = []
    for _ in range(n_resamples):
        sample_idx = [idxs[rng.randrange(n)] for _ in range(n)]
        sample_pairs = [pairs[i] for i in sample_idx]
        sample_judgments = [judgments[i] for i in sample_idx]
        s = score_detector(sample_pairs, sample_judgments, "resample")
        if s.f1 == s.f1:  # not NaN
            f1s.append(s.f1)
    f1s.sort()
    lo = f1s[int(0.025 * len(f1s))]
    hi = f1s[int(0.975 * len(f1s)) - 1]
    return lo, hi


def analyze(name: str, pairs, fact_lookup, backend):
    n_conflict = sum(1 for p in pairs if p.is_conflict)
    n_non_conflict = sum(1 for p in pairs if not p.is_conflict)

    detectors = {
        "embedding_threshold": EmbeddingConflictDetector(threshold=0.80),
        "nli": NLIConflictDetector(contradiction_threshold=0.20),
        "llm_judge": LLMJudgeConflictDetector(backend=backend),
    }
    result = {"n_pairs": len(pairs), "n_conflict_pairs": n_conflict, "n_non_conflict_pairs": n_non_conflict}
    judgments_by_detector = {}
    for dname, detector in detectors.items():
        judgments = [detector.detect(fact_lookup[p.fact_a_id], fact_lookup[p.fact_b_id]) for p in pairs]
        judgments_by_detector[dname] = judgments
        score = score_detector(pairs, judgments, dname)
        lo, hi = bootstrap_f1(pairs, judgments, seed=2026)
        result[dname] = {
            "precision": score.precision, "recall": score.recall, "f1": score.f1,
            "f1_95pct_bootstrap_ci": [lo, hi],
        }

    judge_f1 = result["llm_judge"]["f1"]
    for dname in ("embedding_threshold", "nli"):
        gap = abs(result[dname]["f1"] - judge_f1)
        # Bootstrap the gap directly (paired resample: same resample index for both).
        rng = random.Random(2026)
        n = len(pairs)
        idxs = list(range(n))
        gaps = []
        for _ in range(2000):
            sample_idx = [idxs[rng.randrange(n)] for _ in range(n)]
            sample_pairs = [pairs[i] for i in sample_idx]
            det_j = [judgments_by_detector[dname][i] for i in sample_idx]
            judge_j = [judgments_by_detector["llm_judge"][i] for i in sample_idx]
            s_det = score_detector(sample_pairs, det_j, dname)
            s_judge = score_detector(sample_pairs, judge_j, "llm_judge")
            if s_det.f1 == s_det.f1 and s_judge.f1 == s_judge.f1:
                gaps.append(abs(s_det.f1 - s_judge.f1))
        gaps.sort()
        lo, hi = gaps[int(0.025 * len(gaps))], gaps[int(0.975 * len(gaps)) - 1]
        result[dname]["gap_vs_judge"] = gap
        result[dname]["gap_95pct_bootstrap_ci"] = [lo, hi]

    print(f"[{name}] n_pairs={len(pairs)} n_conflict={n_conflict} n_non_conflict={n_non_conflict}")
    for dname in detectors:
        print(f"  {dname}: F1={result[dname]['f1']:.4f} CI={result[dname]['f1_95pct_bootstrap_ci']}")
    return result


def main() -> None:
    llm_config = {"backend": "openai_compatible", "model": MODEL, "cache_dir": "llm_cache",
                  "temperature": 0.0, "max_tokens": 1024}
    backend = build_backend(llm_config)

    pilot_pairs, pilot_lookup = build_pilot_pairs()
    locked_pairs, locked_lookup = build_locked_pairs()

    pilot_result = analyze("pilot", pilot_pairs, pilot_lookup, backend)
    locked_result = analyze("locked", locked_pairs, locked_lookup, backend)

    out = {
        "note": (
            "Investigation of the pilot-to-locked LLM-judge F1 drift (review action "
            "item #6). No new API calls -- regenerates the exact pilot/locked "
            "scenario sets and re-runs detectors, hitting llm_cache/ for every LLM "
            "call. Bootstrap CIs (2000 resamples, seed=2026) computed over pairs."
        ),
        "pilot": pilot_result,
        "locked": locked_result,
    }
    (OUT_DIR / "rq2_f1_bootstrap.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_DIR / 'rq2_f1_bootstrap.json'}")


if __name__ == "__main__":
    main()
