#!/usr/bin/env python3
"""Phase 6: the LOCKED full experiment run. Uses exactly the configuration
preregistered in ANALYSIS_PLAN.md — do not change parameters here without
logging the change and reason in PROGRESS.md's "Deviations" section.
"""

from __future__ import annotations

import json
from pathlib import Path

from branchmem.evaluation.plots import plot_accuracy_by_strategy
from branchmem.evaluation.runner import run_full_benchmark, write_results
from branchmem.evaluation.stats import holm_bonferroni, paired_comparison
from branchmem.llm.base import build_backend
from branchmem.utils.logging import get_logger
from branchmem.utils.seeding import set_all_seeds

logger = get_logger("run_experiment")

# Locked per ANALYSIS_PLAN.md sec.1 and sec.5.
SEED = 2026
N_SCENARIOS = 60
MODEL = "gpt-5.4-nano"
DIVERGENCE_SPANS = [4.0, 10.0, 20.0]
EMBEDDING_THRESHOLD = 0.80
NLI_THRESHOLD = 0.20
MIN_EFFECT_SIZE = 0.10  # ANALYSIS_PLAN.md sec.2
F1_TOLERANCE = 0.05  # ANALYSIS_PLAN.md sec.2 (H3)
ALPHA = 0.05

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "final"

# H1 and H2 confirmatory comparisons, in the order tested (ANALYSIS_PLAN.md sec.2/4).
CONFIRMATORY_COMPARISONS = [
    ("H1", "three_way_llm", "last_writer_wins"),
    ("H1", "three_way_llm", "naive_concat"),
    ("H2", "three_way_llm", "branch_discard_always_b"),
    ("H2", "three_way_llm", "branch_discard_fewer_updates"),
]


def main() -> None:
    set_all_seeds(SEED)
    llm_config = {"backend": "openai_compatible", "model": MODEL, "cache_dir": "llm_cache",
                  "temperature": 0.0, "max_tokens": 1024}
    backend = build_backend(llm_config)

    logger.info("Starting locked Phase 6 run: n_scenarios=%d model=%s seed=%d", N_SCENARIOS, MODEL, SEED)
    results, detector_scores, all_pairs = run_full_benchmark(
        backend=backend,
        n_scenarios=N_SCENARIOS,
        seed=SEED,
        divergence_spans=DIVERGENCE_SPANS,
        embedding_threshold=EMBEDDING_THRESHOLD,
        nli_threshold=NLI_THRESHOLD,
    )

    run_metadata = {
        "seed": SEED, "n_scenarios": N_SCENARIOS, "model": MODEL,
        "divergence_spans": DIVERGENCE_SPANS, "n_conflict_pairs": len(all_pairs),
    }
    write_results(results, detector_scores, OUT_DIR, run_metadata)
    plot_accuracy_by_strategy(results, OUT_DIR / "accuracy_by_strategy.png")

    # --- H1/H2: preregistered paired tests + Holm-Bonferroni correction ---
    comparisons = []
    p_values = []
    for hyp, name_a, name_b in CONFIRMATORY_COMPARISONS:
        a = [r.accuracy_by_strategy[name_a] for r in results]
        b = [r.accuracy_by_strategy[name_b] for r in results]
        test_result = paired_comparison(name_a, a, name_b, b, seed=SEED)
        comparisons.append((hyp, test_result))
        p_values.append(test_result.p_value)

    reject_flags, adjusted_p = holm_bonferroni(p_values, alpha=ALPHA)

    h1_results, h2_results = [], []
    for (hyp, test_result), reject, adj_p in zip(comparisons, reject_flags, adjusted_p):
        supported = bool(reject) and test_result.ci_low > MIN_EFFECT_SIZE
        entry = {
            "comparison": f"{test_result.name_a}_vs_{test_result.name_b}",
            "mean_diff": test_result.mean_diff,
            "sd_diff": test_result.sd_diff,
            "ci_low": test_result.ci_low,
            "ci_high": test_result.ci_high,
            "wilcoxon_statistic": test_result.wilcoxon_statistic,
            "raw_p_value": test_result.p_value,
            "holm_adjusted_p_value": adj_p,
            "reject_null_at_corrected_alpha": bool(reject),
            "meets_min_effect_size": test_result.ci_low > MIN_EFFECT_SIZE,
            "supported": supported,
        }
        (h1_results if hyp == "H1" else h2_results).append(entry)
        logger.info(
            "%s %s: mean_diff=%.3f [%.3f, %.3f] adj_p=%.4f supported=%s",
            hyp, entry["comparison"], entry["mean_diff"], entry["ci_low"], entry["ci_high"], adj_p, supported,
        )

    h1_supported = all(e["supported"] for e in h1_results)
    h2_supported = all(e["supported"] for e in h2_results)

    # --- H3: cheap detectors vs. llm_judge, fixed tolerance band ---
    llm_judge_f1 = detector_scores["llm_judge"].f1
    h3_results = {}
    for name in ("embedding_threshold", "nli"):
        gap = abs(detector_scores[name].f1 - llm_judge_f1)
        h3_results[name] = {
            "f1": detector_scores[name].f1,
            "llm_judge_f1": llm_judge_f1,
            "gap": gap,
            "within_tolerance": gap <= F1_TOLERANCE,
            "supported": gap <= F1_TOLERANCE,
        }
        logger.info("H3 %s: F1=%.3f gap=%.3f within_tolerance=%s", name, detector_scores[name].f1, gap, gap <= F1_TOLERANCE)

    stats_output = {
        "run_metadata": run_metadata,
        "min_effect_size": MIN_EFFECT_SIZE,
        "f1_tolerance": F1_TOLERANCE,
        "alpha": ALPHA,
        "H1_three_way_vs_naive_baselines": {"comparisons": h1_results, "hypothesis_supported": h1_supported},
        "H2_three_way_vs_branch_discard": {"comparisons": h2_results, "hypothesis_supported": h2_supported},
        "H3_cheap_detectors_vs_llm_judge": h3_results,
    }
    (OUT_DIR / "stats_output.json").write_text(json.dumps(stats_output, indent=2))
    logger.info("Wrote %s", OUT_DIR / "stats_output.json")
    logger.info("H1 supported: %s | H2 supported: %s", h1_supported, h2_supported)


if __name__ == "__main__":
    main()
