"""Paired significance tests, effect sizes, and multiple-comparison correction,
per the tests preregistered in ANALYSIS_PLAN.md.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests


@dataclass
class PairedTestResult:
    name_a: str
    name_b: str
    n: int
    mean_diff: float
    sd_diff: float
    wilcoxon_statistic: float
    p_value: float
    ci_low: float
    ci_high: float


def paired_bootstrap_ci(
    a: list[float], b: list[float], n_resamples: int = 2000, seed: int = 2026, alpha: float = 0.05
) -> tuple[float, float]:
    """95% (by default) CI on the mean paired difference a - b, via paired bootstrap."""
    assert len(a) == len(b) and len(a) > 0
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    rng = random.Random(seed)
    resampled_means = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        resampled_means.append(statistics.mean(sample))
    resampled_means.sort()
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    return resampled_means[lo_idx], resampled_means[hi_idx]


def paired_comparison(name_a: str, a: list[float], name_b: str, b: list[float], seed: int = 2026) -> PairedTestResult:
    """Paired Wilcoxon signed-rank test + paired bootstrap CI for a - b."""
    assert len(a) == len(b) and len(a) > 0
    diffs = [x - y for x, y in zip(a, b)]
    if all(d == 0 for d in diffs):
        # scipy.stats.wilcoxon raises on all-zero differences; report a null result explicitly.
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = scipy_stats.wilcoxon(a, b)
    ci_low, ci_high = paired_bootstrap_ci(a, b, seed=seed)
    return PairedTestResult(
        name_a=name_a,
        name_b=name_b,
        n=len(a),
        mean_diff=statistics.mean(diffs),
        sd_diff=statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
        wilcoxon_statistic=float(statistic),
        p_value=float(p_value),
        ci_low=ci_low,
        ci_high=ci_high,
    )


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> tuple[list[bool], list[float]]:
    """Holm-Bonferroni step-down correction. Returns (reject_flags, adjusted_p_values)."""
    reject, adjusted_p, _, _ = multipletests(p_values, alpha=alpha, method="holm")
    return [bool(r) for r in reject], [float(p) for p in adjusted_p]
