from branchmem.evaluation.stats import holm_bonferroni, paired_bootstrap_ci, paired_comparison


def test_paired_comparison_detects_a_clear_difference():
    a = [0.9, 0.8, 1.0, 0.9, 0.85, 0.95, 0.9, 0.8, 1.0, 0.9]
    b = [0.5, 0.4, 0.6, 0.5, 0.45, 0.55, 0.5, 0.4, 0.6, 0.5]
    result = paired_comparison("a", a, "b", b)
    assert result.mean_diff > 0.3
    assert result.p_value < 0.05
    assert result.ci_low > 0  # a reliably beats b


def test_paired_comparison_identical_sequences_is_null():
    a = [0.7, 0.6, 0.8, 0.7, 0.9]
    result = paired_comparison("a", a, "a_copy", list(a))
    assert result.mean_diff == 0.0
    assert result.p_value == 1.0


def test_paired_bootstrap_ci_contains_true_mean_diff():
    a = [1.0] * 20
    b = [0.5] * 20
    lo, hi = paired_bootstrap_ci(a, b, n_resamples=500)
    assert lo <= 0.5 <= hi


def test_holm_bonferroni_corrects_multiple_comparisons():
    # one clearly significant, three clearly not
    p_values = [0.001, 0.6, 0.7, 0.8]
    reject, adjusted = holm_bonferroni(p_values, alpha=0.05)
    assert reject[0] is True
    assert all(r is False for r in reject[1:])
    assert all(adj >= p for adj, p in zip(adjusted, p_values))


def test_holm_bonferroni_all_significant():
    p_values = [0.001, 0.002, 0.003]
    reject, _ = holm_bonferroni(p_values, alpha=0.05)
    assert all(reject)
