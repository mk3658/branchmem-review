from branchmem.eval.abstention_metrics import compute_abstention_metrics, metrics_from_counts


def _detail(category, correct, got_resolution):
    return {"category": category, "correct": correct, "got_resolution": got_resolution}


def test_perfect_commit_and_flag_gives_full_coverage_zero_risk():
    detail = [
        _detail("resolvable", True, "kept_from_a"),
        _detail("resolvable", True, "kept_from_b"),
        _detail("ambiguous", True, "flagged_unresolved"),
    ]
    m = compute_abstention_metrics(detail)
    assert m.n_questions == 3
    assert m.commit_rate == 2 / 3
    assert m.conditional_on_commit_accuracy == 1.0
    assert m.wrong_commit_rate == 0.0
    assert m.flag_recall == 1.0  # the one ambiguous item was flagged
    assert m.flag_precision == 1.0  # the one flag was on an ambiguous item
    assert m.risk == 0.0


def test_wrong_commit_is_distinguished_from_abstention():
    detail = [
        _detail("resolvable", False, "kept_from_a"),  # wrong commit
        _detail("resolvable", False, "flagged_unresolved"),  # abstained (miss on resolvable)
    ]
    m = compute_abstention_metrics(detail)
    assert m.n_committed == 1
    assert m.n_wrong_committed == 1
    assert m.n_flagged == 1
    assert m.conditional_on_commit_accuracy == 0.0
    assert m.wrong_commit_rate == 0.5
    # False-positive flag: this resolvable item should have been committed.
    assert m.n_resolvable_or_orthog_flagged == 1


def test_zero_commits_gives_undefined_conditional_accuracy_not_zero():
    detail = [_detail("resolvable", False, "flagged_unresolved") for _ in range(5)]
    m = compute_abstention_metrics(detail)
    assert m.n_committed == 0
    assert m.conditional_on_commit_accuracy is None
    assert m.risk is None
    assert m.abstention_rate == 1.0


def test_expected_utility_penalizes_wrong_commits_more_under_higher_cost():
    detail = [_detail("resolvable", False, "kept_from_a")]  # one wrong commit
    m = compute_abstention_metrics(detail, utility_costs={"cheap": -2.0, "expensive": -10.0})
    assert m.expected_utility["cheap"] == -2.0
    assert m.expected_utility["expensive"] == -10.0
    assert m.expected_utility["expensive"] < m.expected_utility["cheap"]


def test_orthogonal_excluded_by_default():
    detail = [
        _detail("orthogonal", True, "kept_from_a"),
        _detail("resolvable", True, "kept_from_a"),
    ]
    m = compute_abstention_metrics(detail)
    assert m.n_questions == 1  # orthogonal excluded from abstention accounting


def test_metrics_from_counts_matches_direct_computation_shape():
    m = metrics_from_counts(n_questions=120, n_committed=119, n_correct_committed=102)
    assert m.n_wrong_committed == 17
    assert m.n_flagged == 1
    assert round(m.conditional_on_commit_accuracy, 4) == round(102 / 119, 4)
    assert m.flag_precision is None  # not derivable from aggregate counts alone


def test_metrics_from_counts_handles_full_abstention():
    m = metrics_from_counts(n_questions=230, n_committed=0, n_correct_committed=0)
    assert m.conditional_on_commit_accuracy is None
    assert m.abstention_rate == 1.0
    assert m.expected_utility["wrong_commit_-5"] == -1.0  # all 230 flagged, cost -1 each, no wrong commits
