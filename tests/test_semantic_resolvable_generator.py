from branchmem.benchmark.semantic_resolvable_generator import generate_semantic_resolvable_scenarios
from branchmem.merge.base import current_facts_by_key


def test_source_and_confidence_are_equal_on_both_branches():
    scenarios = generate_semantic_resolvable_scenarios(n_scenarios=20, seed=1)
    assert len(scenarios) == 20
    for sc in scenarios:
        facts_a = current_facts_by_key(sc.branch_a)
        facts_b = current_facts_by_key(sc.branch_b)
        conflict_keys = {(f.entity, f.predicate) for f in sc.branch_a.facts} & {
            (f.entity, f.predicate) for f in sc.branch_b.facts
        }
        for key in conflict_keys:
            fa, fb = facts_a[key], facts_b[key]
            assert fa.source == fb.source, f"source differs for {sc.scenario_id}"
            assert fa.confidence == fb.confidence, f"confidence differs for {sc.scenario_id}"


def test_ground_truth_is_recoverable_only_from_semantic_content():
    """Ground truth must not always land on the same physical branch (A vs
    B) -- otherwise a strategy could win by a positional shortcut instead of
    reasoning about content."""
    scenarios = generate_semantic_resolvable_scenarios(n_scenarios=60, seed=2)
    correct_is_a_flags = [sc.metadata["correct_is_a"] for sc in scenarios]
    n_a = sum(correct_is_a_flags)
    n_b = len(correct_is_a_flags) - n_a
    assert n_a > 0 and n_b > 0, "correct answer must not always be on the same physical branch"


def test_timestamp_does_not_determine_correctness():
    """Across many scenarios, whether the ground-truth-correct branch also
    has the later timestamp should be close to chance (uninformative),
    matching the category's design requirement."""
    scenarios = generate_semantic_resolvable_scenarios(n_scenarios=200, seed=3)
    later_is_correct = 0
    for sc in scenarios:
        key = sc.conflict_pairs[0].fact_a_id, sc.conflict_pairs[0].fact_b_id
        fact_a = next(f for f in sc.branch_a.facts if f.fact_id == sc.conflict_pairs[0].fact_a_id)
        fact_b = next(f for f in sc.branch_b.facts if f.fact_id == sc.conflict_pairs[0].fact_b_id)
        correct_is_a = sc.metadata["correct_is_a"]
        correct_ts = fact_a.timestamp if correct_is_a else fact_b.timestamp
        other_ts = fact_b.timestamp if correct_is_a else fact_a.timestamp
        if correct_ts > other_ts:
            later_is_correct += 1
    rate = later_is_correct / len(scenarios)
    # Should be close to 50%; a wide but not unbounded tolerance band avoids
    # test flakiness while still catching a real correlation bug.
    assert 0.35 < rate < 0.65, f"timestamp correlates with correctness: rate={rate}"


def test_reproducible_with_same_seed():
    a = generate_semantic_resolvable_scenarios(n_scenarios=10, seed=42)
    b = generate_semantic_resolvable_scenarios(n_scenarios=10, seed=42)
    for sa, sb in zip(a, b):
        assert sa.scenario_id == sb.scenario_id
        assert [f.value for f in sa.branch_a.facts] == [f.value for f in sb.branch_a.facts]


def test_confidence_rule_merge_cannot_solve_above_chance():
    """ConfidenceRuleMerge reads source/confidence, which are equal on both
    branches here -- it must flag every conflict as an exact tie, not
    resolve any of them."""
    from branchmem.merge.confidence_rule import ConfidenceRuleMerge
    from branchmem.memory.schemas import Resolution

    scenarios = generate_semantic_resolvable_scenarios(n_scenarios=20, seed=5)
    strategy = ConfidenceRuleMerge()
    n_flagged = 0
    for sc in scenarios:
        result = strategy.merge(sc.ancestor, sc.branch_a, sc.branch_b)
        flagged = [rf for rf in result.resulting_facts if rf.resolution == Resolution.FLAGGED_UNRESOLVED]
        n_flagged += len(flagged)
    assert n_flagged == len(scenarios), "ConfidenceRuleMerge should flag every equal-confidence tie"
