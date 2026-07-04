import pytest

from branchmem.memory.branch_sim import BranchSimulator, DivergenceError, UpdateSpec


@pytest.fixture
def sim():
    return BranchSimulator()


@pytest.fixture
def ancestor(sim):
    return sim.create_common_ancestor(
        facts=[
            ("alice", "location", "boston"),
            ("alice", "job", "engineer"),
        ],
        timestamp=0.0,
    )


def test_fork_copies_ancestor_facts_with_provenance(sim, ancestor):
    branch_a, branch_b = sim.fork(ancestor, fork_point_timestamp=1.0)
    assert branch_a.parent_branch_id == ancestor.branch_id
    assert branch_b.parent_branch_id == ancestor.branch_id
    assert len(branch_a.facts) == len(ancestor.facts) == len(branch_b.facts)
    for fact in branch_a.facts:
        assert fact.common_ancestor_id is not None
        assert fact.branch_id == branch_a.branch_id


def test_orthogonal_updates_produce_no_conflict(sim, ancestor):
    updates_a = [
        UpdateSpec(entity="alice", predicate="pet", value="cat", timestamp=2.0, divergence_type="orthogonal")
    ]
    updates_b = [
        UpdateSpec(entity="alice", predicate="hobby", value="chess", timestamp=2.0, divergence_type="orthogonal")
    ]
    branch_a, branch_b, pairs = sim.diverge(ancestor, updates_a, updates_b, fork_point_timestamp=1.0)
    assert pairs == []
    a_pet = [f for f in branch_a.facts if f.predicate == "pet"]
    b_hobby = [f for f in branch_b.facts if f.predicate == "hobby"]
    assert len(a_pet) == 1 and len(b_hobby) == 1
    assert not any(f.predicate == "hobby" for f in branch_a.facts)


def test_resolvable_conflict_detected_with_ground_truth(sim, ancestor):
    updates_a = [
        UpdateSpec(
            entity="alice", predicate="location", value="chicago", timestamp=2.0,
            divergence_type="resolvable_conflict", ground_truth_value="chicago",
        )
    ]
    updates_b = [
        UpdateSpec(
            entity="alice", predicate="location", value="boston", timestamp=1.5,
            divergence_type="resolvable_conflict", ground_truth_value="chicago",
        )
    ]
    branch_a, branch_b, pairs = sim.diverge(ancestor, updates_a, updates_b, fork_point_timestamp=1.0)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.is_conflict is True
    assert pair.conflict_type == "resolvable"
    assert pair.ground_truth_value == "chicago"


def test_ambiguous_conflict_has_no_ground_truth(sim, ancestor):
    updates_a = [
        UpdateSpec(
            entity="alice", predicate="favorite_color", value="red", timestamp=2.0,
            divergence_type="ambiguous_conflict",
        )
    ]
    updates_b = [
        UpdateSpec(
            entity="alice", predicate="favorite_color", value="blue", timestamp=2.0,
            divergence_type="ambiguous_conflict",
        )
    ]
    _, _, pairs = sim.diverge(ancestor, updates_a, updates_b, fork_point_timestamp=1.0)
    assert len(pairs) == 1
    assert pairs[0].conflict_type == "ambiguous"
    assert pairs[0].ground_truth_value is None


def test_inconsistent_divergence_type_raises(sim, ancestor):
    updates_a = [
        UpdateSpec(entity="alice", predicate="location", value="chicago", timestamp=2.0, divergence_type="orthogonal")
    ]
    updates_b = [
        UpdateSpec(
            entity="alice", predicate="location", value="boston", timestamp=2.0,
            divergence_type="resolvable_conflict", ground_truth_value="chicago",
        )
    ]
    with pytest.raises(DivergenceError):
        sim.diverge(ancestor, updates_a, updates_b, fork_point_timestamp=1.0)


def test_update_supersedes_prior_fact_and_keeps_history(sim, ancestor):
    updates_a = [
        UpdateSpec(entity="alice", predicate="job", value="manager", timestamp=2.0, divergence_type="orthogonal")
    ]
    branch_a = sim.apply_updates(sim.fork(ancestor, 1.0)[0], updates_a)
    job_facts = [f for f in branch_a.facts if f.predicate == "job"]
    assert len(job_facts) == 2  # original "engineer" + new "manager", history kept
    current = max(job_facts, key=lambda f: f.timestamp)
    assert current.value == "manager"
    original = min(job_facts, key=lambda f: f.timestamp)
    assert current.common_ancestor_id == original.fact_id
