from branchmem.memory.schemas import (
    ConflictPair,
    MemoryBranch,
    MemoryFact,
    MergeResult,
    Resolution,
    ResolvedFact,
)
from branchmem.memory.store import MemoryStore


def test_memory_fact_defaults_and_key():
    fact = MemoryFact(
        entity="alice", predicate="location", value="paris", branch_id="b1", timestamp=1.0
    )
    assert fact.fact_id.startswith("fact_")
    assert fact.key() == ("alice", "location")
    assert fact.source == "observation"
    assert fact.common_ancestor_id is None


def test_conflict_pair_and_merge_result():
    pair = ConflictPair(fact_a_id="a", fact_b_id="b", is_conflict=True, conflict_type="resolvable")
    assert pair.detector_predictions == {}

    kept = ResolvedFact(
        fact=MemoryFact(entity="x", predicate="y", value="z", branch_id="b1", timestamp=1.0),
        resolution=Resolution.KEPT,
    )
    dropped = ResolvedFact(
        fact=MemoryFact(entity="x", predicate="w", value="q", branch_id="b1", timestamp=1.0),
        resolution=Resolution.DROPPED,
    )
    result = MergeResult(strategy_name="test", resulting_facts=[kept, dropped])
    assert result.kept_facts() == [kept.fact]


def test_store_roundtrip():
    branch = MemoryBranch(branch_id="root")
    branch.facts.append(
        MemoryFact(entity="alice", predicate="location", value="paris", branch_id="root", timestamp=1.0)
    )
    with MemoryStore() as store:
        store.create_branch(branch)
        fetched = store.get_branch("root")
        assert fetched is not None
        assert len(fetched.facts) == 1
        assert fetched.facts[0].value == "paris"

        fetched_fact = store.get_fact(fetched.facts[0].fact_id)
        assert fetched_fact is not None
        assert fetched_fact.entity == "alice"

        by_key = store.get_facts_by_key("root", "alice", "location")
        assert len(by_key) == 1

        assert store.get_branch("nonexistent") is None
