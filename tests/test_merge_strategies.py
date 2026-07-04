import json

import pytest

from branchmem.llm.mock_backend import MockBackend
from branchmem.memory.branch_sim import BranchSimulator, UpdateSpec
from branchmem.memory.schemas import Resolution
from branchmem.merge.branch_discard import BranchDiscard
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.last_writer_wins import LastWriterWins
from branchmem.merge.naive_concat import NaiveConcat
from branchmem.merge.raw_text_llm import RawTextLLMMerge
from branchmem.merge.three_way_llm import ThreeWayLLMMerge
from branchmem.merge.two_way_llm import TwoWayLLMMerge


@pytest.fixture
def sim():
    return BranchSimulator()


@pytest.fixture
def scenario(sim):
    """One orthogonal update per branch, plus one resolvable conflict on 'location'."""
    ancestor = sim.create_common_ancestor(
        facts=[("alice", "location", "boston"), ("alice", "job", "engineer")], timestamp=0.0
    )
    updates_a = [
        UpdateSpec(entity="alice", predicate="location", value="chicago", timestamp=5.0,
                    divergence_type="resolvable_conflict", ground_truth_value="chicago"),
        UpdateSpec(entity="alice", predicate="pet", value="cat", timestamp=2.0, divergence_type="orthogonal"),
    ]
    updates_b = [
        UpdateSpec(entity="alice", predicate="location", value="denver", timestamp=3.0,
                    divergence_type="resolvable_conflict", ground_truth_value="chicago"),
        UpdateSpec(entity="alice", predicate="hobby", value="chess", timestamp=2.0, divergence_type="orthogonal"),
    ]
    branch_a, branch_b, pairs = sim.diverge(ancestor, updates_a, updates_b, fork_point_timestamp=1.0)
    return ancestor, branch_a, branch_b, pairs


def _values_by_predicate(facts):
    return {f.predicate: f.value for f in facts}


def test_last_writer_wins_picks_later_timestamp(scenario):
    ancestor, branch_a, branch_b, _ = scenario
    result = LastWriterWins().merge(ancestor, branch_a, branch_b)
    kept = _values_by_predicate(result.kept_facts())
    # branch_a's location update (t=5.0) is later than branch_b's (t=3.0)
    assert kept["location"] == "chicago"
    assert kept["pet"] == "cat"
    assert kept["hobby"] == "chess"
    assert kept["job"] == "engineer"

    location_resolutions = [rf for rf in result.resulting_facts if rf.fact.predicate == "location"]
    assert {rf.resolution for rf in location_resolutions} == {Resolution.KEPT, Resolution.DROPPED}


def test_naive_concat_keeps_both_conflicting_values(scenario):
    ancestor, branch_a, branch_b, _ = scenario
    result = NaiveConcat().merge(ancestor, branch_a, branch_b)
    location_values = sorted(f.value for f in result.kept_facts() if f.predicate == "location")
    assert location_values == ["chicago", "denver"]
    assert all(rf.resolution == Resolution.KEPT for rf in result.resulting_facts)


def test_branch_discard_always_b_drops_branch_b_entirely(scenario):
    ancestor, branch_a, branch_b, _ = scenario
    result = BranchDiscard(policy="always_b").merge(ancestor, branch_a, branch_b)
    kept = _values_by_predicate(result.kept_facts())
    assert kept["location"] == "chicago"
    assert kept["pet"] == "cat"
    assert "hobby" not in kept  # branch B's orthogonal update was lost too
    dropped_predicates = {rf.fact.predicate for rf in result.resulting_facts if rf.resolution == Resolution.DROPPED}
    assert dropped_predicates == {"location", "hobby"}


def test_branch_discard_no_collision_keeps_everything(sim):
    ancestor = sim.create_common_ancestor(facts=[("alice", "location", "boston")], timestamp=0.0)
    updates_a = [UpdateSpec(entity="alice", predicate="pet", value="cat", timestamp=2.0, divergence_type="orthogonal")]
    updates_b = [UpdateSpec(entity="alice", predicate="hobby", value="chess", timestamp=2.0, divergence_type="orthogonal")]
    branch_a, branch_b, _ = sim.diverge(ancestor, updates_a, updates_b, fork_point_timestamp=1.0)
    result = BranchDiscard(policy="always_b").merge(ancestor, branch_a, branch_b)
    kept = _values_by_predicate(result.kept_facts())
    assert kept["pet"] == "cat"
    assert kept["hobby"] == "chess"
    assert all(rf.resolution == Resolution.KEPT for rf in result.resulting_facts)


def test_three_way_llm_merge_resolves_conflict_with_ancestor_context(scenario):
    ancestor, branch_a, branch_b, _ = scenario
    canned = {
        "location": json.dumps(
            {
                "resolutions": [
                    {
                        "entity": "alice",
                        "predicate": "location",
                        "resolution": "kept_from_a",
                        "value": "chicago",
                        "justification": "branch A's update is a plausible correction; branch B's is stale",
                    }
                ]
            }
        )
    }
    backend = MockBackend(canned_responses=canned)
    result = ThreeWayLLMMerge(backend=backend).merge(ancestor, branch_a, branch_b)
    kept = _values_by_predicate(result.kept_facts())
    assert kept["location"] == "chicago"
    assert kept["pet"] == "cat"
    assert kept["hobby"] == "chess"
    assert result.unresolved_conflicts == []


def test_three_way_llm_merge_flags_unresolvable_when_llm_says_so(scenario):
    ancestor, branch_a, branch_b, _ = scenario
    canned = {
        "location": json.dumps(
            {
                "resolutions": [
                    {
                        "entity": "alice",
                        "predicate": "location",
                        "resolution": "flagged_unresolved",
                        "value": None,
                        "justification": "no way to tell which update is more recent or correct",
                    }
                ]
            }
        )
    }
    backend = MockBackend(canned_responses=canned)
    result = ThreeWayLLMMerge(backend=backend).merge(ancestor, branch_a, branch_b)
    flagged = [rf for rf in result.resulting_facts if rf.resolution == Resolution.FLAGGED_UNRESOLVED]
    assert len(flagged) == 1
    assert flagged[0].fact.predicate == "location"
    assert len(result.unresolved_conflicts) == 1


def test_three_way_llm_merge_handles_unparseable_response_by_flagging(scenario):
    ancestor, branch_a, branch_b, _ = scenario
    backend = MockBackend(canned_responses={"location": "not valid json at all"})
    result = ThreeWayLLMMerge(backend=backend).merge(ancestor, branch_a, branch_b)
    flagged = [rf for rf in result.resulting_facts if rf.resolution == Resolution.FLAGGED_UNRESOLVED]
    assert len(flagged) == 1


# --- ConfidenceRuleMerge (deterministic, no LLM) --------------------------------


def test_confidence_rule_picks_higher_confidence_branch(sim):
    ancestor = sim.create_common_ancestor(facts=[("alice", "location", "boston")], timestamp=0.0)
    updates_a = [
        UpdateSpec(entity="alice", predicate="location", value="chicago", timestamp=2.0,
                   source="user", confidence=0.95, divergence_type="resolvable_conflict",
                   ground_truth_value="chicago")
    ]
    updates_b = [
        UpdateSpec(entity="alice", predicate="location", value="denver", timestamp=3.0,
                   source="inference", confidence=0.55, divergence_type="resolvable_conflict",
                   ground_truth_value="chicago")
    ]
    branch_a, branch_b, _ = sim.diverge(ancestor, updates_a, updates_b, fork_point_timestamp=1.0)
    result = ConfidenceRuleMerge().merge(ancestor, branch_a, branch_b)
    kept = _values_by_predicate(result.kept_facts())
    assert kept["location"] == "chicago"


def test_confidence_rule_flags_on_exact_tie(scenario):
    # default scenario fixture gives both branches confidence=1.0 (no source override)
    ancestor, branch_a, branch_b, _ = scenario
    result = ConfidenceRuleMerge().merge(ancestor, branch_a, branch_b)
    flagged = [rf for rf in result.resulting_facts if rf.resolution == Resolution.FLAGGED_UNRESOLVED]
    assert len(flagged) == 1
    assert flagged[0].fact.predicate == "location"


# --- TwoWayLLMMerge (ablation: no ancestor in prompt) ---------------------------


def test_two_way_llm_merge_resolves_without_ancestor(scenario):
    ancestor, branch_a, branch_b, _ = scenario
    canned = {
        "location": json.dumps(
            {
                "resolutions": [
                    {
                        "entity": "alice", "predicate": "location", "resolution": "kept_from_a",
                        "value": "chicago", "justification": "branch A has higher confidence",
                    }
                ]
            }
        )
    }
    backend = MockBackend(canned_responses=canned)
    result = TwoWayLLMMerge(backend=backend).merge(ancestor, branch_a, branch_b)
    kept = _values_by_predicate(result.kept_facts())
    assert kept["location"] == "chicago"
    assert kept["pet"] == "cat"
    assert kept["hobby"] == "chess"


def test_two_way_llm_merge_prompt_omits_ancestor(scenario):
    ancestor, branch_a, branch_b, _ = scenario
    captured = {}

    class RecordingBackend(MockBackend):
        def complete(self, prompt, system=None, use_cache=True):
            captured["prompt"] = prompt
            return super().complete(prompt, system=system, use_cache=use_cache)

    backend = RecordingBackend()
    TwoWayLLMMerge(backend=backend).merge(ancestor, branch_a, branch_b)
    assert "ancestor" not in captured["prompt"].lower()


# --- RawTextLLMMerge (ablation: no ancestor, source, confidence, or timestamp) --


def test_raw_text_llm_merge_resolves_from_raw_values_only(scenario):
    ancestor, branch_a, branch_b, _ = scenario
    canned = {
        "location": json.dumps(
            {
                "resolutions": [
                    {
                        "entity": "alice", "predicate": "location", "resolution": "kept_from_a",
                        "value": "chicago", "justification": "chicago reads as the more specific correction",
                    }
                ]
            }
        )
    }
    backend = MockBackend(canned_responses=canned)
    result = RawTextLLMMerge(backend=backend).merge(ancestor, branch_a, branch_b)
    kept = _values_by_predicate(result.kept_facts())
    assert kept["location"] == "chicago"
    assert kept["pet"] == "cat"
    assert kept["hobby"] == "chess"
    assert result.unresolved_conflicts == []


def test_raw_text_llm_merge_flags_unresolvable_when_llm_says_so(scenario):
    ancestor, branch_a, branch_b, _ = scenario
    canned = {
        "location": json.dumps(
            {
                "resolutions": [
                    {
                        "entity": "alice", "predicate": "location", "resolution": "flagged_unresolved",
                        "value": None, "justification": "nothing about either raw value suggests one is more correct",
                    }
                ]
            }
        )
    }
    backend = MockBackend(canned_responses=canned)
    result = RawTextLLMMerge(backend=backend).merge(ancestor, branch_a, branch_b)
    flagged = [rf for rf in result.resulting_facts if rf.resolution == Resolution.FLAGGED_UNRESOLVED]
    assert len(flagged) == 1
    assert flagged[0].fact.predicate == "location"
    assert len(result.unresolved_conflicts) == 1


def test_raw_text_llm_merge_prompt_omits_ancestor_source_and_confidence(scenario):
    ancestor, branch_a, branch_b, _ = scenario
    captured = {}

    class RecordingBackend(MockBackend):
        def complete(self, prompt, system=None, use_cache=True):
            captured["prompt"] = prompt
            return super().complete(prompt, system=system, use_cache=use_cache)

    backend = RecordingBackend()
    RawTextLLMMerge(backend=backend).merge(ancestor, branch_a, branch_b)
    prompt_lower = captured["prompt"].lower()
    assert "ancestor" not in prompt_lower
    assert "source" not in prompt_lower
    assert "confidence" not in prompt_lower
    # The raw conflicting values themselves must still be present.
    assert "chicago" in captured["prompt"] and "denver" in captured["prompt"]
