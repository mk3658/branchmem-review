import hashlib

import numpy as np
import pandas as pd
import pytest

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.mab_extension import build_branch_scenario_from_mab_row, parse_context_facts
from branchmem.benchmark.scenario_generator import ScenarioConfig, ScenarioGenerator
from branchmem.conflict.embedding_detector import EmbeddingConflictDetector
from branchmem.conflict.llm_judge_detector import LLMJudgeConflictDetector
from branchmem.conflict.nli_detector import NLIConflictDetector
from branchmem.evaluation.metrics import score_detector, score_downstream
from branchmem.llm.mock_backend import MockBackend
from branchmem.merge.branch_discard import BranchDiscard
from branchmem.merge.last_writer_wins import LastWriterWins
from branchmem.merge.naive_concat import NaiveConcat
from branchmem.merge.three_way_llm import ThreeWayLLMMerge


# --- scenario generator -------------------------------------------------------


@pytest.fixture
def generator():
    return ScenarioGenerator()


def test_generate_is_deterministic_given_seed(generator):
    config = ScenarioConfig(n_resolvable=2, n_ambiguous=1)
    batch1 = generator.generate(5, config, seed=123)
    batch2 = generator.generate(5, config, seed=123)
    for s1, s2 in zip(batch1, batch2):
        assert [f.value for f in s1.branch_a.facts] == [f.value for f in s2.branch_a.facts]
        assert [f.value for f in s1.branch_b.facts] == [f.value for f in s2.branch_b.facts]
        assert len(s1.conflict_pairs) == len(s2.conflict_pairs)


def test_generate_ten_scenarios_have_expected_conflict_mix(generator):
    config = ScenarioConfig(n_resolvable=2, n_ambiguous=1)
    scenarios = generator.generate(10, config, seed=7)
    assert len(scenarios) == 10
    for s in scenarios:
        resolvable = [p for p in s.conflict_pairs if p.conflict_type == "resolvable"]
        ambiguous = [p for p in s.conflict_pairs if p.conflict_type == "ambiguous"]
        assert len(resolvable) == config.n_resolvable
        assert len(ambiguous) == config.n_ambiguous
        assert all(p.ground_truth_value is not None for p in resolvable)
        assert all(p.ground_truth_value is None for p in ambiguous)


# --- downstream questions -------------------------------------------------------


def test_conflicts_reuse_real_ancestor_context_where_possible(generator):
    # Regression test: an earlier version seeded used_keys with every
    # ancestor key BEFORE picking conflicts, which made the "prefer an
    # existing ancestor key" path in _sample_conflict_key permanently
    # unreachable -- every conflict silently became a brand-new key with NO
    # ancestor value, so ThreeWayLLMMerge never actually got real
    # three-way (ancestor-aware) context. Caught via a real pilot run, not
    # by the offline unit tests, since the offline tests didn't check
    # whether the *generator* itself was wiring up ancestor context, only
    # that BranchSimulator's already-correct provenance chains worked given
    # hand-built inputs.
    from branchmem.merge.base import current_facts_by_key, find_collisions

    config = ScenarioConfig(n_ancestor_facts=4, n_orthogonal_a=1, n_orthogonal_b=1, n_resolvable=2, n_ambiguous=1)
    scenarios = generator.generate(20, config, seed=2026)
    n_with_ancestor_context = 0
    n_total_conflicts = 0
    for s in scenarios:
        ancestor_facts = current_facts_by_key(s.ancestor)
        collisions = find_collisions(s.branch_a, s.branch_b)
        n_total_conflicts += len(collisions)
        n_with_ancestor_context += sum(1 for key in collisions if key in ancestor_facts)
    assert n_total_conflicts > 0
    assert n_with_ancestor_context / n_total_conflicts > 0.5


def test_downstream_questions_cover_all_categories(generator):
    config = ScenarioConfig(n_orthogonal_a=2, n_orthogonal_b=2, n_resolvable=2, n_ambiguous=1)
    scenario = generator.generate(1, config, seed=1)[0]
    questions = generate_downstream_questions(scenario)
    categories = {q.category for q in questions}
    assert categories == {"orthogonal", "resolvable", "ambiguous"}
    n_orthogonal = sum(1 for q in questions if q.category == "orthogonal")
    assert n_orthogonal == config.n_orthogonal_a + config.n_orthogonal_b


# --- end-to-end: all 4 merge strategies x 10 scenarios, mock backend --------------


def test_end_to_end_all_strategies_on_ten_scenarios(generator):
    config = ScenarioConfig(n_orthogonal_a=1, n_orthogonal_b=1, n_resolvable=2, n_ambiguous=1)
    scenarios = generator.generate(10, config, seed=99)
    backend = MockBackend()  # no canned responses -> every LLM conflict gets flagged unresolved
    strategies = [
        LastWriterWins(),
        NaiveConcat(),
        BranchDiscard(policy="always_b"),
        BranchDiscard(policy="fewer_updates"),
        ThreeWayLLMMerge(backend=backend),
    ]
    for scenario in scenarios:
        questions = generate_downstream_questions(scenario)
        assert len(questions) > 0
        for strategy in strategies:
            result = strategy.merge(scenario.ancestor, scenario.branch_a, scenario.branch_b)
            assert result.strategy_name == strategy.name
            score = score_downstream(questions, result)
            assert 0.0 <= score.accuracy <= 1.0


def _hash_vec(text: str) -> np.ndarray:
    digest = hashlib.sha256(text.encode()).digest()
    return np.frombuffer(digest[:8], dtype=np.uint8).astype(float)


def test_end_to_end_all_detectors_on_ten_scenarios(generator):
    config = ScenarioConfig(n_orthogonal_a=1, n_orthogonal_b=1, n_resolvable=2, n_ambiguous=1)
    scenarios = generator.generate(10, config, seed=55)
    fact_lookup = {}
    for s in scenarios:
        for f in s.branch_a.facts + s.branch_b.facts:
            fact_lookup[f.fact_id] = f

    embedding_detector = EmbeddingConflictDetector(threshold=0.9, embed_fn=lambda texts: np.array([_hash_vec(t) for t in texts]))
    nli_detector = NLIConflictDetector(
        predict_fn=lambda pairs: [{"contradiction": 0.5, "entailment": 0.3, "neutral": 0.2} for _ in pairs]
    )
    llm_detector = LLMJudgeConflictDetector(backend=MockBackend(canned_responses={"conflict": '{"is_conflict": true}'}))

    for detector in (embedding_detector, nli_detector, llm_detector):
        judgments = []
        pairs = []
        for s in scenarios:
            for pair in s.conflict_pairs:
                fact_a, fact_b = fact_lookup[pair.fact_a_id], fact_lookup[pair.fact_b_id]
                judgments.append(detector.detect(fact_a, fact_b))
                pairs.append(pair)
        score = score_detector(pairs, judgments, detector.name)
        assert score.n_pairs == len(pairs)
        assert score.mean_latency_s >= 0.0


# --- MAB extension parser (offline, no network) --------------------------------


def test_parse_context_facts_handles_known_templates():
    context = (
        "Here is a list of facts:\n"
        "0. Thomas Kyd was born in the city of London.\n"
        "1. The chairperson of Fatah is Mahmoud Abbas.\n"
        "2. Hines Ward plays the position of wide receiver.\n"
        "3. Some totally unrecognized sentence structure here!\n"
    )
    parsed, n_matched, n_total = parse_context_facts(context)
    assert n_matched == 3
    assert n_total == 4
    by_entity = {f.entity: f for f in parsed}
    assert by_entity["Thomas Kyd"].value == "London"
    assert by_entity["Fatah"].value == "Mahmoud Abbas"


def test_build_branch_scenario_from_mab_row_splits_edit_chain():
    context = (
        "Here is a list of facts:\n"
        "0. Thomas Kyd was born in the city of London.\n"
        "1. Thomas Kyd was born in the city of Leeds.\n"
        "2. Thomas Kyd was born in the city of Paris.\n"
        "3. Hines Ward plays the position of wide receiver.\n"
    )
    row = pd.Series({"context": context, "questions": [], "answers": [], "metadata": {"source": "test"}})
    scenario = build_branch_scenario_from_mab_row(row, scenario_id="mab_test_0", seed=42)
    assert scenario is not None
    assert scenario.source == "mab_extension"
    resolvable = [p for p in scenario.conflict_pairs if p.conflict_type == "resolvable"]
    assert len(resolvable) == 1
    assert resolvable[0].ground_truth_value == "Paris"  # the true chain-final edit


def test_build_branch_scenario_returns_none_for_unparseable_row():
    row = pd.Series({"context": "Here is a list of facts:\n0. gibberish!!\n", "questions": [], "answers": [], "metadata": {}})
    scenario = build_branch_scenario_from_mab_row(row, scenario_id="mab_test_1", seed=1)
    assert scenario is None


def test_build_branch_scenario_respects_max_conflict_keys():
    # All three chains share the same predicate with 6 distinct values total, so
    # a synthetic second-branch value is always available regardless of which
    # single chain max_conflict_keys=1 happens to sample (avoids flakiness from
    # the "no alternative value available" orthogonal fallback in a chain whose
    # predicate has too little value diversity on its own).
    context = (
        "Here is a list of facts:\n"
        "0. Thomas Kyd was born in the city of London.\n"
        "1. Thomas Kyd was born in the city of Leeds.\n"
        "2. Hines Ward was born in the city of Berlin.\n"
        "3. Hines Ward was born in the city of Madrid.\n"
        "4. Bengaluru was born in the city of Paris.\n"
        "5. Bengaluru was born in the city of Rome.\n"
    )
    row = pd.Series({"context": context, "questions": [], "answers": [], "metadata": {"source": "test"}})
    scenario = build_branch_scenario_from_mab_row(row, scenario_id="mab_test_2", seed=7, max_conflict_keys=1)
    assert scenario is not None
    assert len(scenario.conflict_pairs) == 1
    assert scenario.metadata["max_conflict_keys"] == 1
    assert scenario.metadata["n_conflict_keys_used"] == 1
    # all three chains still contribute ancestor context, not just the sampled one
    assert len(scenario.ancestor.facts) == 3
