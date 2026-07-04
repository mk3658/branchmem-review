import numpy as np

from branchmem.conflict.embedding_detector import EmbeddingConflictDetector
from branchmem.conflict.llm_judge_detector import LLMJudgeConflictDetector
from branchmem.conflict.nli_detector import NLIConflictDetector
from branchmem.llm.mock_backend import MockBackend
from branchmem.memory.schemas import MemoryFact


def _fact(value: str) -> MemoryFact:
    return MemoryFact(entity="alice", predicate="location", value=value, branch_id="b1", timestamp=1.0)


# --- embedding detector -----------------------------------------------------

_TOY_EMBEDDINGS = {
    "boston": np.array([1.0, 0.0]),
    "chicago": np.array([0.0, 1.0]),  # orthogonal to boston -> low similarity -> conflict
    "beantown": np.array([0.9, 0.1]),  # near-paraphrase of boston -> high similarity -> not a conflict
}


def _toy_embed_fn(texts: list[str]) -> np.ndarray:
    return np.array([_TOY_EMBEDDINGS[t.lower()] for t in texts])


def test_embedding_detector_flags_dissimilar_values_as_conflict():
    detector = EmbeddingConflictDetector(threshold=0.55, embed_fn=_toy_embed_fn)
    judgment = detector.detect(_fact("boston"), _fact("chicago"))
    assert judgment.is_conflict is True
    assert judgment.score < 0.55


def test_embedding_detector_does_not_flag_paraphrase_as_conflict():
    detector = EmbeddingConflictDetector(threshold=0.55, embed_fn=_toy_embed_fn)
    judgment = detector.detect(_fact("boston"), _fact("beantown"))
    assert judgment.is_conflict is False
    assert judgment.score >= 0.55


# --- NLI detector ------------------------------------------------------------


def _toy_predict_fn(pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
    out = []
    for premise, hypothesis in pairs:
        if "boston" in premise and "chicago" in hypothesis:
            out.append({"contradiction": 0.9, "entailment": 0.05, "neutral": 0.05})
        else:
            out.append({"contradiction": 0.1, "entailment": 0.8, "neutral": 0.1})
    return out


def test_nli_detector_flags_contradiction():
    detector = NLIConflictDetector(contradiction_threshold=0.5, predict_fn=_toy_predict_fn)
    judgment = detector.detect(_fact("boston"), _fact("chicago"))
    assert judgment.is_conflict is True
    assert judgment.score == 0.9


def test_nli_detector_does_not_flag_entailment():
    detector = NLIConflictDetector(contradiction_threshold=0.5, predict_fn=_toy_predict_fn)
    judgment = detector.detect(_fact("boston"), _fact("boston"))
    assert judgment.is_conflict is False


# --- LLM judge detector --------------------------------------------------------


def test_llm_judge_detector_parses_conflict_json():
    backend = MockBackend(
        canned_responses={
            "chicago": '{"is_conflict": true, "reasoning": "different cities"}',
        }
    )
    detector = LLMJudgeConflictDetector(backend=backend)
    judgment = detector.detect(_fact("boston"), _fact("chicago"))
    assert judgment.is_conflict is True
    assert "different cities" in judgment.detail


def test_llm_judge_detector_parses_no_conflict_json():
    backend = MockBackend(
        canned_responses={
            "beantown": '{"is_conflict": false, "reasoning": "same city, nickname"}',
        }
    )
    detector = LLMJudgeConflictDetector(backend=backend)
    judgment = detector.detect(_fact("boston"), _fact("beantown"))
    assert judgment.is_conflict is False


def test_llm_judge_detector_handles_unparseable_response_gracefully():
    backend = MockBackend(canned_responses={"boston": "I cannot determine this."})
    detector = LLMJudgeConflictDetector(backend=backend)
    judgment = detector.detect(_fact("boston"), _fact("chicago"))
    assert judgment.is_conflict is False
    assert "unparsed" in judgment.detail
