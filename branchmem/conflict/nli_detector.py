"""Lightweight local conflict detector: a small NLI cross-encoder's
contradiction probability between the two facts, phrased as short sentences.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from branchmem.conflict.base import ConflictDetector, ConflictJudgment
from branchmem.memory.schemas import MemoryFact


class NLIConflictDetector(ConflictDetector):
    name = "nli"

    def __init__(
        self,
        contradiction_threshold: float = 0.5,
        predict_fn: Optional[Callable[[list[tuple[str, str]]], list[dict[str, float]]]] = None,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
    ) -> None:
        self.contradiction_threshold = contradiction_threshold
        self._predict_fn = predict_fn
        self._model_name = model_name
        self._model = None  # lazy-loaded only if no predict_fn is injected

    def _predict(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        if self._predict_fn is not None:
            return self._predict_fn(pairs)
        if self._model is None:
            from sentence_transformers import CrossEncoder  # local import: optional dep

            self._model = CrossEncoder(self._model_name)
        raw_scores = self._model.predict([list(p) for p in pairs])
        raw_scores = np.asarray(raw_scores)
        exp = np.exp(raw_scores - raw_scores.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        # cross-encoder/nli-* label order: [contradiction, entailment, neutral]
        return [{"contradiction": float(p[0]), "entailment": float(p[1]), "neutral": float(p[2])} for p in probs]

    def detect(self, fact_a: MemoryFact, fact_b: MemoryFact) -> ConflictJudgment:
        t0 = time.time()
        premise = f"{fact_a.entity}'s {fact_a.predicate} is {fact_a.value}."
        hypothesis = f"{fact_b.entity}'s {fact_b.predicate} is {fact_b.value}."
        result = self._predict([(premise, hypothesis)])[0]
        contradiction_score = result["contradiction"]
        is_conflict = contradiction_score >= self.contradiction_threshold
        return ConflictJudgment(
            is_conflict=is_conflict,
            score=contradiction_score,
            latency_s=time.time() - t0,
            detail=f"contradiction_prob={contradiction_score:.3f}",
        )
