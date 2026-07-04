"""Lightweight local conflict detector: embedding-similarity threshold on the
*value* strings of two facts sharing an (entity, predicate) key.

Rationale: since candidate pairs already share entity+predicate, low semantic
similarity between the values means they're saying genuinely different things
(a conflict); high similarity means they're likely paraphrases of the same
value (not a conflict). The threshold must be calibrated on a held-out pilot
set — see scripts/run_pilot.py and ANALYSIS_PLAN.md for the calibration
procedure. Do not hand-pick this threshold from the final evaluation data.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from branchmem.conflict.base import ConflictDetector, ConflictJudgment
from branchmem.memory.schemas import MemoryFact


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class EmbeddingConflictDetector(ConflictDetector):
    name = "embedding_threshold"

    def __init__(
        self,
        threshold: float = 0.55,
        embed_fn: Optional[Callable[[list[str]], np.ndarray]] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.threshold = threshold
        self._embed_fn = embed_fn
        self._model_name = model_name
        self._model = None  # lazy-loaded only if no embed_fn is injected

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._embed_fn is not None:
            return np.asarray(self._embed_fn(texts))
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # local import: optional dep

            self._model = SentenceTransformer(self._model_name)
        return np.asarray(self._model.encode(texts))

    def detect(self, fact_a: MemoryFact, fact_b: MemoryFact) -> ConflictJudgment:
        t0 = time.time()
        embeddings = self._embed([fact_a.value, fact_b.value])
        sim = _cosine(embeddings[0], embeddings[1])
        is_conflict = sim < self.threshold
        return ConflictJudgment(
            is_conflict=is_conflict,
            score=sim,
            latency_s=time.time() - t0,
            detail=f"cosine_sim={sim:.3f} threshold={self.threshold}",
        )
