"""Common interface for conflict detectors (RQ2): given two facts sharing an
(entity, predicate) key, judge whether they genuinely conflict."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from branchmem.memory.schemas import MemoryFact


@dataclass
class ConflictJudgment:
    is_conflict: bool
    score: float  # detector's raw confidence/similarity — kept for threshold calibration
    latency_s: float = 0.0
    detail: str = ""


class ConflictDetector(ABC):
    name: str = "base"

    @abstractmethod
    def detect(self, fact_a: MemoryFact, fact_b: MemoryFact) -> ConflictJudgment:
        """fact_a and fact_b must share the same (entity, predicate) key."""
