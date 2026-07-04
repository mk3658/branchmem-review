"""Core data structures for BranchMem: facts, branches, conflicts, and merge results."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

FactSource = Literal["user", "observation", "inference"]


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class MemoryFact(BaseModel):
    """A single (entity, predicate, value) belief held by an agent on a branch."""

    fact_id: str = Field(default_factory=lambda: _new_id("fact"))
    entity: str
    predicate: str
    value: str
    branch_id: str
    timestamp: float
    source: FactSource = "observation"
    confidence: float = 1.0
    provenance: str = ""
    common_ancestor_id: Optional[str] = None

    def key(self) -> tuple[str, str]:
        """(entity, predicate) identity used to find corresponding facts across branches."""
        return (self.entity, self.predicate)


class MemoryBranch(BaseModel):
    """A branch of memory: either the root (common ancestor) or a fork of another branch."""

    branch_id: str = Field(default_factory=lambda: _new_id("branch"))
    parent_branch_id: Optional[str] = None
    fork_point_timestamp: Optional[float] = None
    facts: list[MemoryFact] = Field(default_factory=list)


class ConflictPair(BaseModel):
    """A candidate conflict between a fact on branch A and a fact on branch B."""

    fact_a_id: str
    fact_b_id: str
    is_conflict: bool  # ground truth, set by the scenario generator
    conflict_type: Optional[Literal["resolvable", "ambiguous"]] = None
    ground_truth_value: Optional[str] = None
    detector_predictions: dict[str, Any] = Field(default_factory=dict)


class Resolution(str, Enum):
    KEPT = "kept"
    DROPPED = "dropped"
    MERGED = "merged"
    FLAGGED_UNRESOLVED = "flagged_unresolved"


class ResolvedFact(BaseModel):
    """A fact in a MergeResult, tagged with how the merge strategy handled it."""

    fact: MemoryFact
    resolution: Resolution
    source_branch_ids: list[str] = Field(default_factory=list)
    justification: str = ""


class MergeResult(BaseModel):
    """Output of applying a merge strategy to two diverged branches."""

    strategy_name: str
    resulting_facts: list[ResolvedFact] = Field(default_factory=list)
    unresolved_conflicts: list[ConflictPair] = Field(default_factory=list)

    def kept_facts(self) -> list[MemoryFact]:
        return [rf.fact for rf in self.resulting_facts if rf.resolution != Resolution.DROPPED]
