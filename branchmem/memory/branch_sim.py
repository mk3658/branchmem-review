"""Fork a common-ancestor memory state into two branches and apply independent,
scripted update streams, recording ground-truth conflict/resolution data by
construction (never inferred after the fact).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from branchmem.memory.schemas import ConflictPair, MemoryBranch, MemoryFact

DivergenceType = Literal["orthogonal", "resolvable_conflict", "ambiguous_conflict"]


class UpdateSpec(BaseModel):
    """One scripted update applied to a single branch after the fork point.

    `divergence_type` and `ground_truth_value` are supplied by the scenario
    author (by construction) and describe how this update relates to whatever
    the *other* branch does to the same (entity, predicate) key:
      - "orthogonal": nothing on the other branch touches this key, or it does
        so in a non-contradictory way. No ground truth needed.
      - "resolvable_conflict": the other branch also updates this key with a
        contradicting value, and there IS a single correct reconciled value
        (ground_truth_value must be set).
      - "ambiguous_conflict": the other branch also updates this key with a
        contradicting value, and there is NO single correct answer — the
        correct merge behavior is to flag it, not guess (ground_truth_value
        must be None).
    """

    entity: str
    predicate: str
    value: str
    timestamp: float
    source: Literal["user", "observation", "inference"] = "observation"
    confidence: float = 1.0
    provenance: str = ""
    divergence_type: DivergenceType = "orthogonal"
    ground_truth_value: Optional[str] = None

    def key(self) -> tuple[str, str]:
        return (self.entity, self.predicate)


class DivergenceError(ValueError):
    """Raised when the two branches' update streams declare inconsistent ground truth
    for the same (entity, predicate) key — a scenario-authoring bug."""


class BranchSimulator:
    def create_common_ancestor(
        self, facts: list[tuple[str, str, str]], timestamp: float, branch_id: str = "root"
    ) -> MemoryBranch:
        """Build the root/common-ancestor branch from (entity, predicate, value) triples."""
        branch = MemoryBranch(branch_id=branch_id, parent_branch_id=None)
        for entity, predicate, value in facts:
            branch.facts.append(
                MemoryFact(
                    entity=entity,
                    predicate=predicate,
                    value=value,
                    branch_id=branch.branch_id,
                    timestamp=timestamp,
                    source="observation",
                )
            )
        return branch

    def fork(
        self, ancestor: MemoryBranch, fork_point_timestamp: float
    ) -> tuple[MemoryBranch, MemoryBranch]:
        """Fork `ancestor` into two independent child branches, each starting as a
        copy of the ancestor's current facts."""
        branch_a = MemoryBranch(
            parent_branch_id=ancestor.branch_id, fork_point_timestamp=fork_point_timestamp
        )
        branch_b = MemoryBranch(
            parent_branch_id=ancestor.branch_id, fork_point_timestamp=fork_point_timestamp
        )
        for child in (branch_a, branch_b):
            for fact in ancestor.facts:
                child.facts.append(
                    MemoryFact(
                        entity=fact.entity,
                        predicate=fact.predicate,
                        value=fact.value,
                        branch_id=child.branch_id,
                        timestamp=fact.timestamp,
                        source=fact.source,
                        confidence=fact.confidence,
                        provenance=fact.provenance,
                        common_ancestor_id=fact.fact_id,
                    )
                )
        return branch_a, branch_b

    def _current_fact(self, branch: MemoryBranch, key: tuple[str, str]) -> Optional[MemoryFact]:
        matches = [f for f in branch.facts if f.key() == key]
        return max(matches, key=lambda f: f.timestamp) if matches else None

    def apply_updates(self, branch: MemoryBranch, updates: list[UpdateSpec]) -> MemoryBranch:
        """Apply a scripted update stream to a branch in place, superseding prior
        facts at the same (entity, predicate) key and keeping full history."""
        for update in sorted(updates, key=lambda u: u.timestamp):
            prior = self._current_fact(branch, update.key())
            branch.facts.append(
                MemoryFact(
                    entity=update.entity,
                    predicate=update.predicate,
                    value=update.value,
                    branch_id=branch.branch_id,
                    timestamp=update.timestamp,
                    source=update.source,
                    confidence=update.confidence,
                    provenance=update.provenance,
                    common_ancestor_id=prior.fact_id if prior else None,
                )
            )
        return branch

    def diverge(
        self,
        ancestor: MemoryBranch,
        updates_a: list[UpdateSpec],
        updates_b: list[UpdateSpec],
        fork_point_timestamp: float,
    ) -> tuple[MemoryBranch, MemoryBranch, list[ConflictPair]]:
        """Fork `ancestor` and apply each branch's independent update stream, then
        derive ground-truth ConflictPairs for every key touched by both streams.
        """
        branch_a, branch_b = self.fork(ancestor, fork_point_timestamp)
        self.apply_updates(branch_a, updates_a)
        self.apply_updates(branch_b, updates_b)

        by_key_a = {u.key(): u for u in updates_a}
        by_key_b = {u.key(): u for u in updates_b}
        shared_keys = set(by_key_a) & set(by_key_b)

        conflict_pairs: list[ConflictPair] = []
        for key in sorted(shared_keys):
            spec_a, spec_b = by_key_a[key], by_key_b[key]
            if spec_a.divergence_type != spec_b.divergence_type:
                raise DivergenceError(
                    f"Inconsistent divergence_type for key {key}: "
                    f"branch A={spec_a.divergence_type!r} vs branch B={spec_b.divergence_type!r}"
                )
            if spec_a.divergence_type == "resolvable_conflict":
                if spec_a.ground_truth_value != spec_b.ground_truth_value:
                    raise DivergenceError(f"Inconsistent ground_truth_value for key {key}")
                if spec_a.ground_truth_value is None:
                    raise DivergenceError(f"resolvable_conflict at key {key} needs ground_truth_value")
            if spec_a.divergence_type == "ambiguous_conflict":
                if spec_a.ground_truth_value is not None or spec_b.ground_truth_value is not None:
                    raise DivergenceError(f"ambiguous_conflict at key {key} must have no ground_truth_value")

            fact_a = self._current_fact(branch_a, key)
            fact_b = self._current_fact(branch_b, key)
            assert fact_a is not None and fact_b is not None
            conflict_pairs.append(
                ConflictPair(
                    fact_a_id=fact_a.fact_id,
                    fact_b_id=fact_b.fact_id,
                    is_conflict=spec_a.divergence_type != "orthogonal",
                    conflict_type=(
                        "resolvable"
                        if spec_a.divergence_type == "resolvable_conflict"
                        else "ambiguous"
                        if spec_a.divergence_type == "ambiguous_conflict"
                        else None
                    ),
                    ground_truth_value=spec_a.ground_truth_value,
                )
            )
        return branch_a, branch_b, conflict_pairs
