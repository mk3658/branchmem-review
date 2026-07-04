"""Common interface and shared helpers for the four merge strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from branchmem.memory.schemas import MemoryBranch, MemoryFact, MergeResult


class MergeStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def merge(self, ancestor: MemoryBranch, branch_a: MemoryBranch, branch_b: MemoryBranch) -> MergeResult:
        """Reconcile branch_a and branch_b, which both forked from `ancestor`."""


def current_facts_by_key(branch: MemoryBranch) -> dict[tuple[str, str], MemoryFact]:
    """The latest (by timestamp) fact per (entity, predicate) key on a branch."""
    current: dict[tuple[str, str], MemoryFact] = {}
    for fact in branch.facts:
        key = fact.key()
        if key not in current or fact.timestamp > current[key].timestamp:
            current[key] = fact
    return current


def find_collisions(
    branch_a: MemoryBranch, branch_b: MemoryBranch
) -> dict[tuple[str, str], tuple[MemoryFact, MemoryFact]]:
    """Keys present on both branches with a differing current value — candidate conflicts.

    Iterates keys in sorted order rather than raw set order: Python randomizes
    string hash seeds per process by default, so `set` iteration order (and
    thus dict insertion order, and thus any JSON built from this dict) is NOT
    stable across separate process runs even for identical input. Merge
    strategies that build an LLM prompt from this dict (e.g. ThreeWayLLMMerge)
    would otherwise produce a different prompt string — and therefore a
    different content-hash cache key — for the semantically identical
    scenario on every fresh interpreter invocation, silently defeating the
    cache and re-spending on calls that should have been free reuses.
    """
    a_facts = current_facts_by_key(branch_a)
    b_facts = current_facts_by_key(branch_b)
    collisions = {}
    for key in sorted(set(a_facts) & set(b_facts)):
        if a_facts[key].value != b_facts[key].value:
            collisions[key] = (a_facts[key], b_facts[key])
    return collisions
