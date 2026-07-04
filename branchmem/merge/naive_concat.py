"""NaiveConcat baseline: keep every current fact from both branches, no
deduplication, no conflict resolution. If contradictions confuse the
downstream agent, that's the expected failure mode this baseline exposes."""

from __future__ import annotations

from branchmem.memory.schemas import MemoryBranch, MergeResult, Resolution, ResolvedFact
from branchmem.merge.base import MergeStrategy, current_facts_by_key


class NaiveConcat(MergeStrategy):
    name = "naive_concat"

    def merge(self, ancestor: MemoryBranch, branch_a: MemoryBranch, branch_b: MemoryBranch) -> MergeResult:
        resolved: list[ResolvedFact] = []
        for branch in (branch_a, branch_b):
            for fact in current_facts_by_key(branch).values():
                resolved.append(
                    ResolvedFact(fact=fact, resolution=Resolution.KEPT, source_branch_ids=[branch.branch_id])
                )
        return MergeResult(strategy_name=self.name, resulting_facts=resolved)
