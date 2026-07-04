"""LastWriterWins baseline: on a key collision, keep whichever fact has the
later absolute timestamp. This has NO branch-awareness — it's the "what if
you ignore branch structure entirely and trust wall-clock time" baseline,
carried over naively from the single-timeline freshness literature (TOKI)."""

from __future__ import annotations

from branchmem.memory.schemas import MemoryBranch, MergeResult, Resolution, ResolvedFact
from branchmem.merge.base import MergeStrategy, current_facts_by_key


class LastWriterWins(MergeStrategy):
    name = "last_writer_wins"

    def merge(self, ancestor: MemoryBranch, branch_a: MemoryBranch, branch_b: MemoryBranch) -> MergeResult:
        a_facts = current_facts_by_key(branch_a)
        b_facts = current_facts_by_key(branch_b)
        all_keys = set(a_facts) | set(b_facts)

        resolved: list[ResolvedFact] = []
        for key in sorted(all_keys):
            fact_a, fact_b = a_facts.get(key), b_facts.get(key)
            if fact_a is not None and fact_b is not None:
                if fact_a.value == fact_b.value:
                    resolved.append(
                        ResolvedFact(
                            fact=fact_a,
                            resolution=Resolution.KEPT,
                            source_branch_ids=[branch_a.branch_id, branch_b.branch_id],
                            justification="both branches agree",
                        )
                    )
                else:
                    winner, loser = (
                        (fact_a, fact_b) if fact_a.timestamp >= fact_b.timestamp else (fact_b, fact_a)
                    )
                    resolved.append(
                        ResolvedFact(
                            fact=winner,
                            resolution=Resolution.KEPT,
                            source_branch_ids=[winner.branch_id],
                            justification=f"later timestamp ({winner.timestamp}) beats {loser.timestamp}",
                        )
                    )
                    resolved.append(
                        ResolvedFact(
                            fact=loser,
                            resolution=Resolution.DROPPED,
                            source_branch_ids=[loser.branch_id],
                            justification=f"earlier timestamp ({loser.timestamp}), overwritten by LWW",
                        )
                    )
            elif fact_a is not None:
                resolved.append(
                    ResolvedFact(fact=fact_a, resolution=Resolution.KEPT, source_branch_ids=[branch_a.branch_id])
                )
            else:
                assert fact_b is not None
                resolved.append(
                    ResolvedFact(fact=fact_b, resolution=Resolution.KEPT, source_branch_ids=[branch_b.branch_id])
                )
        return MergeResult(strategy_name=self.name, resulting_facts=resolved)
