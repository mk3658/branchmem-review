"""ConfidenceRuleMerge: a deterministic, provenance-aware baseline with no LLM
call at all. On a conflict, keep whichever branch's fact has higher
`confidence`; on an exact tie, flag unresolved rather than guess.

This isolates a specific question the LLM-based ThreeWayLLMMerge leaves open:
is an LLM's semantic judgment adding anything beyond simply reading the
confidence field and picking the larger number? If ConfidenceRuleMerge scores
close to ThreeWayLLMMerge, the LLM is mostly re-deriving a rule a single
`if`-statement could express; if it scores meaningfully worse, the LLM is
doing something the deterministic rule cannot (e.g. using ancestor context,
or reasoning about cases confidence alone doesn't resolve).
"""

from __future__ import annotations

from branchmem.memory.schemas import MemoryBranch, MergeResult, Resolution, ResolvedFact
from branchmem.merge.base import MergeStrategy, current_facts_by_key, find_collisions


class ConfidenceRuleMerge(MergeStrategy):
    name = "confidence_rule"

    def merge(self, ancestor: MemoryBranch, branch_a: MemoryBranch, branch_b: MemoryBranch) -> MergeResult:
        a_facts = current_facts_by_key(branch_a)
        b_facts = current_facts_by_key(branch_b)
        collisions = find_collisions(branch_a, branch_b)

        resolved: list[ResolvedFact] = []

        for key in sorted((set(a_facts) | set(b_facts)) - set(collisions)):
            fact_a, fact_b = a_facts.get(key), b_facts.get(key)
            if fact_a is not None and fact_b is not None:
                resolved.append(
                    ResolvedFact(
                        fact=fact_a, resolution=Resolution.KEPT,
                        source_branch_ids=[branch_a.branch_id, branch_b.branch_id],
                        justification="both branches agree",
                    )
                )
            else:
                fact = fact_a or fact_b
                resolved.append(
                    ResolvedFact(fact=fact, resolution=Resolution.KEPT, source_branch_ids=[fact.branch_id])
                )

        for key, (fact_a, fact_b) in collisions.items():
            if fact_a.confidence > fact_b.confidence:
                resolved.append(
                    ResolvedFact(
                        fact=fact_a, resolution=Resolution.KEPT, source_branch_ids=[branch_a.branch_id],
                        justification=f"higher confidence ({fact_a.confidence} > {fact_b.confidence})",
                    )
                )
            elif fact_b.confidence > fact_a.confidence:
                resolved.append(
                    ResolvedFact(
                        fact=fact_b, resolution=Resolution.KEPT, source_branch_ids=[branch_b.branch_id],
                        justification=f"higher confidence ({fact_b.confidence} > {fact_a.confidence})",
                    )
                )
            else:
                resolved.append(
                    ResolvedFact(
                        fact=fact_a, resolution=Resolution.FLAGGED_UNRESOLVED,
                        source_branch_ids=[branch_a.branch_id, branch_b.branch_id],
                        justification=f"tied confidence ({fact_a.confidence})",
                    )
                )

        return MergeResult(strategy_name=self.name, resulting_facts=resolved)
