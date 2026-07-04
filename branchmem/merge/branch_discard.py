"""BranchDiscard: the post-hoc analogue of S-Bus's OCC-abort. If ANY key
collision is detected between the two branches, discard one entire branch's
post-fork updates (including its orthogonal, non-conflicting ones) and keep
only the other branch's current facts. If no collision exists, nothing needs
discarding — both branches' (non-conflicting) updates are kept.

Two discard policies are both reportable, since the choice matters (H2):
  - "always_b": always discard branch B.
  - "fewer_updates": discard whichever branch made fewer post-fork updates.
"""

from __future__ import annotations

from typing import Literal

from branchmem.memory.schemas import MemoryBranch, MergeResult, Resolution, ResolvedFact
from branchmem.merge.base import MergeStrategy, current_facts_by_key, find_collisions

DiscardPolicy = Literal["always_b", "fewer_updates"]


class BranchDiscard(MergeStrategy):
    def __init__(self, policy: DiscardPolicy = "always_b") -> None:
        self.policy = policy
        self.name = f"branch_discard_{policy}"

    def _pick_discard_branch(self, branch_a: MemoryBranch, branch_b: MemoryBranch) -> MemoryBranch:
        if self.policy == "always_b":
            return branch_b
        # "fewer_updates": discard whichever branch made fewer post-fork updates
        # (ties broken by discarding B, for determinism).
        n_a = len(current_facts_by_key(branch_a))
        n_b = len(current_facts_by_key(branch_b))
        return branch_b if n_b <= n_a else branch_a

    def merge(self, ancestor: MemoryBranch, branch_a: MemoryBranch, branch_b: MemoryBranch) -> MergeResult:
        collisions = find_collisions(branch_a, branch_b)
        resolved: list[ResolvedFact] = []

        if not collisions:
            a_facts = current_facts_by_key(branch_a)
            b_facts = current_facts_by_key(branch_b)
            for key in sorted(set(a_facts) | set(b_facts)):
                fact = a_facts.get(key) or b_facts.get(key)
                resolved.append(
                    ResolvedFact(fact=fact, resolution=Resolution.KEPT, source_branch_ids=[fact.branch_id])
                )
            return MergeResult(strategy_name=self.name, resulting_facts=resolved)

        discarded = self._pick_discard_branch(branch_a, branch_b)
        kept_branch = branch_b if discarded is branch_a else branch_a
        kept_facts = current_facts_by_key(kept_branch)

        for fact in kept_facts.values():
            resolved.append(
                ResolvedFact(
                    fact=fact,
                    resolution=Resolution.KEPT,
                    source_branch_ids=[kept_branch.branch_id],
                    justification=f"branch {discarded.branch_id} discarded on conflict detection",
                )
            )
        # Only report genuine information loss: facts unique to (or differing
        # in) the discarded branch. A fact identical to the kept branch's
        # value (e.g. untouched since the ancestor) isn't actually lost.
        for key, fact in current_facts_by_key(discarded).items():
            kept_fact = kept_facts.get(key)
            if kept_fact is not None and kept_fact.value == fact.value:
                continue
            resolved.append(
                ResolvedFact(
                    fact=fact,
                    resolution=Resolution.DROPPED,
                    source_branch_ids=[discarded.branch_id],
                    justification=f"entire branch {discarded.branch_id} discarded on conflict detection ({self.policy})",
                )
            )
        return MergeResult(strategy_name=self.name, resulting_facts=resolved)
