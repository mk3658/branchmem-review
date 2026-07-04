"""TwoWayLLMMerge: an ablation of ThreeWayLLMMerge that withholds the common
ancestor from the prompt — the LLM sees only each branch's current value,
source, and confidence, not what the key looked like before either branch
diverged. Isolates whether ancestor context is doing real work, or whether
source/confidence alone is enough for the LLM to reconcile conflicts.

Everything else (non-conflicting keys resolved in code, batched single call
per merge, flag-rather-than-guess instruction) is identical to
ThreeWayLLMMerge; only the prompt's ancestor section is removed.
"""

from __future__ import annotations

import json

from branchmem.llm.base import LLMBackend
from branchmem.memory.schemas import ConflictPair, MemoryBranch, MergeResult, Resolution, ResolvedFact
from branchmem.merge.base import MergeStrategy, current_facts_by_key, find_collisions
from branchmem.utils.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are reconciling two independently-updated branches of an agent's memory "
    "that diverged while disconnected from each other. You are NOT given the "
    "common ancestor's value for these keys — only each branch's current value, "
    "source, and confidence. 'source' is how that update was recorded: \"user\" "
    "means the user explicitly stated or corrected it; \"inference\" means the "
    "agent inferred or guessed it; \"observation\" means the agent observed it "
    "indirectly. Resolve each conflict using whatever signal actually "
    "distinguishes the two updates — e.g. a user-sourced, higher-confidence "
    "update is ordinarily more trustworthy than an inferred, lower-confidence "
    "one. Do NOT use timestamps to decide, even if you can infer relative "
    "recency — two branches that were disconnected from each other have no "
    "shared clock, so 'later' does not mean 'more correct' here. If source and "
    "confidence give no real basis to prefer one branch over the other, "
    "explicitly flag it as unresolvable rather than guessing."
)

_PROMPT_TEMPLATE = """Branch A's current value/source/confidence for each conflicting key:
{branch_a_json}

Branch B's current value/source/confidence for each conflicting key:
{branch_b_json}

For EACH key listed above, decide how to reconcile it. Respond with strict JSON only,
no other text, in this exact shape:

{{"resolutions": [
  {{"entity": "...", "predicate": "...", "resolution": "kept_from_a" | "kept_from_b" | "merged" | "flagged_unresolved",
    "value": "<the reconciled value, or null if flagged_unresolved>", "justification": "<one sentence>"}}
]}}
"""


class TwoWayLLMMerge(MergeStrategy):
    name = "two_way_llm"

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    def merge(
        self, ancestor: MemoryBranch, branch_a: MemoryBranch, branch_b: MemoryBranch, use_cache: bool = True
    ) -> MergeResult:
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

        if not collisions:
            return MergeResult(strategy_name=self.name, resulting_facts=resolved)

        def _fact_json(fact) -> dict:
            return {"value": fact.value, "source": fact.source, "confidence": fact.confidence}

        branch_a_json = json.dumps({f"{k[0]}|{k[1]}": _fact_json(v[0]) for k, v in collisions.items()}, indent=2)
        branch_b_json = json.dumps({f"{k[0]}|{k[1]}": _fact_json(v[1]) for k, v in collisions.items()}, indent=2)
        prompt = _PROMPT_TEMPLATE.format(branch_a_json=branch_a_json, branch_b_json=branch_b_json)
        response = self.backend.complete(prompt, system=_SYSTEM_PROMPT, use_cache=use_cache)
        resolutions_by_key = _parse_response(response.text)

        unresolved = []
        for key, (fact_a, fact_b) in collisions.items():
            key_str = f"{key[0]}|{key[1]}"
            decision = resolutions_by_key.get(key_str)
            if decision is None:
                logger.warning("no LLM resolution for key %s; flagging unresolved", key_str)
                decision = {"resolution": "flagged_unresolved", "value": None, "justification": "unparsed LLM response"}

            res_type = decision.get("resolution", "flagged_unresolved")
            justification = decision.get("justification", "")
            if res_type == "flagged_unresolved" or not decision.get("value"):
                resolved.append(
                    ResolvedFact(
                        fact=fact_a, resolution=Resolution.FLAGGED_UNRESOLVED,
                        source_branch_ids=[branch_a.branch_id, branch_b.branch_id],
                        justification=justification or "flagged unresolvable by LLM",
                    )
                )
                unresolved.append(ConflictPair(fact_a_id=fact_a.fact_id, fact_b_id=fact_b.fact_id, is_conflict=True))
                continue

            value = decision["value"]
            resolution = (
                Resolution.KEPT
                if (res_type == "kept_from_a" and value == fact_a.value)
                or (res_type == "kept_from_b" and value == fact_b.value)
                else Resolution.MERGED
            )
            source_fact = fact_a if res_type == "kept_from_a" else fact_b
            merged_fact = source_fact.model_copy(update={"value": value})
            resolved.append(
                ResolvedFact(
                    fact=merged_fact, resolution=resolution,
                    source_branch_ids=[branch_a.branch_id, branch_b.branch_id],
                    justification=justification,
                )
            )

        return MergeResult(strategy_name=self.name, resulting_facts=resolved, unresolved_conflicts=unresolved)


def _parse_response(text: str) -> dict[str, dict]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    out = {}
    for item in parsed.get("resolutions", []):
        try:
            out[f"{item['entity']}|{item['predicate']}"] = item
        except (KeyError, TypeError):
            continue
    return out
