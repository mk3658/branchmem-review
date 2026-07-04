"""RawTextLLMMerge: the minimal-metadata LLM ablation, added as a later
post-hoc check. TwoWayLLMMerge already withholds the common ancestor;
RawTextLLMMerge goes further and withholds source, confidence, and
timestamps too — the LLM sees only the entity, predicate, and each branch's
raw conflicting value, nothing else.

Purpose: TwoWayLLMMerge and ThreeWayLLMMerge both still hand the model a
structured `confidence` field that a single deterministic rule
(ConfidenceRuleMerge) can read directly, which is exactly what
Table~\\ref{tab:ablations} shows happening — a bare `if` statement matches or
beats the LLM. RawTextLLMMerge closes the one remaining gap in the ablation
ladder: with no structured signal available *at all*, does the LLM do
anything better than chance on `resolvable` conflicts (where a real, if
unobservable-to-us, correct answer exists), or does it default to flagging
everything unresolved (correct behavior on `ambiguous` conflicts, but wrong
on `resolvable` ones under this benchmark's scoring rule)? Either outcome is
informative and is reported as-is (see `results/final/raw_text_ablation.json`
and Section 5.2 of the paper) — this file does not encode an expected
direction.

Everything else (non-conflicting keys resolved in code, batched single call
per merge, flag-rather-than-guess instruction) is identical to
ThreeWayLLMMerge/TwoWayLLMMerge; only the prompt's content differs.
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
    "common ancestor's value for these keys, and you are NOT given any source, "
    "confidence, or timestamp metadata — only the bare (entity, predicate) key "
    "and each branch's raw current value. Decide, from the content of the two "
    "values alone, whether one is more likely correct than the other (e.g. one "
    "reads as a specific correction of the other, or as more precise/plausible "
    "phrasing of the same fact), whether they can be sensibly merged into one "
    "value, or whether nothing about the two raw values themselves gives you any "
    "real basis to prefer one over the other. If there is no such basis, "
    "explicitly flag it as unresolvable rather than guessing — do not invent a "
    "preference where the text gives you none."
)

_PROMPT_TEMPLATE = """Branch A's current value for each conflicting key:
{branch_a_json}

Branch B's current value for each conflicting key:
{branch_b_json}

For EACH key listed above, decide how to reconcile it. Respond with strict JSON only,
no other text, in this exact shape:

{{"resolutions": [
  {{"entity": "...", "predicate": "...", "resolution": "kept_from_a" | "kept_from_b" | "merged" | "flagged_unresolved",
    "value": "<the reconciled value, or null if flagged_unresolved>", "justification": "<one sentence>"}}
]}}
"""


class RawTextLLMMerge(MergeStrategy):
    name = "raw_text_llm"

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

        # Deliberately just the value — no source, confidence, or timestamp.
        branch_a_json = json.dumps({f"{k[0]}|{k[1]}": v[0].value for k, v in collisions.items()}, indent=2)
        branch_b_json = json.dumps({f"{k[0]}|{k[1]}": v[1].value for k, v in collisions.items()}, indent=2)
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
