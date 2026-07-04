"""LLM-as-judge conflict detector: the reference standard for RQ2, not a
fourth candidate competing to be "better than" the cheap detectors. Embedding
and NLI detectors are evaluated against this baseline's precision/recall gap.
"""

from __future__ import annotations

import json
import time

from branchmem.conflict.base import ConflictDetector, ConflictJudgment
from branchmem.llm.base import LLMBackend
from branchmem.memory.schemas import MemoryFact

_PROMPT_TEMPLATE = """You are judging whether two memory facts about the same entity and \
predicate genuinely conflict (state incompatible values) or not (e.g. one is a \
paraphrase, refinement, or otherwise-compatible elaboration of the other).

Fact A: {entity_a} | {predicate_a} = "{value_a}"
Fact B: {entity_b} | {predicate_b} = "{value_b}"

Respond with strict JSON only, no other text: {{"is_conflict": true or false, "reasoning": "<one sentence>"}}
"""


class LLMJudgeConflictDetector(ConflictDetector):
    name = "llm_judge"

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    def detect(self, fact_a: MemoryFact, fact_b: MemoryFact, use_cache: bool = True) -> ConflictJudgment:
        t0 = time.time()
        prompt = _PROMPT_TEMPLATE.format(
            entity_a=fact_a.entity,
            predicate_a=fact_a.predicate,
            value_a=fact_a.value,
            entity_b=fact_b.entity,
            predicate_b=fact_b.predicate,
            value_b=fact_b.value,
        )
        response = self.backend.complete(prompt, use_cache=use_cache)
        is_conflict, reasoning = _parse_response(response.text)
        return ConflictJudgment(
            is_conflict=is_conflict,
            score=1.0 if is_conflict else 0.0,
            latency_s=time.time() - t0,
            detail=reasoning,
        )


def _parse_response(text: str) -> tuple[bool, str]:
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            parsed = json.loads(text[start : end + 1])
            return bool(parsed["is_conflict"]), str(parsed.get("reasoning", ""))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return False, f"[unparsed LLM response, defaulted to no-conflict] {text[:200]}"
