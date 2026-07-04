"""QA-style downstream tasks: for each scenario, generate questions answerable
only if the memory was reconciled correctly. Answers are scored against the
scenario generator's ground truth (see evaluation/metrics.py), never against a
reference LLM's opinion, to avoid circularity between merger and grader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from branchmem.benchmark.scenario_generator import Scenario
from branchmem.merge.base import current_facts_by_key


@dataclass
class DownstreamQuestion:
    scenario_id: str
    entity: str
    predicate: str
    question: str
    expected_answer: Optional[str]  # None => no single correct answer; correct behavior is to flag/abstain
    category: str  # "orthogonal" | "resolvable" | "ambiguous"


def generate_downstream_questions(scenario: Scenario) -> list[DownstreamQuestion]:
    questions: list[DownstreamQuestion] = []
    fact_lookup = {f.fact_id: f for f in scenario.branch_a.facts + scenario.branch_b.facts}

    conflict_keys: set[tuple[str, str]] = set()
    for pair in scenario.conflict_pairs:
        fact_a = fact_lookup[pair.fact_a_id]
        conflict_keys.add(fact_a.key())
        category = pair.conflict_type or "resolvable"
        questions.append(
            DownstreamQuestion(
                scenario_id=scenario.scenario_id,
                entity=fact_a.entity,
                predicate=fact_a.predicate,
                question=f"What is {fact_a.entity}'s current {fact_a.predicate.replace('_', ' ')}?",
                expected_answer=pair.ground_truth_value,
                category=category,
            )
        )

    # Orthogonal single-branch additions: unambiguous ground truth, tests
    # whether the merge strategy preserved non-conflicting info from BOTH sides.
    ancestor_keys = {f.key() for f in scenario.ancestor.facts}
    for branch in (scenario.branch_a, scenario.branch_b):
        for key, fact in current_facts_by_key(branch).items():
            if key in conflict_keys or key in ancestor_keys:
                continue
            questions.append(
                DownstreamQuestion(
                    scenario_id=scenario.scenario_id,
                    entity=fact.entity,
                    predicate=fact.predicate,
                    question=f"What is {fact.entity}'s current {fact.predicate.replace('_', ' ')}?",
                    expected_answer=fact.value,
                    category="orthogonal",
                )
            )
    return questions
