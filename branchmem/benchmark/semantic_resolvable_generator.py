"""Generator for the `semantic_resolvable` conflict category (post-hoc,
non-preregistered; see ANALYSIS_PLAN_ADDENDUM.md item A2).

Every existing "resolvable" conflict in the locked generator ties ground
truth to a source/confidence signal a deterministic rule
(`ConfidenceRuleMerge`) can read directly. This category equalizes source
and confidence on both branches and randomizes timestamp independently of
correctness, so ground truth is recoverable only by checking which branch's
raw-text value is semantically compatible with the ancestor's stated
constraint (e.g. ancestor "is vegetarian" + branch value "wants tofu" is
compatible; "wants chicken" is not). This isolates whether an LLM merge
strategy's apparent competence is genuine semantic reasoning about content,
or just reading a structured confidence field.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from branchmem.benchmark.scenario_generator import Scenario
from branchmem.memory.branch_sim import BranchSimulator, UpdateSpec

_DEFAULT_TEMPLATES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "semantic_resolvable_templates.json"
)


@dataclass
class SemanticResolvableConfig:
    entity: str = "user"
    equal_confidence: float = 0.7  # identical on both branches -- no distinguishing signal
    equal_source: str = "observation"  # identical on both branches
    divergence_span: float = 10.0


def load_templates(path: Path | str = _DEFAULT_TEMPLATES_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)["templates"]


def generate_semantic_resolvable_scenarios(
    n_scenarios: int, seed: int, config: SemanticResolvableConfig | None = None,
    templates_path: Path | str = _DEFAULT_TEMPLATES_PATH,
) -> list[Scenario]:
    """One semantic_resolvable conflict per scenario (plus the ancestor fact
    stating the constraint). Templates are cycled through in shuffled order
    with each scenario using a fresh, seeded RNG, so re-running with the
    same seed reproduces byte-identical scenarios.
    """
    config = config or SemanticResolvableConfig()
    templates = load_templates(templates_path)
    sim = BranchSimulator()
    scenarios = []
    for i in range(n_scenarios):
        rng = random.Random(f"semres:{seed}:{i}")
        template = templates[i % len(templates)]
        # Randomize which physical branch (A or B) carries the correct value,
        # so a strategy can't learn a positional shortcut ("B is always
        # right") instead of reasoning about content.
        swap = rng.random() < 0.5
        value_a_text = template["branch_b_value"] if swap else template["branch_a_value"]
        value_b_text = template["branch_a_value"] if swap else template["branch_b_value"]
        # template's "correct_branch" is always "b" by authoring convention
        # (see data file); after the swap, track which physical side that maps to.
        correct_is_a = swap  # if swapped, the original "b" (correct) text is now on branch A

        ancestor = sim.create_common_ancestor(
            facts=[(config.entity, template["ancestor_predicate"], template["ancestor_value"])],
            timestamp=0.0,
        )

        ground_truth = value_a_text if correct_is_a else value_b_text

        # Timestamps drawn independently of correctness -- fully uninformative,
        # unlike the locked generator's tunable
        # timestamp_correlates_with_correctness (fixed at "uncorrelated" here
        # by construction, not by a probability parameter).
        ts_a = config.divergence_span * rng.uniform(0.1, 1.0)
        ts_b = config.divergence_span * rng.uniform(0.1, 1.0)

        updates_a = [
            UpdateSpec(
                entity=config.entity, predicate=template["predicate"], value=value_a_text,
                timestamp=1.0 + ts_a, source=config.equal_source, confidence=config.equal_confidence,
                divergence_type="resolvable_conflict", ground_truth_value=ground_truth,
            )
        ]
        updates_b = [
            UpdateSpec(
                entity=config.entity, predicate=template["predicate"], value=value_b_text,
                timestamp=1.0 + ts_b, source=config.equal_source, confidence=config.equal_confidence,
                divergence_type="resolvable_conflict", ground_truth_value=ground_truth,
            )
        ]

        branch_a, branch_b, conflict_pairs = sim.diverge(
            ancestor, updates_a, updates_b, fork_point_timestamp=1.0
        )
        scenarios.append(
            Scenario(
                scenario_id=f"semres_{seed}_{i:04d}",
                ancestor=ancestor,
                branch_a=branch_a,
                branch_b=branch_b,
                conflict_pairs=conflict_pairs,
                source="synthetic",
                metadata={
                    "category": "semantic_resolvable",
                    "template_predicate": template["predicate"],
                    "correct_is_a": correct_is_a,
                    "divergence_span": config.divergence_span,
                },
            )
        )
    return scenarios
