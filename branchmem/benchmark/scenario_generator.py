"""Synthetic branch-divergence scenario generator.

Builds a common-ancestor memory state, forks it, and applies independently
scripted update streams to each branch with a configurable mix of orthogonal
additions, resolvable conflicts, and genuinely ambiguous conflicts — per the
project's requirement that the benchmark test both conflict *resolution* and
correct *preservation* of non-conflicting information, and both "flag as
ambiguous" and "resolve confidently" as valid correct behaviors depending on
scenario type.

Design note on ground truth for resolvable conflicts (revised after a Phase 5
pilot run surfaced a construct-validity problem): a
resolvable conflict's correct value is tied to a source-reliability signal —
one branch's update is "user"-sourced (explicit, high confidence), the other
"inference"-sourced (agent-inferred, lower confidence) — with ground truth
set to the user-sourced branch's value. An earlier version tied ground truth
to nothing visible in the data at all (a coin flip independent of every
observable signal), which made "resolvable" conflicts indistinguishable from
"ambiguous" ones from any merger's point of view — no strategy, however
good, could do better than chance on them, which would have made H1
untestable as designed. Source/confidence are real MemoryFact fields
(entity/predicate/value plus provenance), not new schema; ThreeWayLLMMerge's
prompt was updated to expose them.

`timestamp_correlates_with_correctness` is now independent of *why* a branch
is correct (source reliability) and controls only whether the
ground-truth-correct branch also happens to carry the later timestamp. If
this were always 1.0, LastWriterWins would trivially match the three-way
merge on every resolvable conflict, collapsing the H1 comparison; if always
0.0, LWW would be adversarially defeated by construction. The default (0.5)
makes timestamp ordering uninformative about correctness, which is the
realistic case for asynchronous branches with no shared clock. This choice
is preregistered here, before Phase 5/6 results are seen.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from branchmem.memory.branch_sim import BranchSimulator, UpdateSpec
from branchmem.memory.schemas import ConflictPair, MemoryBranch

_DEFAULT_TEMPLATES_PATH = Path(__file__).resolve().parents[2] / "data" / "scenario_templates.json"


@dataclass
class Scenario:
    scenario_id: str
    ancestor: MemoryBranch
    branch_a: MemoryBranch
    branch_b: MemoryBranch
    conflict_pairs: list[ConflictPair]
    source: str  # "synthetic" | "mab_extension"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioConfig:
    n_entities: int = 1
    n_ancestor_facts: int = 4
    n_orthogonal_a: int = 2
    n_orthogonal_b: int = 2
    n_resolvable: int = 2
    n_ambiguous: int = 1
    timestamp_correlates_with_correctness: float = 0.5
    divergence_span: float = 10.0  # "short" (small) vs "long" (large) disconnection window
    source_signal_noise: float = 0.15  # P(ground truth is the LOWER-reliability branch anyway)


class ScenarioGenerator:
    def __init__(self, templates_path: Path | str = _DEFAULT_TEMPLATES_PATH) -> None:
        with open(templates_path) as f:
            self.templates = json.load(f)
        self.entities: list[str] = self.templates["entities"]
        self.objective: dict[str, list[str]] = self.templates["objective_predicates"]
        self.subjective: dict[str, list[str]] = self.templates["subjective_predicates"]

    def generate(self, n_scenarios: int, config: ScenarioConfig, seed: int) -> list[Scenario]:
        scenarios = []
        for i in range(n_scenarios):
            # Independent, reproducible sub-seed per scenario so a subset can be
            # regenerated/inspected without regenerating the whole batch.
            rng = random.Random(f"{seed}:{i}")
            scenarios.append(self._generate_one(f"synth_{seed}_{i:04d}", config, rng))
        return scenarios

    def _generate_one(self, scenario_id: str, config: ScenarioConfig, rng: random.Random) -> Scenario:
        sim = BranchSimulator()
        entities = rng.sample(self.entities, k=min(config.n_entities, len(self.entities)))

        all_predicates = {**self.objective, **self.subjective}
        ancestor_keys = self._sample_keys(entities, all_predicates, config.n_ancestor_facts, rng)
        ancestor_facts = [(e, p, rng.choice(all_predicates[p])) for e, p in ancestor_keys]
        ancestor = sim.create_common_ancestor(facts=ancestor_facts, timestamp=0.0)
        ancestor_values = {(e, p): v for e, p, v in ancestor_facts}

        # `used_keys` blocks re-picking a key for orthogonal additions or for a
        # second conflict (they must target distinct keys). It is deliberately
        # NOT used to filter `_sample_conflict_key`'s ancestor-reuse candidates
        # below — conflicts SHOULD be free to target real ancestor keys (that's
        # the entire point of testing three-way, ancestor-aware merging); only
        # `consumed_ancestor_keys` (grown as conflicts are picked) excludes an
        # ancestor key already claimed by an earlier conflict this scenario.
        used_keys = set(ancestor_keys)
        consumed_ancestor_keys: set[tuple[str, str]] = set()
        updates_a: list[UpdateSpec] = []
        updates_b: list[UpdateSpec] = []

        # Orthogonal: new keys touched by exactly one branch.
        for _ in range(config.n_orthogonal_a):
            key = self._sample_new_key(entities, all_predicates, used_keys, rng)
            used_keys.add(key)
            e, p = key
            updates_a.append(self._make_update(e, p, all_predicates[p], rng, config, divergence_type="orthogonal"))
        for _ in range(config.n_orthogonal_b):
            key = self._sample_new_key(entities, all_predicates, used_keys, rng)
            used_keys.add(key)
            e, p = key
            updates_b.append(self._make_update(e, p, all_predicates[p], rng, config, divergence_type="orthogonal"))

        # Resolvable conflicts: prefer objective predicates (have a fact-of-the-matter).
        #
        # Ground truth is tied to a source-reliability signal (one branch's
        # update is "user"-sourced/high-confidence, the other "inference"
        # -sourced/low-confidence), NOT to which branch has the later
        # timestamp — a resolvable conflict needs an actual signal a merger
        # could exploit, or it is indistinguishable from an ambiguous one.
        # `timestamp_correlates_with_correctness` independently controls
        # whether the ground-truth-correct branch also happens to have the
        # later timestamp, so LastWriterWins isn't trivially right or wrong
        # by construction (see module docstring).
        for _ in range(config.n_resolvable):
            e, p = self._sample_conflict_key(entities, self.objective, ancestor_keys, consumed_ancestor_keys, used_keys, rng)
            used_keys.add((e, p))
            consumed_ancestor_keys.add((e, p))
            pool = [v for v in self.objective[p] if v != ancestor_values.get((e, p))]
            value_a, value_b = rng.sample(pool, k=2) if len(pool) >= 2 else (pool[0], pool[0])

            a_is_user_sourced = rng.random() < 0.5
            source_a, confidence_a = ("user", 0.95) if a_is_user_sourced else ("inference", 0.55)
            source_b, confidence_b = ("inference", 0.55) if a_is_user_sourced else ("user", 0.95)
            reliable_value = value_a if a_is_user_sourced else value_b
            unreliable_value = value_b if a_is_user_sourced else value_a
            # Source reliability is informative, not deterministic: the
            # higher-confidence branch is occasionally wrong anyway (a user
            # can misspeak; an inference can happen to be right), so a
            # strategy that blindly pattern-matches "always trust user
            # source" doesn't get a free 100%.
            ground_truth = unreliable_value if rng.random() < config.source_signal_noise else reliable_value

            t1 = config.divergence_span * rng.uniform(0.1, 1.0)
            t2 = config.divergence_span * rng.uniform(0.1, 1.0)
            later_ts, earlier_ts = max(t1, t2), min(t1, t2)
            correct_is_a = a_is_user_sourced
            if rng.random() < config.timestamp_correlates_with_correctness:
                ts_a, ts_b = (later_ts, earlier_ts) if correct_is_a else (earlier_ts, later_ts)
            else:
                ts_a, ts_b = (earlier_ts, later_ts) if correct_is_a else (later_ts, earlier_ts)

            updates_a.append(
                UpdateSpec(entity=e, predicate=p, value=value_a, timestamp=1.0 + ts_a,
                           source=source_a, confidence=confidence_a,
                           divergence_type="resolvable_conflict", ground_truth_value=ground_truth)
            )
            updates_b.append(
                UpdateSpec(entity=e, predicate=p, value=value_b, timestamp=1.0 + ts_b,
                           source=source_b, confidence=confidence_b,
                           divergence_type="resolvable_conflict", ground_truth_value=ground_truth)
            )

        # Ambiguous conflicts: prefer subjective predicates (no fact-of-the-matter).
        # Both branches get the SAME source/confidence -- deliberately no
        # distinguishing signal, so the correct behavior is to flag, not guess.
        for _ in range(config.n_ambiguous):
            e, p = self._sample_conflict_key(entities, self.subjective, ancestor_keys, consumed_ancestor_keys, used_keys, rng)
            used_keys.add((e, p))
            consumed_ancestor_keys.add((e, p))
            pool = [v for v in self.subjective[p] if v != ancestor_values.get((e, p))]
            value_a, value_b = rng.sample(pool, k=2) if len(pool) >= 2 else (pool[0], pool[0])
            updates_a.append(
                UpdateSpec(entity=e, predicate=p, value=value_a,
                           timestamp=1.0 + config.divergence_span * rng.uniform(0.1, 1.0),
                           source="observation", confidence=0.7,
                           divergence_type="ambiguous_conflict")
            )
            updates_b.append(
                UpdateSpec(entity=e, predicate=p, value=value_b,
                           timestamp=1.0 + config.divergence_span * rng.uniform(0.1, 1.0),
                           source="observation", confidence=0.7,
                           divergence_type="ambiguous_conflict")
            )

        branch_a, branch_b, conflict_pairs = sim.diverge(ancestor, updates_a, updates_b, fork_point_timestamp=1.0)
        return Scenario(
            scenario_id=scenario_id,
            ancestor=ancestor,
            branch_a=branch_a,
            branch_b=branch_b,
            conflict_pairs=conflict_pairs,
            source="synthetic",
            metadata={
                "entities": entities,
                "n_orthogonal": config.n_orthogonal_a + config.n_orthogonal_b,
                "n_resolvable": config.n_resolvable,
                "n_ambiguous": config.n_ambiguous,
                "divergence_span": config.divergence_span,
                "timestamp_correlates_with_correctness": config.timestamp_correlates_with_correctness,
            },
        )

    @staticmethod
    def _make_update(e: str, p: str, values: list[str], rng: random.Random, config: ScenarioConfig, divergence_type: str) -> UpdateSpec:
        return UpdateSpec(
            entity=e, predicate=p, value=rng.choice(values),
            timestamp=1.0 + config.divergence_span * rng.uniform(0.1, 1.0),
            divergence_type=divergence_type,
        )

    @staticmethod
    def _sample_keys(
        entities: list[str], predicates: dict[str, list[str]], n: int, rng: random.Random
    ) -> list[tuple[str, str]]:
        candidates = [(e, p) for e in entities for p in predicates]
        rng.shuffle(candidates)
        return candidates[:n]

    @staticmethod
    def _sample_new_key(
        entities: list[str], predicates: dict[str, list[str]], used: set[tuple[str, str]], rng: random.Random
    ) -> tuple[str, str]:
        candidates = [(e, p) for e in entities for p in predicates if (e, p) not in used]
        if not candidates:
            raise ValueError("Ran out of unused (entity, predicate) keys — widen the template pool or lower counts.")
        return rng.choice(candidates)

    @staticmethod
    def _sample_conflict_key(
        entities: list[str],
        predicates: dict[str, list[str]],
        ancestor_keys: list[tuple[str, str]],
        consumed_ancestor_keys: set[tuple[str, str]],
        used: set[tuple[str, str]],
        rng: random.Random,
    ) -> tuple[str, str]:
        # Prefer reusing an existing ancestor key (a real "update" to something
        # already known, giving ThreeWayLLMMerge actual ancestor context to
        # reason from); fall back to a fresh shared key (both branches
        # independently invent the same new fact with no ancestor value at
        # all) only if every matching ancestor key is already claimed by an
        # earlier conflict this scenario.
        ancestor_candidates = [k for k in ancestor_keys if k[1] in predicates and k not in consumed_ancestor_keys]
        if ancestor_candidates:
            return rng.choice(ancestor_candidates)
        return ScenarioGenerator._sample_new_key(entities, predicates, used, rng)
