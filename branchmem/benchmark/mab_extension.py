"""Extends MemoryAgentBench's Fact Consolidation task (Hu, Wang & McAuley,
ICLR 2026, arXiv:2507.05257) with a branch-fork step.

Accessibility check (done during Phase 4): the dataset
IS publicly accessible — MIT-licensed code at
https://github.com/HUST-AI-HYZ/MemoryAgentBench and data at
https://huggingface.co/datasets/ai-hyz/MemoryAgentBench (Conflict_Resolution
config, `fact_sh`/`fact_mh` sources). However, the published parquet only
exposes a flattened natural-language `context` string (a numbered list of
templated sentences derived from MQuAKE-style Wikidata triples) — there is no
structured (subject, relation, object) field in the public schema. Reusing
this data for our branch-structured, entity/predicate/value-typed pipeline
therefore requires re-parsing the natural-language sentences back into
triples ourselves.

What is reused: the actual fact content, its ordering (which encodes MAB's
own freshness ground truth — later restatements of the same (subject,
relation) supersede earlier ones), and the overall "numbered list of facts
with counterfactual edits" structure MAB built from MQuAKE.

What is newly built: (1) the sentence-template regex parser below
(`parse_context_facts`), calibrated against a real sample of the dataset
(measured at 99.9% coverage of 18,337 unique context lines across 38
hand-written templates); (2) the branch-fork step
(`build_branch_scenario_from_mab_row`), which is BranchMem's actual research
contribution and has no analogue in MAB (MAB's Fact Consolidation is a single
continuous timeline, not two disconnected branches); (3) ground-truth
resolvable/ambiguous conflict labeling for the branch-split version.

Ground truth derivation for the branch-split: MAB's edit chains for a given
(subject, relation) key already have a defined "correct" final value (the
last edit in their original total order). When we split a chain's post-first
edits across the two branches, whichever branch ends up containing MAB's
*true final* edit is assigned as the ground-truth-correct branch for that
key (divergence_type="resolvable_conflict"). MAB's Fact Consolidation task is
deterministic by construction — it has no "genuinely ambiguous, no correct
answer" cases — so this extension does not produce ambiguous_conflict
divergences; only scenario_generator.py's from-scratch scenarios cover H1's
ambiguous/flagging behavior. This is a real limitation, documented rather
than hidden.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import pandas as pd

from branchmem.benchmark.scenario_generator import Scenario
from branchmem.memory.branch_sim import BranchSimulator, UpdateSpec
from branchmem.utils.logging import get_logger

logger = get_logger(__name__)

_PARQUET_URL = (
    "https://huggingface.co/api/datasets/ai-hyz/MemoryAgentBench/parquet/"
    "default/Conflict_Resolution/0.parquet"
)

# (relation_name, regex). Order matters: more specific patterns first.
# Coverage measured at 99.9% of 18,337 unique context lines sampled from the
# public Conflict_Resolution split (see docstring above).
_TEMPLATES: list[tuple[str, str]] = [
    ("born_in_city", r"^(?P<subject>.+?) was born in the city of (?P<value>.+)\.$"),
    ("died_in_city", r"^(?P<subject>.+?) died in the city of (?P<value>.+)\.$"),
    ("citizen_of", r"^(?P<subject>.+?) is a citizen of (?P<value>.+)\.$"),
    ("speaks_language", r"^(?P<subject>.+?) speaks the language of (?P<value>.+)\.$"),
    ("religion", r"^(?P<subject>.+?) is affiliated with the religion of (?P<value>.+)\.$"),
    ("sport", r"^(?P<subject>.+?) is associated with the sport of (?P<value>.+)\.$"),
    ("plays_position", r"^(?P<subject>.+?) plays the position of (?P<value>.+)\.$"),
    ("founded_in_city", r"^(?P<subject>.+?) was founded in the city of (?P<value>.+)\.$"),
    ("created_in_country", r"^(?P<subject>.+?) was created in the country of (?P<value>.+)\.$"),
    ("developed_by", r"^(?P<subject>.+?) was developed by (?P<value>.+)\.$"),
    ("founded_by", r"^(?P<subject>.+?) was founded by (?P<value>.+)\.$"),
    ("performed_by", r"^(?P<subject>.+?) was performed by (?P<value>.+)\.$"),
    ("child_is", r"^(?P<subject>.+?)'s child is (?P<value>.+)\.$"),
    ("worked_in_city", r"^(?P<subject>.+?) worked in the city of (?P<value>.+)\.$"),
    ("continent", r"^(?P<subject>.+?) is located in the continent of (?P<value>.+)\.$"),
    ("married_to", r"^(?P<subject>.+?) is married to (?P<value>.+)\.$"),
    ("created_by", r"^(?P<subject>.+?) was created by (?P<value>.+)\.$"),
    ("employed_by", r"^(?P<subject>.+?) is employed by (?P<value>.+)\.$"),
    ("written_in_language", r"^(?P<subject>.+?) was written in the language of (?P<value>.+)\.$"),
    ("famous_for", r"^(?P<subject>.+?) is famous for (?P<value>.+)\.$"),
    ("works_in_field", r"^(?P<subject>.+?) works in the field of (?P<value>.+)\.$"),
    ("chairperson_of", r"^The chairperson of (?P<subject>.+?) is (?P<value>.+)\.$"),
    ("director_of", r"^The director of (?P<subject>.+?) is (?P<value>.+)\.$"),
    ("ceo_of", r"^The chief executive officer of (?P<subject>.+?) is (?P<value>.+)\.$"),
    ("hq_city", r"^The headquarters of (?P<subject>.+?) is located in the city of (?P<value>.+)\.$"),
    ("capital_of", r"^The capital of (?P<subject>.+?) is (?P<value>.+)\.$"),
    ("author_of", r"^The author of (?P<subject>.+?) is (?P<value>.+)\.$"),
    ("educated_at", r"^The univeristy where (?P<subject>.+?) was educated is (?P<value>.+)\.$"),
    ("educated_at2", r"^The university where (?P<subject>.+?) was educated is (?P<value>.+)\.$"),
    ("produced_by", r"^The company that produced (?P<subject>.+?) is (?P<value>.+)\.$"),
    ("music_type", r"^The type of music that (?P<subject>.+?) plays is (?P<value>.+)\.$"),
    ("gov_head", r"^The name of the current head of the (?P<subject>.+?) government is (?P<value>.+)\.$"),
    ("head_of_state", r"^The name of the current head of state in (?P<subject>.+?) is (?P<value>.+)\.$"),
    ("official_language", r"^The official language of (?P<subject>.+?) is (?P<value>.+)\.$"),
    ("original_language", r"^The original language of (?P<subject>.+?) is (?P<value>.+)\.$"),
    ("head_coach_of", r"^The head coach of (?P<subject>.+?) is (?P<value>.+)\.$"),
    ("original_broadcaster", r"^The origianl broadcaster of (?P<subject>.+?) is (?P<value>.+)\.$"),
    ("office_role_of", r"^The (?P<title>[A-Z][A-Za-z .'-]+?) of (?P<subject>.+?) is (?P<value>.+)\.$"),
]
_COMPILED_TEMPLATES = [(name, re.compile(pat)) for name, pat in _TEMPLATES]


@dataclass
class ParsedFact:
    line_index: int
    entity: str
    predicate: str
    value: str


def parse_context_facts(context: str) -> tuple[list[ParsedFact], int, int]:
    """Parse a MAB `context` blob into (entity, predicate, value) triples.

    Returns (parsed_facts, n_matched, n_total_lines). Unparseable lines are
    skipped, not guessed — callers should check the coverage ratio and decide
    whether it's acceptable for their use, not assume 100%.
    """
    lines = [ln for ln in context.strip().split("\n") if ln.strip()]
    parsed: list[ParsedFact] = []
    n_matched = 0
    n_total = 0
    for idx, raw_line in enumerate(lines):
        line = re.sub(r"^\d+\.\s+", "", raw_line).strip()
        if line == "Here is a list of facts:" or not line:
            continue
        n_total += 1
        for name, pattern in _COMPILED_TEMPLATES:
            m = pattern.match(line)
            if m:
                groups = m.groupdict()
                predicate = f"{name}_{groups['title']}" if "title" in groups else name
                parsed.append(
                    ParsedFact(line_index=idx, entity=groups["subject"].strip(), predicate=predicate, value=groups["value"].strip())
                )
                n_matched += 1
                break
    return parsed, n_matched, n_total


def load_mab_conflict_resolution() -> pd.DataFrame:
    """Download the public Conflict_Resolution parquet split. Raises a clear
    error on failure rather than silently falling back to synthetic data —
    callers who want a network-independent path should use
    scenario_generator.py directly."""
    try:
        return pd.read_parquet(_PARQUET_URL)
    except Exception as exc:  # network/library errors vary by backend
        raise RuntimeError(
            f"Could not load MemoryAgentBench Conflict_Resolution split from "
            f"{_PARQUET_URL}: {exc}. This requires network access to "
            f"huggingface.co. Use scenario_generator.py for a fully "
            f"offline/synthetic benchmark instead."
        ) from exc


def build_branch_scenario_from_mab_row(
    row: pd.Series, scenario_id: str, seed: int, fork_point_timestamp: float = 1.0,
    max_conflict_keys: int | None = None,
) -> Scenario | None:
    """Split one MAB Conflict_Resolution row's edit chains across two branches.

    MAB's Fact Consolidation chains are almost always exactly one counterfactual
    edit deep (ancestor value + one MQuAKE-style replacement) — measured on the
    live dataset, ~55% of (entity, predicate) keys with any edit have exactly
    one, and essentially none have two or more in the sampled rows. A single
    edit has nowhere to go in a two-branch fork (it can only land on one
    branch, which makes it orthogonal, not a conflict) — so a strategy that
    only used chains with >=2 real edits found ~0 conflicts per row on
    real data, which we discovered when validating the pipeline end to end.

    So for the common single-edit case, this function reuses MAB's real
    ancestor value and real edit for branch A's side of the conflict, and
    constructs branch B's competing value by resampling another value that
    genuinely occurs elsewhere in this row for the SAME predicate (e.g.
    another city that appears as some other entity's `born_in_city` value in
    the same context) — i.e., real MAB values, recombined onto a different
    entity, standing in for "what an independently-disconnected branch might
    have written instead." Ground truth is always MAB's real edit. This
    synthetic recombination is flagged in the returned scenario's metadata
    (`n_synthetic_second_branch_edits`) and is NOT literal MAB reuse for that
    branch's value — documented per the module's reuse-vs-newly-built split.

    For the rare case of a genuine >=2-edit chain, both branches get real,
    distinct MAB edits with no synthetic augmentation, and whichever branch
    ends up with MAB's true chain-final edit is ground truth.

    Returns None if the row yields no usable conflict key at all (e.g.
    parsing coverage was too low, or no predicate had any alternative value
    to construct a conflict from).

    `max_conflict_keys`, if set, caps how many (entity, predicate) chains are
    actually turned into branch conflicts (sampled with `seed`), keeping every
    other parsed fact as unconflicted ancestor context. Real MAB rows can have
    thousands of edit chains (e.g. one Conflict_Resolution row observed with
    7221), which would blow both the ThreeWayLLMMerge prompt (all of a
    scenario's collisions are batched into one call) past reasonable size and
    cost. This does not change what's real vs. synthetic per conflict (see
    above) — it only bounds how many of a row's real conflicts are evaluated
    in one scenario.
    """
    rng = random.Random(seed)
    parsed, n_matched, n_total = parse_context_facts(row["context"])
    if n_total and n_matched / n_total < 0.5:
        logger.warning(
            "scenario %s: low MAB parse coverage %d/%d (%.0f%%), results may be unrepresentative",
            scenario_id, n_matched, n_total, 100 * n_matched / n_total,
        )

    chains: dict[tuple[str, str], list[ParsedFact]] = {}
    for fact in parsed:
        chains.setdefault((fact.entity, fact.predicate), []).append(fact)
    for key in chains:
        chains[key].sort(key=lambda f: f.line_index)

    conflict_keys = set(chains.keys())
    if max_conflict_keys is not None and len(chains) > max_conflict_keys:
        sampled = rng.sample(sorted(chains.keys()), max_conflict_keys)
        conflict_keys = set(sampled)

    values_by_predicate: dict[str, set[str]] = {}
    for fact in parsed:
        values_by_predicate.setdefault(fact.predicate, set()).add(fact.value)

    sim = BranchSimulator()
    ancestor_facts = [(key[0], key[1], chain[0].value) for key, chain in chains.items()]
    if not ancestor_facts:
        return None
    ancestor = sim.create_common_ancestor(facts=ancestor_facts, timestamp=0.0)

    updates_a: list[UpdateSpec] = []
    updates_b: list[UpdateSpec] = []
    n_synthetic = 0
    for (entity, predicate), chain in chains.items():
        if (entity, predicate) not in conflict_keys:
            continue
        edits = chain[1:]  # values after the ancestor's
        if not edits:
            continue

        if len(edits) >= 2:
            # Randomize which half of the real chain lands on which branch,
            # rather than always giving the later half (and thus later
            # timestamp + true-final edit) to branch B — that would rig
            # LastWriterWins to always win this path by construction.
            split_idx = rng.randint(1, len(edits) - 1)
            first_half, second_half = edits[:split_idx], edits[split_idx:]
            first_is_a = rng.random() < 0.5
            a_edits, b_edits = (first_half, second_half) if first_is_a else (second_half, first_half)
            a_value, a_ts = a_edits[-1].value, 1.0 + a_edits[-1].line_index
            b_value, b_ts = b_edits[-1].value, 1.0 + b_edits[-1].line_index
            ground_truth = a_value if chain[-1] in a_edits else b_value
        else:
            real_value, real_ts = edits[0].value, 1.0 + edits[0].line_index
            candidates = [v for v in values_by_predicate.get(predicate, set()) if v not in (real_value, chain[0].value)]
            if not candidates:
                branch_updates = updates_a if rng.random() < 0.5 else updates_b
                branch_updates.append(
                    UpdateSpec(entity=entity, predicate=predicate, value=real_value,
                               timestamp=real_ts, divergence_type="orthogonal")
                )
                continue
            synthetic_value = rng.choice(candidates)
            # Randomize branch assignment AND whether the synthetic edit's
            # timestamp is earlier or later than the real edit's — otherwise
            # the synthetic branch is always later and LWW always loses (or
            # always wins), which is a rigged, not a fair, test.
            synthetic_ts = max(0.0, real_ts + rng.choice([-1, 1]) * rng.uniform(0.1, 2.0))
            if rng.random() < 0.5:
                a_value, a_ts, b_value, b_ts = real_value, real_ts, synthetic_value, synthetic_ts
            else:
                a_value, a_ts, b_value, b_ts = synthetic_value, synthetic_ts, real_value, real_ts
            ground_truth = real_value
            n_synthetic += 1

        updates_a.append(
            UpdateSpec(entity=entity, predicate=predicate, value=a_value, timestamp=a_ts,
                       divergence_type="resolvable_conflict", ground_truth_value=ground_truth)
        )
        updates_b.append(
            UpdateSpec(entity=entity, predicate=predicate, value=b_value, timestamp=b_ts,
                       divergence_type="resolvable_conflict", ground_truth_value=ground_truth)
        )

    if not updates_a and not updates_b:
        return None

    branch_a, branch_b, conflict_pairs = sim.diverge(ancestor, updates_a, updates_b, fork_point_timestamp)
    return Scenario(
        scenario_id=scenario_id,
        ancestor=ancestor,
        branch_a=branch_a,
        branch_b=branch_b,
        conflict_pairs=conflict_pairs,
        source="mab_extension",
        metadata={
            "mab_source": row["metadata"].get("source") if isinstance(row["metadata"], dict) else None,
            "parse_coverage": n_matched / n_total if n_total else 0.0,
            "n_chains": len(chains),
            "n_conflict_keys_used": len(conflict_keys),
            "max_conflict_keys": max_conflict_keys,
            "n_synthetic_second_branch_edits": n_synthetic,
        },
    )
