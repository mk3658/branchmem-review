#!/usr/bin/env python3
"""Post-hoc, exploratory: provenance-proxy experiment (see
ANALYSIS_PLAN_ADDENDUM.md). NOT part of ANALYSIS_PLAN.md confirmatory tests.

Question (reviewer Q6): the locked benchmark exposes the reliability signal at
*per-fact* granularity (each conflicting fact carries its own source/confidence).
A realistic deployment more often has provenance at *branch* (replica)
granularity -- "this replica has been reliable historically" -- not per fact.
Does the LLM (and a deterministic rule) exploit a branch-level reliability
proxy as well as the per-fact engineered signal?

Construction: one entity, a 4-fact ancestor, and several resolvable conflicts.
Unlike the locked generator (which randomizes which branch is reliable *per
conflict*), here ONE branch is the reliable replica for the WHOLE scenario:
all its conflicting updates are source="user"/confidence=0.9, the other
branch's are source="inference"/confidence=0.5, and ground truth is the
reliable replica's value with the same 15% noise. The per-fact fields are thus
a *constant per-branch* stand-in for a track record. We run the UNCHANGED
ThreeWayLLMMerge and a branch-level deterministic rule, and compare resolvable
accuracy to the per-fact locked result (0.858 three-way / 0.867 rule).

This is a fair, non-circular test: no ground-truth peeking; the proxy is the
kind of coarse provenance a real system could actually maintain.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.scenario_generator import Scenario
from branchmem.memory.branch_sim import BranchSimulator, UpdateSpec
from branchmem.evaluation.metrics import score_downstream
from branchmem.evaluation.stats import paired_comparison
from branchmem.llm.base import build_backend
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.last_writer_wins import LastWriterWins
from branchmem.merge.three_way_llm import ThreeWayLLMMerge
from branchmem.utils.seeding import set_all_seeds

SEED = 5001  # disjoint from locked (2026), power-exp (2029), semres (3001)
N_SCENARIOS = 60
N_RESOLVABLE = 2
N_ANCESTOR = 4
NOISE = 0.15
MODEL = "gpt-5.4-nano"
MAX_TOKENS = 1024
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "final"
TEMPLATES = Path(__file__).resolve().parents[1] / "data" / "scenario_templates.json"


def build_scenarios(n, seed):
    pools = json.loads(TEMPLATES.read_text())
    entities = pools["entities"]
    objective = pools["objective_predicates"]
    sim = BranchSimulator()
    scenarios = []
    for i in range(n):
        rng = random.Random(f"provproxy:{seed}:{i}")
        entity = rng.choice(entities)
        preds = rng.sample(list(objective.keys()), k=min(N_ANCESTOR + N_RESOLVABLE, len(objective)))
        ancestor_preds = preds[:N_ANCESTOR]
        conflict_preds = preds[N_ANCESTOR:N_ANCESTOR + N_RESOLVABLE]
        ancestor_facts = [(entity, p, rng.choice(objective[p])) for p in ancestor_preds]
        ancestor = sim.create_common_ancestor(facts=ancestor_facts, timestamp=0.0)

        # ONE reliable replica for the whole scenario (branch-level track record).
        a_is_reliable = rng.random() < 0.5
        updates_a, updates_b = [], []
        for p in conflict_preds:
            pool = objective[p]
            va, vb = rng.sample(pool, k=2)
            reliable_val, unreliable_val = (va, vb) if a_is_reliable else (vb, va)
            ground_truth = unreliable_val if rng.random() < NOISE else reliable_val
            # branch-level: reliable replica's facts all user/0.9, other all inference/0.5
            src_a, conf_a = ("user", 0.9) if a_is_reliable else ("inference", 0.5)
            src_b, conf_b = ("inference", 0.5) if a_is_reliable else ("user", 0.9)
            ta = 1.0 + rng.uniform(0.1, 10.0)
            tb = 1.0 + rng.uniform(0.1, 10.0)
            updates_a.append(UpdateSpec(entity=entity, predicate=p, value=va, timestamp=ta,
                                        source=src_a, confidence=conf_a,
                                        divergence_type="resolvable_conflict", ground_truth_value=ground_truth))
            updates_b.append(UpdateSpec(entity=entity, predicate=p, value=vb, timestamp=tb,
                                        source=src_b, confidence=conf_b,
                                        divergence_type="resolvable_conflict", ground_truth_value=ground_truth))
        ba, bb, pairs = sim.diverge(ancestor, updates_a, updates_b, fork_point_timestamp=1.0)
        scenarios.append(Scenario(scenario_id=f"provproxy_{seed}_{i:04d}", ancestor=ancestor,
                                  branch_a=ba, branch_b=bb, conflict_pairs=pairs, source="synthetic",
                                  metadata={"category": "provenance_proxy_branch_level",
                                            "a_is_reliable": a_is_reliable}))
    return scenarios


def main() -> None:
    set_all_seeds(SEED)
    backend = build_backend({"backend": "openai_compatible", "model": MODEL, "cache_dir": "llm_cache",
                             "temperature": 0.0, "max_tokens": MAX_TOKENS})
    strategies = {
        "last_writer_wins": LastWriterWins(),
        "confidence_rule": ConfidenceRuleMerge(),
        "three_way_llm": ThreeWayLLMMerge(backend=backend),
    }
    scenarios = build_scenarios(N_SCENARIOS, SEED)
    resolvable = {n: [] for n in strategies}
    for scenario in scenarios:
        questions = generate_downstream_questions(scenario)
        for name, strat in strategies.items():
            merged = strat.merge(scenario.ancestor, scenario.branch_a, scenario.branch_b)
            resolvable[name].append(score_downstream(questions, merged).accuracy_for("resolvable"))

    def mean(v):
        v = [x for x in v if x == x]
        return sum(v) / len(v) if v else float("nan")

    means = {n: mean(v) for n, v in resolvable.items()}
    test = paired_comparison("confidence_rule", resolvable["confidence_rule"],
                             "three_way_llm", resolvable["three_way_llm"], seed=SEED)
    result = {
        "note": (
            "post-hoc, exploratory provenance-proxy experiment: reliability signal at "
            "BRANCH (replica) granularity instead of per-fact. Same n=60, noise=0.15 as the "
            "locked resolvable construction. Tests whether the LLM/rule exploit a coarse "
            "per-branch track record as well as the per-fact engineered signal. NOT part of "
            "ANALYSIS_PLAN.md confirmatory tests."
        ),
        "run_metadata": {"seed": SEED, "n_scenarios": N_SCENARIOS, "noise": NOISE, "model": MODEL},
        "resolvable_mean_accuracy": means,
        "locked_per_fact_reference": {"three_way_llm": 0.858, "confidence_rule": 0.867},
        "confidence_rule_vs_three_way": {"mean_diff": test.mean_diff, "ci_low": test.ci_low,
                                         "ci_high": test.ci_high, "p_value": test.p_value},
    }
    out_path = OUT_DIR / "provenance_proxy.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("Wrote", out_path)
    print(json.dumps(means, indent=2))


if __name__ == "__main__":
    main()
