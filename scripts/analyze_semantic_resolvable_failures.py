#!/usr/bin/env python3
"""Post-hoc, exploratory: failure-mode diagnostic for the semantic_resolvable
category (see ANALYSIS_PLAN_ADDENDUM.md item A2). NO new API calls --
regenerates the identical 150 scenarios (seed 3001) and replays
ThreeWayLLMMerge against the committed cache, then categorizes each per-key
resolution by (a) what the model did (flag / merge / commit-correct /
commit-wrong) and (b) the reasoning phenomenon the template requires. Answers
the reviewer's question of which reasoning patterns are most error-prone.

The phenomenon labels below are the authors' manual classification of the 12
hand-authored templates in data/semantic_resolvable_templates.json; they are
descriptive, not learned.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from branchmem.benchmark.semantic_resolvable_generator import generate_semantic_resolvable_scenarios
from branchmem.evaluation.metrics import score_downstream
from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.llm.base import build_backend
from branchmem.merge.three_way_llm import ThreeWayLLMMerge
from branchmem.utils.seeding import set_all_seeds

SEED = 3001
N_SCENARIOS = 150
MODEL = "gpt-5.4-nano"
MAX_TOKENS = 8000
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "final"

# Manual classification of each template predicate by the reasoning phenomenon
# a correct resolution requires (see data/semantic_resolvable_templates.json).
PHENOMENON = {
    "dinner_choice":       "constraint-exclusion",   # vegetarian -> reject meat
    "sauce_choice":        "constraint-exclusion",   # peanut allergy -> reject peanut
    "pairing_choice":      "constraint-exclusion",   # no alcohol -> reject wine
    "seating_choice":      "constraint-exclusion",   # no smoke -> reject smoking section
    "transport_choice":    "constraint-exclusion",   # boat motion sickness -> reject ferry
    "workout_choice":      "constraint-exclusion",   # knee injury -> reject high-impact
    "venue_choice":        "precondition",           # needs ramp -> reject stairs-only
    "gift_choice":         "precondition",           # no pet -> reject pet gift
    "childcare_choice":    "precondition",           # toddler -> reject adults-only
    "meeting_time_choice": "numeric-temporal",       # before noon -> reject 3pm
    "hotel_choice":        "attribute-matching",     # quiet -> reject nightclub-adjacent
    "room_choice":         "attribute-matching",     # dislikes AC -> reject AC-on
}


def main() -> None:
    set_all_seeds(SEED)
    backend = build_backend({"backend": "openai_compatible", "model": MODEL,
                             "cache_dir": "llm_cache", "temperature": 0.0, "max_tokens": MAX_TOKENS})
    strat = ThreeWayLLMMerge(backend=backend)
    scenarios = generate_semantic_resolvable_scenarios(N_SCENARIOS, seed=SEED)

    # outcome per key: correct_commit | wrong_commit | flagged | merged
    by_phenomenon = defaultdict(lambda: defaultdict(int))
    by_outcome_total = defaultdict(int)
    n_keys = 0

    for scenario in scenarios:
        predicate = scenario.metadata["template_predicate"]
        phenom = PHENOMENON.get(predicate, "other")
        questions = generate_downstream_questions(scenario)
        merged = strat.merge(scenario.ancestor, scenario.branch_a, scenario.branch_b)
        current = {}
        for rf in merged.resulting_facts:
            if rf.resolution.value == "dropped":
                continue
            current[(rf.fact.entity, rf.fact.predicate)] = (rf.fact.value, rf.resolution.value)
        for q in questions:
            if q.category != "resolvable":
                continue
            n_keys += 1
            got_value, res_str = current.get((q.entity, q.predicate), (None, None))
            if res_str == "flagged_unresolved":
                outcome = "flagged"
            elif res_str == "merged":
                outcome = "merged"
            elif got_value == q.expected_answer:
                outcome = "correct_commit"
            else:
                outcome = "wrong_commit"
            by_phenomenon[phenom][outcome] += 1
            by_outcome_total[outcome] += 1

    result = {
        "note": (
            "post-hoc, exploratory failure-mode diagnostic for semantic_resolvable "
            "(n=150, seed=3001). No new API calls: replays cached ThreeWayLLMMerge "
            "responses. Phenomenon labels are the authors' manual classification of the "
            "12 templates. NOT part of ANALYSIS_PLAN.md confirmatory tests."
        ),
        "n_resolvable_keys": n_keys,
        "outcome_totals": dict(by_outcome_total),
        "by_phenomenon": {ph: dict(d) for ph, d in by_phenomenon.items()},
    }
    out_path = OUT_DIR / "semantic_resolvable_failure_modes.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("Wrote", out_path)
    print(json.dumps(result["outcome_totals"], indent=2))
    print(json.dumps(result["by_phenomenon"], indent=2))


if __name__ == "__main__":
    main()
