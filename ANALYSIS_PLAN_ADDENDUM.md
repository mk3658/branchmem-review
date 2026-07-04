# ANALYSIS_PLAN_ADDENDUM.md

`ANALYSIS_PLAN.md` is locked and was not edited after Phase 6. Everything in
this file is **post-hoc and exploratory** — added after seeing the locked
results. None of it is folded into the confirmatory tests (Table 1/2 in the
paper) or treated as satisfying H1/H2/H3.
Where a number here is compared against a locked-run number, the comparison
itself is descriptive, not a new preregistered test.

## A1. Abstention-aware metrics (added 2026-07-04)

Motivation: raw accuracy penalizes `FLAGGED_UNRESOLVED` identically whether
the underlying situation is genuinely ambiguous (correct behavior) or
resolvable-with-signal (a missed resolution). In a safety-sensitive memory
merge, a wrong commit and an honest abstention are not equally bad. This
addendum computes, from the **already-cached results** (no new API calls):
commit rate, abstention rate, conditional accuracy given commit, flag
precision/recall, wrong-commit rate, and expected utility under three cost
regimes (correct commit +1; wrong commit −2/−5/−10; abstention −1).
Implementation: `branchmem/eval/abstention_metrics.py`.

## A2. `semantic_resolvable` conflict category (added 2026-07-04)

Motivation: every existing "resolvable" conflict in the locked generator
ties ground truth to a source/confidence signal that a single `if`-statement
(`ConfidenceRuleMerge`) can read directly. This category equalizes source
and confidence across both branches and makes timestamp uninformative, so
ground truth is recoverable only from semantic compatibility with the
ancestor/context — a condition genuinely hard for `ConfidenceRuleMerge` and
genuinely informative about whether an LLM's semantic reasoning adds value
beyond reading metadata. This is a **new, non-preregistered generator
condition**, not a reinterpretation of the locked `resolvable` category.
Sample size, exact templates, and results are in the paper's new subsection
and `results/final/semantic_resolvable.json`.

## A3. Balanced conflict-detector benchmark (added 2026-07-04)

Motivation: RQ2's locked evaluation set contains only genuine conflict
pairs by construction (an orthogonal key is never shared between branches),
so every detector's precision is mechanically 1.0 — this measures recall
only, not real-world discriminative power. This addendum adds five negative
categories (paraphrase, entailment, unrelated-same-entity,
preference-refinement, ambiguous-but-not-conflicting) alongside genuine
contradictions, and reports full precision/recall/F1/specificity/FPR (and
AUROC where a continuous score is available) for
`EmbeddingConflictDetector`, `NLIConflictDetector`, and
`LLMJudgeConflictDetector`. This does not change or retract RQ2's locked
result — it is a harder, separate, descriptive follow-up.

## A4. Independent-validation audit protocol (added 2026-07-04)

No human annotation is collected in this project (no annotator available in
this environment). `annotation/audit_sample.csv` and
`annotation/audit_instructions.md` provide a sampling protocol and template
so an independent party could validate ground-truth labels, extraction
correctness, and preferred resolutions. The paper does not claim this
validation was performed — only that the protocol and a sample exist.

## Standing rule for this addendum

Any further post-hoc analysis added after this point must be appended here
with the same structure (motivation, exact scope, what it does and does not
change about the locked/confirmatory results) before being reported in the
paper.
