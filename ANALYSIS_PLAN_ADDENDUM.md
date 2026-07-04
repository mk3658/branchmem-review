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

## A5. Cross-vendor robustness check (added 2026-07-04)

Motivation: the existing robustness check (`results/final/robustness_second_model.json`)
swaps `gpt-5.4-nano` for `gpt-5.4-mini` — a same-provider, same-family model
swap — leaving open whether the paper's effects are properties of LLMs in
general or specific to this provider's models. With real access to a second
vendor's API (Anthropic) obtained after the locked run, three existing
post-hoc analyses were replicated with `claude-haiku-4-5-20251001` (chosen
to match the "cheapest available model" cost tier already used for the
OpenAI-side results), on identical scenario constructions (same seeds):

- **RQ1, same 20 scenarios as the `gpt-5.4-mini` check**
  (`results/final/cross_vendor_rq1.json`): `claude-haiku-4-5` scores
  $0.936$ mean accuracy vs. $0.914$ (`gpt-5.4-nano`) and $0.929$
  (`gpt-5.4-mini`) on the identical scenarios — paired diff. vs. nano
  $+0.021$, $95\%$ CI $[0.000, 0.043]$, $p{=}0.10$. Direction and magnitude
  hold across a genuinely different vendor, not just a same-provider swap.
- **`semantic_resolvable`, identical 150-scenario construction**
  (`results/final/cross_vendor_semantic_resolvable.json`): every
  confidence-independent strategy, including `ThreeWayLLMMerge`,
  `TwoWayLLMMerge`, and `RawTextLLMMerge`, again scores **exactly $0.000$**
  — the same striking negative result as the original `gpt-5.4-nano` run,
  now shown to generalize across vendors rather than being an artifact of
  one cheap model.
- **Real MemoryAgentBench content** (`results/final/cross_vendor_mab.json`):
  `ThreeWayLLMMerge` again scores exactly $0.000$ (vs. `gpt-5.4-nano`'s
  $0.004$), confirming complete abstention on real content is a property of
  the data (no source-reliability signal exists to read), not an
  OpenAI-specific quirk. Note: this replication's naive-baseline numbers
  (e.g. `last_writer_wins` $0.355$ vs. the original $0.506$) differ from
  the original MAB secondary analysis because that earlier ad hoc script
  did not record the random seed used for branch construction — a
  reproducibility gap in that earlier analysis, now fixed going forward
  (this script uses and discloses `seed=2026`). This does not affect the
  `ThreeWayLLMMerge`/`ConfidenceRuleMerge` result: both are forced to
  exactly $0$ by the data's construction (source/confidence are never
  randomized, only which value lands on which branch is), so that result
  is seed-invariant; only the naive baselines' absolute values are
  seed-sensitive, as expected for a randomized construction.

Scripts: `scripts/run_cross_vendor_rq1.py`,
`scripts/run_cross_vendor_semantic_resolvable.py`,
`scripts/run_cross_vendor_mab.py`. Does not change or retract any
confirmatory or prior post-hoc result — it is additional, disclosed
evidence on the "single model family" limitation.

## Standing rule for this addendum

Any further post-hoc analysis added after this point must be appended here
with the same structure (motivation, exact scope, what it does and does not
change about the locked/confirmatory results) before being reported in the
paper.
