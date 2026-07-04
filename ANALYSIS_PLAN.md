# ANALYSIS_PLAN.md — preregistered before Phase 6

Written after the Phase 5 pilot (`results/pilot/pilot_summary.json`, 20 synthetic
scenarios, `gpt-5.4-nano`, seed=2026) and BEFORE the Phase 6 locked run. Per
project protocol, this plan is not changed after seeing Phase 6 results except
for a documented bug fix, logged in `PROGRESS.md` with a note on whether it
could have affected pilot conclusions.

## 0. What changed during piloting (before this plan was written)

Preregistration happens once, after the pilot is done informing it — the pilot
itself is allowed to iterate. Three fixes were made to the scenario generator
and `ThreeWayLLMMerge` during Phase 5, each logged in `PROGRESS.md` with the
bug/gap that motivated it:

1. **Ancestor-context bug**: `_sample_conflict_key`'s "prefer an existing
   ancestor key" logic was dead code (a pre-seeded `used_keys` set excluded
   every ancestor key before conflicts were picked), so every conflict in the
   first pilot run targeted a brand-new key with NO ancestor value —
   `ThreeWayLLMMerge` never actually got the ancestor context its own design
   depends on. Fixed.
2. **Construct-validity gap**: even after fixing (1), resolvable conflicts'
   ground truth was a coin flip independent of every signal visible in the
   data (source, confidence, ancestor value). This makes "resolvable"
   indistinguishable from "ambiguous" from any merger's point of view — no
   strategy could do better than chance, which would make H1 untestable as
   designed. Fixed by tying resolvable conflicts' ground truth to a
   source-reliability signal (one branch's update is `source="user"`/high
   confidence, the other `source="inference"`/low confidence), exposed in
   `ThreeWayLLMMerge`'s prompt.
3. **Ceiling effect**: with an always-reliable signal, `three_way_llm` hit a
   perfect 1.0 pilot accuracy — too easy to be an interesting or
   generalizable test. Added `source_signal_noise` (default 0.15): the
   higher-reliability branch is occasionally wrong anyway, so pattern-matching
   "always trust user source" doesn't get a free 100%.

The scenario generator and `ThreeWayLLMMerge` prompt are now locked. No further
changes before Phase 6 except documented bug fixes.

## 1. Locked scenario configuration for Phase 6

```
ScenarioConfig(
    n_entities=1, n_ancestor_facts=4,
    n_orthogonal_a=1, n_orthogonal_b=1,
    n_resolvable=2, n_ambiguous=1,
    timestamp_correlates_with_correctness=0.5,
    source_signal_noise=0.15,
)
```
varied across three `divergence_span` settings (short=4.0, medium=10.0,
long=20.0) to cover the "vary branch divergence duration" requirement, evenly
split across scenarios. Model: `gpt-5.4-nano` for all LLM-dependent components
(`ThreeWayLLMMerge`, `LLMJudgeConflictDetector`) — cheapest model available on
the provided key, per explicit cost-minimization instruction. `n_scenarios =
60` (see §5 for the power justification), seed=2026 (same as pilot, for
continuity — Phase 6 scenarios are NOT a superset of the pilot's; regenerating
with the same seed and n=60 produces a different, larger, independent set
since `ScenarioGenerator.generate` derives a fresh per-scenario sub-seed from
`(seed, index)`).

## 2. Hypotheses, tests, and falsification (verbatim intent from the build spec)

### H1 (RQ1): three-way merge vs. LWW and naive concat

- **Metric**: downstream accuracy (`evaluation/metrics.score_downstream`),
  per scenario, per strategy — a paired design (same scenario, all 5
  strategies: LWW, NaiveConcat, BranchDiscard-always_b, BranchDiscard-
  fewer_updates, ThreeWayLLM).
- **Test**: paired Wilcoxon signed-rank test (`scipy.stats.wilcoxon`) on
  per-scenario accuracy differences, for each of the pairs `three_way_llm vs.
  last_writer_wins` and `three_way_llm vs. naive_concat`. Wilcoxon chosen over
  a paired t-test because per-scenario accuracy is a bounded proportion over a
  small number of questions per scenario (3-6), not well-approximated as
  normal.
- **Effect size**: mean paired difference in accuracy, with 2000-resample
  paired bootstrap 95% CI (`evaluation/stats.py`, to be implemented in Phase
  6 alongside `run_experiment.py`).
- **Minimum effect size**: **0.10** (10 percentage points of downstream
  accuracy). Chosen from pilot data: observed `three_way_llm` − baseline
  differences were +0.30 (vs. LWW) and +0.33 (vs. NaiveConcat), with paired SD
  ≈ 0.19-0.20 — a large effect (Cohen's d for paired differences ≈ 1.5-1.6).
  0.10 is roughly a third of the observed pilot effect, giving comfortable
  power margin while still requiring a practically meaningful gap, not just
  statistical significance on a trivial difference.
- **Support**: both pairwise comparisons reject the null at the
  Holm-Bonferroni-corrected alpha (see §4) AND the bootstrap CI's lower bound
  exceeds the 0.10 minimum effect size.
- **Falsification**: naive concatenation is statistically indistinguishable
  from three-way merge (fails to reject the null, or the CI includes 0) — this
  would mean the merge machinery is unnecessary complexity, a legitimate,
  reportable negative finding per protocol.

### H2 (RQ3): three-way merge vs. branch-discard

- **Metric/test/effect size**: identical setup to H1, comparing `three_way_llm`
  vs. `branch_discard_always_b` and vs. `branch_discard_fewer_updates`.
  Pilot observed differences: +0.47 for both (branch-discard was the weakest
  strategy in the pilot, largely because it structurally loses ~50% of
  orthogonal information by construction). Same 0.10 minimum effect size,
  same correction.
- **Support**: three-way merge beats both branch-discard variants, corrected
  significant, CI lower bound > 0.10.
- **Falsification**: branch-discard performs equivalently or better — would
  mean the discarded branch's information wasn't actually load-bearing for
  the tasks tested (the benchmark needs harder scenarios where discarded info
  truly matters), or that discard is a good-enough pragmatic default.
  **Caveat pre-registered here**: because `BranchDiscard`'s orthogonal-loss
  penalty is a direct, near-mechanical consequence of the benchmark's own
  `orthogonal` question category (§ downstream_tasks.py always asks about
  every orthogonal addition), a support finding for H2 is expected almost by
  construction whenever orthogonal additions are present in a scenario. This
  is disclosed explicitly rather than treated as a surprising discovery in
  Phase 7's writeup.

### H3 (RQ2): cheap detectors vs. LLM-judge

- **Metric**: F1 (precision/recall computed from `evaluation/metrics.
  score_detector`), for `embedding_threshold` (threshold=0.80, calibrated in
  §3) and `nli` (threshold=0.20, calibrated in §3), each compared to
  `llm_judge` (the reference standard, not a competitor).
- **Tolerance**: **F1 within 0.05** of `llm_judge`, matching the build spec's
  own example value — used as-is, not tuned to either detector's pilot
  numbers. At calibrated thresholds on the pilot set: embedding F1=0.938,
  NLI F1=0.938 (both detectors hit the same ceiling — apparently bounded by
  the same handful of pairs neither cheap method catches), llm_judge
  F1=0.966 (at un-calibrated default settings reported in
  `pilot_summary.json`, since llm_judge has no threshold to calibrate). Gaps:
  embedding 0.028, NLI 0.028 — both well inside the 0.05 tolerance. Reported
  honestly; Phase 6 uses a larger, independent scenario set so this could
  still change, and would be a legitimate, reportable H3 outcome either way.
- **Support**: |F1_cheap − F1_llm_judge| ≤ 0.05 for a given detector.
- **Falsification**: the gap exceeds 0.05 — conflict detection genuinely
  requires LLM-level reasoning for that detector and cannot be cheaply
  approximated.
- Latency and estimated cost per detection are reported descriptively
  (no hypothesis test), since the spec asks for this as a secondary
  cost/benefit dimension, not a pass/fail criterion.

## 3. Detector threshold calibration (done, locked)

Procedure: swept `EmbeddingConflictDetector.threshold` over [0.30, 0.90] in
steps of 0.05, and `NLIConflictDetector.contradiction_threshold` over [0.10,
0.90] in steps of 0.05, against the 60 ground-truth conflict pairs in the
Phase 5 pilot scenario set (held out from Phase 6's scenario set, which uses a
larger, freshly-generated n=60). F1 plateaued at 0.938 for embedding at
threshold ≥ 0.80 (recall 0.883, precision 1.0 throughout the entire sweep —
neither detector ever produced a false positive on this pilot set, only
false negatives). NLI peaked at the same F1=0.938 for thresholds 0.10-0.25,
then declined (0.929 at 0.30-0.55, 0.919 at 0.60-0.75, 0.909 at 0.80-0.90).
Chose 0.80 for embedding (start of its plateau) and 0.20 for NLI (middle of
its higher-F1 plateau), for robustness to distributional shift in the Phase 6
set rather than an edge value. Locked in `configs/default.yaml`.

## 4. Multiple comparisons and reporting

- 5 merge strategies → 10 pairwise comparisons possible; only the 4 named in
  H1/H2 (three_way_llm vs. each of the other 4) are hypothesis-tested here —
  the remaining pairwise comparisons (e.g. LWW vs. NaiveConcat) are reported
  descriptively in `findings.md` but are not part of the preregistered
  confirmatory analysis, to avoid inflating the family of tests with
  comparisons that don't map to a stated hypothesis.
- Holm-Bonferroni correction applied across the 4 confirmatory tests (2 for
  H1, 2 for H2). Corrected alpha at each step: smallest p-value tested at
  0.05/4 = 0.0125, next at 0.05/3 ≈ 0.0167, next at 0.05/2 = 0.025, largest at
  0.05/1 = 0.05 (standard step-down Holm procedure).
- H3's two comparisons (embedding, NLI vs. llm_judge) use a fixed tolerance
  band, not a significance test, so no correction applies there.
- Every headline number reported with a 95% CI (paired bootstrap, 2000
  resamples, seeded for reproducibility) alongside the point estimate and the
  raw p-value.

## 5. Sample size for Phase 6

Target: detect a 0.10 minimum effect size at Holm-corrected alpha (≈0.0125
for the strictest of the 4 confirmatory tests) with 80% power, given the
pilot's paired SD ≈ 0.19-0.20 (Cohen's d ≈ 0.5). Using the standard paired
z-approximation n ≈ ((z_{α/2} + z_β)/d)², with z_{0.0125/2} ≈ 2.50 and
z_{0.80} ≈ 0.84: n ≈ ((2.50+0.84)/0.5)² ≈ 45. **n_scenarios = 60** is set for
Phase 6 (comfortable margin above the ~45 estimated minimum, still cheap on
`gpt-5.4-nano`: the pilot's 20 scenarios cost ~57K input / ~25K output tokens
across all LLM-dependent components combined, so 60 scenarios is estimated at
roughly 3x that — well within "minimize spend" — see PROGRESS.md for the
actual Phase 6 token count once run).

## 6. Non-determinism handling (locked from pilot observation)

The pilot's `three_way_llm_merge_variance` (5 scenarios × 3 independent,
uncached repeats) showed accuracy was NOT perfectly stable across repeats:
1 of 5 scenarios flipped on one repeat (0.8 → 1.0 → 1.0), the other 4 were
stable. `llm_judge` conflict-detection verdicts were stable across all 10
pairs tested. **Phase 6 will run `ThreeWayLLMMerge` and `LLMJudgeConflictDetector`
exactly once per scenario/pair** (not 3x majority-vote) for the main results,
because the pilot's observed flip rate (1/15 = 6.7%) is low enough that a
single run is a reasonable point estimate for a between-strategy comparison,
and 3x-ing every LLM call would triple the already-considered cost. This
single-run choice is disclosed as a limitation in `findings.md`: reported
`three_way_llm`/`llm_judge` numbers carry unquantified additional variance
from this non-determinism beyond the between-scenario variance captured by
the paired bootstrap CI.

## 7. What counts as a deviation

Any change to scenario generation parameters, merge/detector implementations,
or the statistical tests above, made after this document is committed, must
be logged in `PROGRESS.md`'s "Deviations from analysis plan" section with the
reason — including whether it could have affected the interpretation of any
result already computed. Bug fixes are allowed; result-shopping is not.
