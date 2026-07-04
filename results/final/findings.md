# BranchMem Phase 6 findings

This is a draft of the results section: what was found, mapped directly to
the preregistered hypotheses in `ANALYSIS_PLAN.md`, with effect sizes and
caveats stated plainly. Nothing here was decided after seeing the numbers —
the tests, thresholds, and minimum effect sizes were locked before this run.

**Setup**: n=60 synthetic branch-divergence scenarios (20 each at
divergence_span 4.0/10.0/20.0), `gpt-5.4-nano`, seed=2026. 180 ground-truth
conflict pairs. Full numbers in `results.json` / `stats_output.json`;
per-scenario raw data in `results.csv`.

## H1 (RQ1): does three-way LLM-mediated merge beat naive baselines? — **SUPPORTED**

| Comparison | Mean accuracy diff | 95% CI | Holm-adjusted p |
|---|---|---|---|
| three_way_llm vs. last_writer_wins | +0.219 | [0.183, 0.252] | ≈1.2×10⁻¹⁰ |
| three_way_llm vs. naive_concat | +0.236 | [0.205, 0.267] | ≈1.2×10⁻¹⁰ |

Both comparisons clear the preregistered 0.10 minimum effect size by a wide
margin (CI lower bound over 0.18 in both cases) and are significant after
Holm-Bonferroni correction. `three_way_llm` scored 0.936 mean downstream
accuracy overall vs. 0.717 (LWW) and 0.700 (naive concat).

Breaking down by question category (orthogonal / resolvable / ambiguous)
clarifies *why*: all four baselines matched `three_way_llm` on orthogonal
questions (1.000 across the board — trivial, since nothing conflicts there).
The gap is entirely in **resolvable** conflicts (three_way_llm 0.858 vs. LWW
0.508, naive_concat 0.450) and **ambiguous** conflicts (three_way_llm 0.833
vs. 0.000 for every baseline — mechanically guaranteed, since only
`ThreeWayLLMMerge` can produce a `flagged_unresolved` resolution at all; see
caveats). The resolvable-conflict gap is the real, non-mechanical finding:
`gpt-5.4-nano`, given the common ancestor plus each branch's source/
confidence metadata, correctly identified the more-reliable branch's update
most of the time; LWW's blind trust in wall-clock timestamp did little
better than the ~50% chance rate implied by the benchmark's
timestamp-uncorrelated-with-correctness design.

**Divergence duration had little effect**: `three_way_llm` accuracy was
0.914 / 0.957 / 0.936 at divergence_span 4.0 / 10.0 / 20.0 respectively — no
clear monotonic trend, suggesting (within the range tested) that longer
disconnection windows didn't make reconciliation harder for this method on
this benchmark. This wasn't a preregistered comparison and should be read as
descriptive, not confirmatory.

## H2 (RQ3): does three-way merge beat branch-discard? — **SUPPORTED**

| Comparison | Mean accuracy diff | 95% CI | Holm-adjusted p |
|---|---|---|---|
| three_way_llm vs. branch_discard_always_b | +0.493 | [0.460, 0.526] | ≈3.7×10⁻¹¹ |
| three_way_llm vs. branch_discard_fewer_updates | +0.493 | [0.460, 0.526] | ≈3.7×10⁻¹¹ |

The largest effect in the study (both discard policies scored identically —
the pilot's difference between "always_b" and "fewer_updates" policies
apparently doesn't matter much at this scale, since scenarios have one
entity and symmetric update counts per branch by construction). Branch-
discard's orthogonal accuracy (0.500) confirms the core prediction: on any
detected conflict, discarding a whole branch throws away real, useful
information (here, that branch's orthogonal additions), exactly the failure
mode H2 was designed to probe.

**Caveat, disclosed in advance in ANALYSIS_PLAN.md**: this benchmark's
`orthogonal` question category is present in essentially every scenario, and
branch-discard's loss of orthogonal information on conflict is a
near-mechanical consequence of the policy, not a subtle discovery. A harder,
more realistic test of H2 would vary how *load-bearing* the discarded
branch's information actually is for the downstream task (e.g., scenarios
where the discarded branch's updates matter much more, or much less, to the
questions asked) — the current benchmark doesn't yet vary this
independently of conflict presence.

## H3 (RQ2): can conflict detection be done cheaply? — **MIXED**

| Detector | F1 | Gap vs. llm_judge (0.986) | Within 0.05 tolerance? |
|---|---|---|---|
| embedding_threshold (t=0.80) | 0.957 | 0.029 | **Yes — supported** |
| nli (t=0.20) | 0.925 | 0.061 | **No — falsified** |

The embedding-similarity detector held up on the larger, independent Phase 6
scenario set. The NLI cross-encoder detector did not: its threshold was
calibrated on the pilot's 60 conflict pairs (where it also hit F1=0.938,
tied with embedding), but generalized worse to Phase 6's 180 pairs — its gap
grew from ≈0.028 in the pilot to 0.061 in the locked run, crossing the
preregistered tolerance. This is reported as the locked result, not
re-tuned: doing so after seeing Phase 6 data would defeat the point of
preregistration. The honest reading is that a single fixed threshold, picked
from a modest calibration set, is not obviously robust for this cross-
encoder model on this task — a larger or more diverse calibration set might
close the gap, but that's future work, not something this run can claim.

Both cheap detectors had perfect precision throughout the threshold sweep in
Phase 5 (never a false positive on the calibration set) — their errors are
exclusively missed conflicts (false negatives), not spurious ones. Latency:
embedding and NLI are both local-inference and roughly an order of magnitude
faster than `llm_judge` (Phase 5 pilot: ≈0.13-0.21s vs. ≈0.55-1.57s per
call), and cost nothing per call versus `llm_judge`'s API cost — the
practical latency/cost case for the cheap detectors stands even where H3's
strict F1-tolerance criterion doesn't fully hold.

## Cross-cutting limitations

- **Grader/merger overlap**: none. Downstream answers are scored against the
  scenario generator's ground truth (a Python equality check), never against
  a reference LLM's opinion — there is no circularity between the model
  doing the merging and the metric scoring it. `llm_judge`'s F1 IS scored
  against ground-truth conflict labels the same way.
- **Non-determinism**: `ThreeWayLLMMerge` and `LLMJudgeConflictDetector` were
  each called once per scenario/pair (not 3x-majority-vote), per the
  preregistered decision in ANALYSIS_PLAN.md sec.6 — justified by the
  pilot's observed 1/15 (6.7%) resample flip rate. The reported numbers
  therefore carry some unquantified additional variance beyond the
  between-scenario paired bootstrap CI. A 3x-repeated Phase 6 run was not
  done, for cost reasons (would have ~3x'd the already-modest spend).
- **Benchmark realism**: scenarios are synthetic, single-entity,
  template-generated from a fixed pool of ~14 predicates and a handful of
  values each (`data/scenario_templates.json`). The "resolvable" conflict
  signal (explicit `source="user"` vs. `source="inference"` metadata, with
  15% noise) is a deliberately clean, legible proxy for real-world source
  reliability; real agent memory systems may not have such a clean signal
  available. The MAB extension (`branchmem/benchmark/mab_extension.py`,
  validated against real MemoryAgentBench data in Phase 4 — see
  PROGRESS.md) is a secondary, more realistic-content validation path not
  included in this locked run; extending Phase 6 to include it is a natural
  next step (see below).
- **Single model**: all LLM-dependent results (`three_way_llm`, `llm_judge`)
  used one model (`gpt-5.4-nano`, the cheapest available, per an explicit
  cost-minimization instruction). Whether the H1/H2 effect sizes hold with a
  more capable (or a different-family) model is untested here.
- **H2's near-mechanical caveat** (repeated from above): the branch-discard
  comparison's effect size is partly an artifact of how the benchmark always
  includes orthogonal questions, not purely evidence about real-world
  information loss.

## Bottom line

H1 and H2 are both clearly supported with large, well-powered effect sizes —
three-way, ancestor-and-source-aware LLM merging meaningfully outperforms
naive last-writer-wins, naive concatenation, and branch-discard on this
benchmark's downstream task. H3 is a genuine mixed result: embedding-
similarity conflict detection is a viable cheap substitute for LLM-judge
detection on this benchmark; the NLI cross-encoder, at least with the
threshold calibrated here, is not — a legitimate negative finding for that
specific detector, not a benchmark or implementation failure.
