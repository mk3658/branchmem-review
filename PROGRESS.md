# BranchMem — Progress Log

Research artifact: asynchronous partition-and-merge for semantic agent memory.
See build prompt (conversation) for full spec. This file tracks phase gates,
current state, and open issues. Update after every phase.

## Repo / environment notes

- This project's git repo is **nested and independent**: `git init` was run
  inside `BranchMem/` specifically because the parent home directory
  (`/Users/quankienminh`) is itself an unrelated pre-existing git repo
  (contains an unrelated "CRA framework" project and many untracked
  sensitive-looking home-dir files). All git operations for this project are
  scoped to `BranchMem/` only — never run git commands from `~` for this work.
- Python 3.11+. Dependencies pinned in `requirements.txt`.

## Phase checklist

- [x] **Phase 0** — Repo scaffolding, PROGRESS.md, git init (nested, scoped to BranchMem/)
- [x] **Phase 1** — schemas, SQLite store, branch simulator, config/logging.
      Gate: fork memory state into two branches, apply scripted divergent
      updates, inspect result. `pytest tests/test_memory.py tests/test_branch_sim.py`
      — **9 passed**.
- [x] **Phase 2** — LLM backend abstraction (Anthropic / OpenAI-compatible / mock) + cache.
      Gate: real API call succeeds and is cached; repeated identical call hits cache.
      **Real-API half of this gate is now fully verified** (see "OpenAI key
      provided" note below) — an actual `gpt-5.4-nano` call succeeded and a
      repeated identical call hit cache. 7/7 new tests pass initially
      (`tests/test_llm.py`), later +2 more for the retry fix, 16/16 → 40/40
      total as of Phase 4+fix.
- [x] **Phase 3** — 3 conflict detectors + 4 merge strategies, unit tested against
      hand-built scenarios with known-correct answers.
      Gate: `pytest tests/test_conflict_detectors.py tests/test_merge_strategies.py`
      — **14 passed**. 30/30 total across the suite.
- [x] **Phase 4** — scenario generator + downstream task generator + benchmark
      construction (incl. MemoryAgentBench accessibility check).
      Gate: generate N=10 scenarios w/ ground truth, run all 4 merge strategies +
      3 detectors end-to-end on mock LLM backend. **8/8 new tests pass**, 38/38 total.
- [x] **Phase 5** — pilot run (`scripts/run_pilot.py`) on small real-LLM sample;
      write `ANALYSIS_PLAN.md` (tests, corrections, falsification criteria) based
      on pilot variance. Gate: ANALYSIS_PLAN.md committed before Phase 6 begins. **Done.**
- [x] **Phase 6** — locked full experiment run (`scripts/run_experiment.py`),
      no method/benchmark changes after this except documented bug fixes.
      Gate: `results/final/` contains results.csv, results.json, stats output
      matching every test in ANALYSIS_PLAN.md. **Done.**
- [x] **Phase 7** — `results/final/findings.md`: plain-language summary mapping
      each result to H1/H2/H3 (supported/falsified), effect sizes, caveats. **Done.**

## Current state

Phase 1 complete: `branchmem/memory/{schemas,store,branch_sim}.py`,
`branchmem/utils/{config,logging,seeding}.py`, `configs/{default,experiment}.yaml`.
`BranchSimulator.diverge()` forks a common ancestor into two branches, applies
independently scripted `UpdateSpec` streams (orthogonal / resolvable_conflict /
ambiguous_conflict, with ground truth attached by construction), and emits
ground-truth `ConflictPair`s. Validates that both branches' authors agree on
divergence_type/ground_truth_value for shared keys (raises `DivergenceError`
otherwise). History is kept per branch (superseded facts aren't deleted),
`common_ancestor_id` chains link each fact to what it updates.

Phase 2 complete: `branchmem/llm/{base,cache,mock_backend,anthropic_backend,
openai_compatible_backend}.py`. `LLMBackend.complete()` is the single caching
wrapper — content-hash key over (backend, model, system, prompt, temperature,
max_tokens), stores full prompt+response JSON per call for auditability.
`build_backend()` factory reads `llm.backend` from config. Anthropic/OpenAI
backends fail fast with a clear RuntimeError if their API key env var is
missing, rather than silently falling back.

Phase 3 complete. Conflict detectors (`branchmem/conflict/`):
`EmbeddingConflictDetector` (cosine similarity of value strings, threshold —
placeholder 0.55, real calibration deferred to Phase 5 pilot), `NLIConflictDetector`
(contradiction-probability cross-encoder), `LLMJudgeConflictDetector` (reference
standard for RQ2, not a competitor). All three accept an injectable
embed_fn/predict_fn so unit tests don't require downloading real models or
network access — real model loading (sentence-transformers) is lazy and only
triggered when no fn is injected.

Merge strategies (`branchmem/merge/`): `LastWriterWins` (timestamp-only, no
branch-awareness), `NaiveConcat` (keeps every current fact from both branches,
duplicates included), `BranchDiscard` (policy="always_b" or "fewer_updates";
discards a branch's entire post-fork update set on ANY key collision, but only
reports genuinely-lost facts as DROPPED — facts identical to the kept branch's
value aren't double-counted as loss), `ThreeWayLLMMerge` (gives the LLM the
common ancestor + both branches' conflicting values in one batched call per
merge for cost control; non-conflicting keys resolved deterministically without
an LLM call; unresolvable conflicts are flagged, not guessed).

Note: merge strategies detect "candidate conflicts" via structural key
collision (same entity+predicate, differing current value) — not via the
Phase 3 conflict detectors. The detectors are for RQ2 (can conflict
*detection* be done cheaply); merge strategies for RQ1/RQ3 assume conflicts
are already structurally identifiable, which is realistic for this benchmark's
canonicalized (entity, predicate, value) fact representation.

Phase 4 complete.

**scenario_generator.py**: `ScenarioGenerator.generate(n, config, seed)` builds
N fully-deterministic synthetic scenarios (given a seed) from
`data/scenario_templates.json` (10 entities, 8 "objective" predicates with a
fact-of-the-matter, 6 "subjective" predicates with none). Each scenario has a
configurable mix of orthogonal additions, resolvable conflicts (ground truth
assigned independently of timestamp — `timestamp_correlates_with_correctness`
defaults to 0.5, preregistered here so LWW isn't trivially right or trivially
wrong by construction), and ambiguous conflicts (no ground truth; correct
behavior is to flag, not guess). `divergence_span` controls short-vs-long
disconnection.

**downstream_tasks.py**: `generate_downstream_questions(scenario)` builds one
QA-style question per conflict pair and per orthogonal addition. Ambiguous
questions have `expected_answer=None`; scored correct only if the merge
strategy explicitly flagged that key unresolved (evaluation/metrics.py).

**evaluation/metrics.py**: `score_downstream` (accuracy overall + by category)
and `score_detector` (precision/recall/F1/latency) — enough to exercise the
Phase 4 gate; will be extended with the preregistered paired significance
tests in Phase 5.

**mab_extension.py — MemoryAgentBench extension**: confirmed the dataset is
real and accessible (see Known Issues above). Built a 38-template regex
parser (`parse_context_facts`) for MAB's flattened natural-language context
strings — no structured (subject, relation, object) field exists in the
public schema, so this was necessary rather than optional. **Measured
coverage: 99.9% of 18,337 unique context lines** parsed successfully across
an 8-row live sample.

Key finding from real-data validation (not anticipated at design time): MAB's
edit chains are almost always exactly ONE counterfactual edit deep (ancestor +
one MQuAKE replacement) — essentially no chains reach 2+ edits in the sampled
rows. A single edit can only land on one branch, so an early version of
`build_branch_scenario_from_mab_row` that only created conflicts from >=2-edit
chains found **zero conflicts across all 8 real rows** (caught by manually
running the pipeline against the live dataset, not by the offline unit
tests — the offline tests used hand-built 3-edit chains that don't reflect
real chain-length distribution). Fixed by having the common single-edit case
reuse MAB's real ancestor value + real edit for one branch, and constructing
the other branch's competing value by resampling a real value that occurs
elsewhere in the same row for the same predicate (documented in the module
docstring as NOT literal MAB reuse for that half, and logged per-scenario via
`metadata["n_synthetic_second_branch_edits"]`).

A second bug surfaced during that same validation run: the initial fix always
assigned the synthetic branch a *later* timestamp than the real edit, which
silently rigged LastWriterWins to lose 100% of the time (measured: LWW_acc=0.00
across all 8 real rows). Fixed by randomizing both branch assignment and
timestamp ordering (same "correctness independent of timestamp" principle as
`scenario_generator.py`). Re-validated on the same 8 live rows post-fix:
LWW accuracy now clusters around 49-56% (as expected for an unbiased
construction), 17,816 total conflict pairs generated across the 8 rows. This
validation was exploratory/manual (not part of the automated test suite,
since it requires network access), but is the reason the module docstring
and this log describe the real, validated behavior rather than the
originally-intended design.

**Limitation carried into Phase 5/6**: MAB's Fact Consolidation task is
deterministic by construction (always has a single correct final value), so
`mab_extension.py` produces resolvable_conflict and orthogonal divergences
only — no ambiguous_conflict cases. H1's ambiguous/flagging-behavior test is
covered by `scenario_generator.py` alone. The **primary preregistered
experiment (Phase 5/6) will run on `scenario_generator.py`**, where we have
full control over the resolvable/ambiguous ratio and divergence duration
required by the analysis plan; `mab_extension.py` serves as a secondary,
real-data validation/robustness check, reported separately in findings.md.

## OpenAI key provided (2026-07-03)

User provided a real OpenAI API key. Stored ONLY in `.env` (gitignored,
never committed) and read via the existing `OPENAI_API_KEY` env var
mechanism already built into `OpenAICompatibleBackend` — no code or config
changes needed for key handling itself. User asked to minimize spend, so
Phase 5's pilot (and Phase 6, pending further discussion) will use
`gpt-5.4-nano`, the cheapest current-generation model available on this key
(confirmed via `GET /v1/models`).

**Bug found and fixed while wiring this up**: newer OpenAI models
(o1/o3/gpt-5.x reasoning family) reject the `max_tokens` parameter and
require `max_completion_tokens` instead; some also reject a non-default
`temperature`. `OpenAICompatibleBackend._call()` now retries once with the
offending parameter patched (renamed or dropped) based on the API's own
error message, rather than hardcoding per-model quirks that will keep
changing. Verified with a real call to `gpt-5.4-nano` (succeeded; repeat
call hit cache) and locked in with two new unit tests using a mocked client
(`test_openai_backend_retries_with_max_completion_tokens`,
`test_openai_backend_retries_by_dropping_temperature`). 40/40 tests passing.

**`llm_cache/` is now tracked in git** (removed from `.gitignore`) rather
than ignored — per the project's auditability requirement, every real LLM
call's full prompt+response should be a committed, reproducible artifact,
not a local-only cache. The one throwaway manual connectivity-test cache
entry was deleted before this decision took effect; Phase 5/6's real cache
entries will be committed as generated.

## Phase 5: pilot run and preregistration (2026-07-03)

Ran `scripts/run_pilot.py` (N=20 synthetic scenarios, `gpt-5.4-nano`) three
times, iterating the scenario generator each time in response to real
findings — this is expected/appropriate pilot behavior (the plan wasn't
locked yet), documented here in full for transparency:

**Pilot run 1** (buggy): `three_way_llm` scored 0.59 overall, and only 0.075
on resolvable conflicts — much worse than `last_writer_wins` (0.625 on
resolvable). Inspecting the actual cached LLM prompts/responses showed why:
every "Common ancestor value" block sent to the model was `{}` — completely
empty — even for conflicts on keys that genuinely existed in the ancestor.
Root cause: `scenario_generator.py`'s `_generate_one` seeded `used_keys =
set(ancestor_keys)` BEFORE picking any conflicts, which made
`_sample_conflict_key`'s "prefer reusing an existing ancestor key" branch
permanently unreachable (`k not in used` was always False for every ancestor
key). So `ThreeWayLLMMerge` — whose entire design premise is using ancestor
context — never once received real ancestor context in the pilot. This was
NOT caught by the Phase 4 offline unit tests, because those tests hand-built
scenarios with explicit ancestor context already wired up; they exercised
`BranchSimulator`'s correctness given valid inputs, not whether the
*generator* was producing valid inputs. Fixed: ancestor-key reuse now tracked
via a separate `consumed_ancestor_keys` set, independent of the `used_keys`
set that (correctly) still blocks orthogonal additions from reusing ancestor
keys. Verified: ancestor context coverage went from 0/60 to 53/60 (88%)
conflicting keys.

**Pilot run 2** (construct-validity gap, not a bug): with ancestor context
now present, `three_way_llm` still scored only 0.1 on resolvable conflicts.
Inspecting the responses showed the model wasn't confused — it correctly
reported having no principled basis to choose between the two branches, and
mostly flagged conflicts unresolved (scored wrong under our rule that
resolvable conflicts must be confidently and correctly resolved, not
flagged). The actual problem: resolvable conflicts' ground truth was decided
by an internal coin flip with NO signal exposed anywhere in the visible data
(not in ancestor, not in the two branches' values) — making "resolvable"
literally indistinguishable from "ambiguous" to any merger, human or LLM.
This would have made H1 untestable as designed: no strategy could beat chance
on a category with no learnable signal. Fixed: resolvable conflicts now tie
ground truth to a source-reliability signal (`MemoryFact.source`/
`confidence` — real schema fields, not new ones); one branch's update is
`source="user"` (high confidence), the other `source="inference"` (low
confidence), ground truth = the user-sourced branch. `ThreeWayLLMMerge`'s
prompt and system message were updated to expose source/confidence (they
previously stripped everything but the bare value) and explicitly told NOT
to use timestamp as a signal (branches have no shared clock). Result:
`three_way_llm` jumped to 1.0 overall, 1.0/1.0/1.0 across all three
categories, with zero variance across repeat calls.

**Pilot run 3** (ceiling effect, not a bug): a perfect, zero-variance 1.0 is
too easy to be an interesting or generalizable test, and risks the real
Phase 6 experiment being an uninformative rubber stamp. Added
`source_signal_noise=0.15`: the higher-reliability branch is occasionally
(15% of the time) wrong anyway, so a strategy can't get a free 100% by
blindly pattern-matching "trust the user-sourced value." Final pilot
result: `three_way_llm`=0.90 overall (orthogonal=1.0, resolvable=0.825,
ambiguous=0.85), `last_writer_wins`=0.60, `naive_concat`=0.57,
`branch_discard`(both policies)=0.43. This is the locked scenario design —
see `ANALYSIS_PLAN.md` sec.0 for the full summary and sec.1 for the exact
locked `ScenarioConfig`.

**Detector threshold calibration**: swept `EmbeddingConflictDetector`
threshold and `NLIConflictDetector` contradiction_threshold against the
pilot's 60 ground-truth conflict pairs (precision was 1.0 across the entire
sweep for both — zero false positives, only false negatives at low
thresholds). Locked: embedding threshold=0.80 (F1=0.938), NLI
threshold=0.20 (F1=0.938, same ceiling as embedding). Both within 0.05 F1 of
`llm_judge` (F1=0.966) — H3's tolerance band. Full procedure and numbers in
`ANALYSIS_PLAN.md` sec.3.

**Non-determinism**: resampled `ThreeWayLLMMerge` 3x per scenario
(use_cache=False, so genuinely independent calls, not cache hits) on 5
scenarios: 1/5 flipped on one repeat (0.8→1.0→1.0), 4/5 were stable.
`LLMJudgeConflictDetector` resampled on 10 pairs: 0/10 flipped. Locked
decision (ANALYSIS_PLAN.md sec.6): Phase 6 runs each LLM-dependent call once
per scenario/pair (not majority-vote), disclosed as a limitation.

**Bug fixed en route**: newer OpenAI models (o1/o3/gpt-5.x) reject
`max_tokens`/require `max_completion_tokens`, and some reject non-default
`temperature`. `OpenAICompatibleBackend._call()` now retries with the
offending param patched (and remembers the fix per-instance after the first
call, so later calls in a Phase 6 run of ~1000s of calls don't each pay a
wasted round-trip). See the "Fix OpenAI backend..." commit.

`ANALYSIS_PLAN.md` written and committed: preregisters the exact statistical
tests (paired Wilcoxon signed-rank per H1/H2 comparison, Holm-Bonferroni
correction across 4 confirmatory tests, 2000-resample paired bootstrap CIs),
minimum effect size (0.10 accuracy, informed by pilot SD ≈0.19-0.20 and
observed effect ≈0.30-0.47), F1 tolerance (0.05, per the build spec's own
example, applied without tuning to either detector's pilot number), and
Phase 6 sample size (n=60, power-justified against the 0.10 minimum effect).
`configs/experiment.yaml` updated to match (gpt-5.4-nano, n=60, seed=2026,
3 divergence spans).

Starting Phase 6 (locked full experiment run) next.

## Known Issues / Open Questions

- MemoryAgentBench accessibility: **confirmed accessible.** MIT-licensed code
  at github.com/HUST-AI-HYZ/MemoryAgentBench, data at
  huggingface.co/datasets/ai-hyz/MemoryAgentBench (arXiv:2507.05257, ICLR
  2026). Its Conflict_Resolution split's Fact Consolidation task is a flat,
  single-timeline numbered fact list with MQuAKE-style counterfactual edits
  (later restatement of the same (subject, relation) supersedes the earlier
  one) — exactly the "freshness" framing our project distinguishes itself
  from, which makes it good raw material to extend with branching. Full
  details and design decisions logged in `branchmem/benchmark/mab_extension.py`'s
  module docstring; summary below under "MAB extension" in Current State.
- Need to confirm which API keys are available in this environment
  (ANTHROPIC_API_KEY / OPENAI_API_KEY) before Phase 2's real-call gate and
  before Phase 5's pilot run. Everything through Phase 4 can run on the mock
  backend without keys.
- Minimum effect size for H1, and the F1 tolerance band for H3, are only
  determined in Phase 5 from pilot variance — not decided yet, per protocol
  (must not be picked post hoc after seeing full results).

## Phase 6: locked experiment run (2026-07-03)

Ran `scripts/run_experiment.py` exactly as preregistered in
`ANALYSIS_PLAN.md` — no parameter changes. n=60 scenarios (20 per
divergence_span in [4.0, 10.0, 20.0]), `gpt-5.4-nano`, seed=2026, calibrated
detector thresholds (embedding=0.80, NLI=0.20). 180 conflict pairs total.
Wrote `results/final/{results.csv,results.json,stats_output.json,
accuracy_by_strategy.png}`. Added `branchmem/evaluation/{stats,runner,
plots}.py` and `tests/test_stats.py`/`test_runner.py` beforehand (47/47
tests passing before the real run); smoke-tested the full pipeline with
MockBackend first to catch bugs before spending on the real n=60 call.

**Results** (full numbers in `results/final/stats_output.json`):

- **H1 SUPPORTED**: `three_way_llm` beats `last_writer_wins` by +0.219
  accuracy (95% CI [0.183, 0.252], Holm-adjusted p≈1.2e-10) and beats
  `naive_concat` by +0.236 (CI [0.205, 0.267], adjusted p≈1.2e-10). Both
  comfortably clear the preregistered 0.10 minimum effect size.
- **H2 SUPPORTED**: `three_way_llm` beats both `branch_discard` policies by
  +0.493 (CI [0.460, 0.526], adjusted p≈3.7e-11) — the largest effect in the
  study, consistent with the pre-disclosed expectation in ANALYSIS_PLAN.md
  sec.2 that this follows near-mechanically from BranchDiscard's structural
  loss of orthogonal information.
- **H3 MIXED**: `embedding_threshold` F1=0.957 vs. `llm_judge` F1=0.986 — gap
  0.029, within the 0.05 tolerance, **supported**. `nli` F1=0.925 vs. 0.986 —
  gap 0.061, **exceeds** tolerance, **falsified** for this detector. Notably
  different from the pilot, where both detectors were well inside tolerance
  (pilot gaps ≈0.028 for both) — the NLI threshold was calibrated on the
  20-scenario pilot set and didn't fully generalize to the independently-
  generated 60-scenario Phase 6 set. This is disclosed as-is, not re-tuned
  post hoc (re-tuning after seeing Phase 6 results would violate the
  preregistration).

Token usage: 460 total cached LLM calls across pilot+Phase 6 combined
(111,403 input / 45,462 output tokens cumulative, all in `llm_cache/`,
committed). No dollar cost computed (see pilot's note on why) but this is
solidly inexpensive on nano-tier pricing for ~460 short completions.

## Deviations from analysis plan (Phase 6+)

(none — Phase 6 ran exactly as preregistered in ANALYSIS_PLAN.md, no
parameter changes after the plan was committed)

## Phase 7: findings write-up (2026-07-03)

Wrote `results/final/findings.md` (H1/H2/H3 mapped to supported/falsified
with effect sizes, CIs, and disclosed caveats) and `README.md` (install/run
commands, module map, results summary, known limitations). All 8 phases
complete. Project is in a state where `pytest tests/ -q` passes (47/47),
`results/final/` contains the full locked-run artifact set, and every real
LLM call across the whole project (pilot + Phase 6) is committed in
`llm_cache/` for reproducibility.

**Suggested next steps** (not undertaken here — out of scope for the
preregistered Phase 6 run): (1) run Phase 6 again with the MemoryAgentBench
extension included as a second scenario source, now that its construction is
validated; (2) confirm H1/H2 effect sizes with a second, more capable model,
since all LLM-dependent results here used the cheapest available model per
an explicit cost-minimization instruction; (3) run `ThreeWayLLMMerge` and
`LLMJudgeConflictDetector` with majority-vote across 3 repeats to get a
tighter, non-determinism-aware estimate, now that the pilot's low flip rate
has been established as a baseline; (4) investigate why the NLI detector's
calibrated threshold didn't generalize from the 20-scenario pilot set to the
60-scenario Phase 6 set (H3), e.g. with a larger calibration set.

## Post-Phase-7: paper writing and peer-review-driven secondary analyses (2026-07-03)

At the user's request, wrote an ACL-format paper from this project (writer
agent) and had it reviewed by a second agent simulating ACL reviewers. The
reviewer verified every number in the paper against `stats_output.json`,
`ANALYSIS_PLAN.md`, and `PROGRESS.md` directly (found zero discrepancies —
the paper accurately reports what was actually done) and scored draft v1
**5/10**, with the two highest-impact fixes for reaching 8/10 both requiring
new experiments: (1) a second model for RQ1/RQ3 robustness, (2) actually
running the already-built MAB extension as a real (secondary/exploratory,
not preregistered) analysis. User approved doing both. These are
**post-hoc, exploratory, NOT part of ANALYSIS_PLAN.md's confirmatory
tests** — logged here per the "deviations must be logged" rule, though
strictly these are new supplementary analyses rather than deviations from
the original locked plan.

**Second-model robustness check** (`results/final/robustness_second_model.json`):
regenerated the same 20 scenarios used in Phase 6's divergence_span=4.0
group (seed=2026) and ran `ThreeWayLLMMerge` with `gpt-5.4-mini` instead of
`gpt-5.4-nano` (baselines are model-independent, reused from Phase 6, not
rerun). Result: gpt-5.4-mini scored 0.929 mean accuracy vs. gpt-5.4-nano's
0.914 on the identical 20 scenarios — paired difference +0.014, 95% CI
[-0.014, 0.043], p=0.45 (not significant). The direction and magnitude of
the RQ1 effect hold with a more capable model in the same provider family;
this is a real but limited robustness result (still OpenAI-only, no
cross-family model tested, since only an OpenAI key is available in this
environment).

**MAB extension secondary analysis** (`results/final/mab_secondary_analysis.json`)
— **an important negative finding, reported as found, not adjusted**: added
a `max_conflict_keys` parameter to `build_branch_scenario_from_mab_row` (real
MAB rows have up to 7221 edit chains, too many for one batched
`ThreeWayLLMMerge` prompt) and ran all 4 merge strategies + `ThreeWayLLMMerge`
(`gpt-5.4-nano`) on all 8 real MemoryAgentBench Conflict_Resolution rows,
capped at 15 conflicts/row. Result: **`three_way_llm` scored a mean of 0.056
accuracy — far WORSE than every naive baseline** (last_writer_wins=0.591,
naive_concat=0.528, branch_discard=0.472). Root cause, confirmed by
inspecting the actual cached LLM responses: `mab_extension.py` never sets
differentiated `source`/`confidence` on its `UpdateSpec`s (both branches
default to `source="observation", confidence=1.0`), so — unlike the
synthetic generator's `source="user"` vs. `source="inference"` construction
— there is NO exploitable signal in MAB-derived scenarios. `ThreeWayLLMMerge`
correctly recognizes this (its system prompt explicitly tells it to flag
rather than guess when there's no real basis to prefer a branch) and flags
nearly every conflict unresolved, which our strict scoring counts as wrong
for a category whose ground truth requires a confident, correct answer.
Naive baselines score ~50-60% simply because they always commit to *a*
answer (never abstain), which beats a strategy that correctly declines to
guess under this scoring rule. **This is not a bug and was not adjusted or
re-run with a different construction to get a better number** — it directly
and honestly demonstrates that RQ1's headline win is highly contingent on
the presence of an explicit, legible reliability signal, which is exactly
the kind of finding the peer review's "the LLM wins mostly because the
benchmark is built so it can" concern predicted. Reported to the writer
agent for inclusion in the paper as a major, prominent finding — not buried
or softened.

## Round 2 of paper revisions: real ARR-calibrated review response (2026-07-04)

User supplied a second, independently-produced external review calibrated
against the real ACL/ARR + EACL 2027 review form (1-5 scale: 3.5/5,
"Borderline Conference; strong Findings accept"). Its diagnosis matched our
own reviewer's: the method's positive result is signal-contingent, the
real-data extension is small, no human validation exists. User approved
doing the two spend-requiring items (new baselines, expanded MAB) in full.

**New baseline: `ConfidenceRuleMerge`** (`branchmem/merge/confidence_rule.py`)
— fully deterministic, no LLM: on a conflict, keep the higher-`confidence`
branch; exact tie -> flag. Isolates whether `ThreeWayLLMMerge`'s LLM call is
adding anything beyond reading the confidence field. Run on the same n=60
locked scenarios: **0.962 mean accuracy, SIGNIFICANTLY BEATS
`ThreeWayLLMMerge`'s 0.936** (paired diff -0.026, 95% CI [-0.040, -0.014],
p=0.0021). This is an important, humbling finding: on this benchmark, a
one-line deterministic rule matches or exceeds the LLM. Reasons: the
generator's `source_signal_noise=0.15` construction means "always trust
higher confidence" is correct ~85% of the time by design, same as what the
LLM achieves; and ambiguous conflicts have exactly-tied confidence by
construction, so the deterministic tie-check flags them just as correctly as
the LLM's judgment does, with zero LLM-judgment noise.

**New baseline: `TwoWayLLMMerge`** (`branchmem/merge/two_way_llm.py`) — same
LLM-based approach as `ThreeWayLLMMerge` but the prompt omits the common
ancestor entirely (only branch A/B value+source+confidence). Isolates
whether ancestor context matters. Run on the same n=60 scenarios: **0.926
mean accuracy, NOT significantly different from `ThreeWayLLMMerge`'s 0.936**
(paired diff +0.0095, 95% CI [-0.0095, 0.0286], p=0.29). The ancestor
context's added value is not statistically distinguishable from zero on this
benchmark's current construction — another honest, humbling finding.

Both added to `tests/test_merge_strategies.py` (4 new tests), 52/52 total
passing. Results in `results/final/new_baselines.json`.

**Expanded MAB secondary analysis** (`results/final/mab_secondary_analysis_expanded.json`):
confirmed via the HuggingFace parquet API that MemoryAgentBench's
Conflict_Resolution split has exactly 8 rows total — this is the FULL real
dataset for this task, not a cost-driven subsample, so "more rows" isn't
available without fabricating data. Instead raised `max_conflict_keys` from
15 to 50 (same 8 real rows, ~3.4x more real conflict pairs evaluated: 230 vs.
67). **Hit a real bug on the first attempt**: `max_tokens=2048` was
insufficient for the larger per-row response (up to 50 verbose JSON
resolutions), causing silent truncation that forced spurious
`flagged_unresolved` fallbacks unrelated to genuine model judgment — caught
by checking `output_tokens` against the response's JSON parseability, not by
trusting the aggregate number. Fixed with `max_tokens=8000`; verified zero
truncated/unparseable responses in the corrected run. The 8 corrupted cache
entries (from both this and a second, unrelated repeat of the same mistake
in a follow-up script) were deleted, not committed.

Corrected result: `three_way_llm` scores essentially 0 (1/230 correct;
mean 0.004) on the expanded real sample — confirms the original finding at
~3.4x the sample size, not a fluke of a small n=8. `confidence_rule` scores
exactly 0.0 on MAB data too (since `mab_extension.py` never differentiates
confidence — every MAB conflict is an exact tie, so the deterministic rule
flags all of them, same mechanism as the LLM's collapse, confirming the root
cause is the missing signal in the data, not something specific to LLM
judgment).

**Exploratory "conditional-on-commit" metric** (`results/final/abstention_adjusted_metric.json`,
computed entirely from already-cached calls, no new spend beyond what's
above): separates "how often does the model even attempt an answer" from
"how often is it right when it does." Synthetic resolvable conflicts:
committed 119/120 times (99.2%), correct 0.857 of those. **MAB expanded:
committed 0/230 times — the model NEVER guessed, only ever flagged.**
Conditional-on-commit accuracy is undefined because there were no commits to
condition on. This is the cleanest possible confirmation that the collapse
is 100% calibrated abstention, not degraded reasoning when the model does
commit.

**Bug fixed in `branchmem/merge/base.py`**: `find_collisions` iterated
`set(a_facts) & set(b_facts)` in raw set order, which Python randomizes per
process by default (string hash seed). This made LLM-prompt JSON key
ordering non-deterministic across separate process runs for the
*semantically identical* scenario, silently defeating the content-hash cache
and causing avoidable re-spend (caught when a supposedly-cache-hit rerun of
the conditional-on-commit script produced fresh `cache MISS` calls and, in
one case, a config mismatch during debugging that repeated the truncation
bug at `max_tokens=1024`). Fixed by sorting keys before use. All 52 tests
still pass.

Next: integrate all of this into the paper (new results section for the two
baselines, updated/expanded MAB section, the conditional-on-commit metric,
a deployment decision-matrix table per the review's suggestion #2, and an
abstract/intro reframing toward "benchmark + boundary condition" per
suggestion #1) plus a double-blind anonymization check (suggestion #7).

## Paper v5: full integration of round-2 review response (2026-07-04)

Integrated all of the above into `paper/acl_latex.tex` directly (writer
agent was not used for this round — the previous writer agent hit its
session limit mid-task on the prior round, so all LaTeX editing, figure
regeneration, and page-fitting for this round was done directly):

- New `\subsection{Ablations: is the LLM doing anything?}` (Section 5.2)
  with Table 3 reporting `ConfidenceRuleMerge` (0.962) beating
  `ThreeWayLLMMerge` (0.936, p=0.002) and `TwoWayLLMMerge` (0.926, not
  significant, p=0.29) — both framed honestly as narrowing, not retracting,
  RQ1's result.
- Section 5.6 (MAB negative result) rewritten around the expanded
  230-conflict sample (mean 0.004, not 0.056) as the primary reported
  number, with the "zero commits across 230 real conflicts" finding as the
  section's strongest evidence. Figure 3 and Table 6 (Appendix D)
  regenerated from `mab_secondary_analysis_expanded.json` to match —
  caught and fixed several now-stale hardcoded numbers (old 0.056 in two
  Discussion/Limitations paragraphs and one figure caption) that a
  restructuring pass would have silently left inconsistent with the
  updated headline number had they not been individually greped for.
- New Table 8 (Appendix G): the reviewer's suggested deployment decision
  matrix, moved to the appendix (not body) purely for page-budget reasons,
  referenced prominently from Section 5.6.
- Abstract and Introduction's contributions list rewritten to lead with the
  boundary-condition/benchmark framing and the ablation finding, per
  reviewer suggestion #1, rather than "three-way merge works."
- Anonymization pass: found and generalized two phrases ("the provided
  key", "no Anthropic key was available in this environment") that read as
  revealing something about the specific non-standard dev environment
  rather than a standard academic API subscription; no other identity
  leaks found (no first-person singular, no personal names/emails/repo
  URLs beyond legitimate citations).
- Item 5 (human validation of ground truth) explicitly declined as
  something that cannot be produced honestly in this project (no real
  annotators available) — the Limitations section already disclosed this
  gap and explicitly states an LLM-based self-consistency check was NOT
  substituted for genuine human validation, since presenting one as the
  other would mislead.

**Page-budget management**: adding ~150 lines of new body content pushed
the numbered body (Introduction through Conclusion) to 10 pages, 2 over the
ACL limit. Fixed by: moving the decision-matrix table to the appendix,
compressing the Ablations section prose by roughly half without cutting any
numbers or the two headline findings, compressing the expanded-MAB
prose similarly, and trimming the Introduction's contributions list and
Conclusion by a few lines each. Final state: body ends cleanly on page 9
alongside Limitations' start (verified via `pdftotext -layout` boundary
check, not just total page count), zero Overfull-hbox warnings, 13 total
pages (PDF metadata `kMDItemNumberOfPages`, not the pdftotext form-feed
split which had a spurious trailing empty page).

Two real bugs caught and fixed in this round, both logged in detail above:
(1) an insufficient `max_tokens` budget on the first expanded-MAB attempt
silently truncated ~50-key JSON responses, forcing spurious
`flagged_unresolved` fallbacks unrelated to genuine model judgment — caught
by checking output-token counts against JSON parseability, not by trusting
the aggregate number; (2) `find_collisions` iterated a raw Python `set` in
hash-randomized (non-deterministic-per-process) order, silently defeating
the LLM prompt cache across separate script invocations and causing
avoidable re-spend — fixed by sorting.

All 52 tests still pass. Full new artifacts: `results/final/new_baselines.json`,
`results/final/mab_secondary_analysis_expanded.json`,
`results/final/abstention_adjusted_metric.json`,
`branchmem/merge/{confidence_rule,two_way_llm}.py`.

Not yet done: a fresh simulated-reviewer pass on this v5 (the prior 8/10
score was on v4, before this round's substantial new content). Deferred
pending user direction on whether to spend the additional review round.

## Round 3: ACL 2027 reviewer pass + fixes (2026-07-04)

Spawned one agent to review the paper as a fair ACL 2027 / ACL Rolling
Review reviewer (real web search for current ARR guidelines/score scale,
since no ACL-2027-specific CFP exists yet as of this date; full review
saved at `paper/reviews/acl2027_review.md`; scores Soundness 3.5/5, Overall
3.5/5, Confidence 4/5, Reproducibility 5/5 on the ARR 1-5 scale). Spawned a
second agent to fix the review's top action items, with three items scoped
by explicit user decision before spending: (5) new metadata-free
`RawTextLLMMerge` ablation — approved, run; (8) resolvable-category
statistical-power expansion — approved, run as an explicitly post-hoc
replication (new seed 2029, 120 new resolvable pairs); (1) cross-model-family
check and (2) independent human ground-truth validation — declined (no
non-OpenAI key, no human annotator available in this environment), left as
honestly-stated open Limitations.

The fixer agent was killed by the user mid-task (after completing the real
experiments and LaTeX integration, but before final page-limit verification
and this PROGRESS.md update). Rather than re-run the real API spend, the
surviving work was verified in place (55/55 tests passing, both new result
files present and internally consistent) and the remaining page-overflow
issue was finished directly: after the new content (Table 3's
`RawTextLLMMerge` row, the new post-hoc power-expansion subsection, the RQ2
F1-drift bootstrap paragraph, the parser-precision self-audit paragraph)
pushed the Conclusion from page 8 to page 9, roughly a dozen rounds of
prose tightening across Related Work, Introduction, Method, Experimental
Setup, Results (RQ1/RQ2), Discussion, and Conclusion (no numbers or claims
cut, only wordiness), plus a global `\arraystretch{0.92}`/reduced float-margin
tweak and a ~13% figure-panel height reduction, pulled the Conclusion back
onto page 8 with zero Overfull/error warnings. New artifacts:
`branchmem/merge/raw_text_llm.py`, `results/final/raw_text_ablation.json`,
`results/final/resolvable_power_expansion.json`,
`results/final/parser_precision_self_audit.json`,
`results/final/rq2_f1_bootstrap.json`, `scripts/run_raw_text_ablation.py`,
`scripts/run_resolvable_power_expansion.py`, `scripts/audit_parser_precision.py`,
`scripts/analyze_rq2_f1_drift.py`.

## Round 4: reframing around provenance + abstention boundary conditions (2026-07-04, in progress)

A second, independent reviewer verdict (pasted directly by the user, not an
agent) judged the paper "strong Findings / borderline main conference" and
identified the same core fragility Round 3's fixes had already surfaced
(`ConfidenceRuleMerge` matches/beats `ThreeWayLLMMerge`; the advantage
inverts on real MemoryAgentBench content) as the paper's central framing
problem, not just a caveat to disclose. Direction: stop presenting
`ThreeWayLLMMerge`'s win as the headline and instead make the
provenance/abstention boundary condition itself the contribution. User
confirmed (2026-07-04) to execute the full reframing plan, including two new
real-API-cost experiment additions (`semantic_resolvable` category, balanced
conflict-detector benchmark) — see `CHANGELOG.md` and
`ANALYSIS_PLAN_ADDENDUM.md` for the structured summary of every addition in
this round; this section is updated as each phase completes.

**Phase 0**: created `CHANGELOG.md`, `REPRODUCIBILITY.md`,
`ANALYSIS_PLAN_ADDENDUM.md`.

**Phases 1-13 (complete)**:

- **Phase 1**: New title ("BranchMem: Boundary Conditions for Asynchronous
  Semantic Agent Memory Merging"), rewritten 200-word abstract leading with
  the boundary condition rather than the aggregate win, new Introduction
  paragraph stating the boundary-condition framing up front, a new "Why
  abstention matters here" paragraph, and a rewritten 7-item contributions
  list matching the reframing.
- **Phase 2**: `ConfidenceRuleMerge` promoted from ablation-only into
  Table 2 (category breakdown) and Figure 1, computed via a new script
  (`scripts/compute_category_breakdown_extra_strategies.py`) that reruns it
  (and `TwoWayLLMMerge`/`RawTextLLMMerge`) on the identical locked scenarios
  — zero LLM cost for `ConfidenceRuleMerge` (deterministic), a handful of
  new cached calls for the other two (14 cache misses out of 120 possible
  calls; the rest hit the existing cache). RQ1/Ablations prose reframed
  around "the strongest baseline is deterministic abstention, not LWW."
- **Phase 3**: New `branchmem/eval/abstention_metrics.py` (commit rate,
  conditional-on-commit accuracy, wrong-commit rate, flag precision/recall,
  expected utility under 3 cost regimes, coverage-risk), 7 unit tests. New
  script `scripts/compute_abstention_report.py` recomputes the synthetic
  side from cache (6 new cache misses) and reuses the existing
  `abstention_adjusted_metric.json` for the MAB side (zero new calls). New
  Section 5.7 "Abstention-aware evaluation" with a table and two new
  figures (`abstention_coverage.png`, `abstention_utility.png`), full metric
  definitions in a new Appendix H.
- **Phase 4**: New `semantic_resolvable` conflict category
  (`branchmem/benchmark/semantic_resolvable_generator.py`,
  `data/semantic_resolvable_templates.json`, 12 hand-authored templates, 5
  unit tests covering source/confidence equality, non-correlation with
  timestamp, and reproducibility). **Caught a real construct-validity bug
  on the first run**: the ancestor constraint was stored under a different
  predicate than the conflicting key, so `ThreeWayLLMMerge`'s prompt (which
  only includes ancestor value for the exact colliding key) never saw it —
  found by inspecting a cached response directly rather than trusting the
  0.0 aggregate. Fixed the templates and re-ran (both versions disclosed in
  the paper). Real result, n=150, seed 3001: every confidence-independent
  strategy scores exactly 0.000, including `ThreeWayLLMMerge` even with the
  constraint properly exposed — transcripts show inconsistent behavior
  (correct flags, incoherent merges, and at least one confident wrong
  commit that ignored a stated allergy constraint). Reported as a genuine
  negative result strengthening the paper's reframing.
- **Phase 5**: New balanced conflict-detector benchmark
  (`data/balanced_detector_benchmark.json`, 48 hand-authored pairs across 6
  categories, `scripts/run_balanced_detector_benchmark.py`, 4 unit tests for
  the AUROC/metrics helpers). Real result: embedding detector's locked-set
  precision (1.0) was a pure construction artifact — on the balanced set it
  is the weakest detector (precision 0.381); NLI is the best cost-free
  option here (F1 0.800), reversing its locked-set "falsified" standing.
  New Section 5.6 + Appendix G with full metrics.
- **Phase 6**: `annotation/audit_sample.csv` (40 real sampled conflict pairs,
  `scripts/sample_for_audit.py`) and `annotation/audit_instructions.md`.
  `human_*` columns are empty by design; Limitations updated to reference
  this protocol without claiming validation occurred.
- **Phase 7**: Related Work restructured into 6 subsections (agent memory,
  live coordination, single-timeline freshness, structured
  replication/CRDTs/local-first, conflict detection/NLI, abstention/
  selective prediction), with 7 new real, verified citations added to
  `custom.bib` (Reflexion, MemGPT, local-first software, operational
  transformation, and two selective-classification papers).
- **Phase 8**: `REPRODUCIBILITY.md` (exact model/decoding params, retry and
  JSON-parsing policy, cache-key construction, reproduce-without-spending
  commands) and an updated `README.md` module map / results summary.
- **Phase 9**: Replaced several defensive phrases ("not result-shopping",
  "we report this as-is") with more direct scientific framing ("this result
  shows that provenance is load-bearing") without softening any fact.
- **Phase 10**: Balanced-detector-benchmark and full metric-definition
  detail moved to two new appendices (matching the existing pattern of
  moving supporting detail out of the body for page-budget reasons); all
  new table/figure captions state pre-registered/post-hoc status, sample
  size, and takeaway.
- **Phase 11**: Full recompile clean (zero Overfull/error warnings, zero
  undefined references, zero undefined citations); anonymity re-verified
  (`\author{Anonymous submission}`, `[review]` mode, no identifying
  strings); full test suite 71/71 passing (up from 55; +16 new tests across
  3 new test files).
- **Phase 12-13 (final report + ARR audit)**: see the summary delivered to
  the user in this session. **Known open item, disclosed rather than
  hidden**: the substantial legitimate new content this round added (two
  new experiment subsections, a restructured 6-subsection Related Work, an
  expanded contributions list, a new abstention-evaluation section with
  table+figure) pushed the body from 8 pages to approximately 11 pages
  before Limitations. Several rounds of prose tightening and two
  detail-to-appendix moves recovered some of this, but full return to the
  8-page limit needs either further compression or moving additional
  detail to appendices in a follow-up pass — this is the single largest
  remaining task before submission-readiness.

New real LLM API spend this round (all cheap, `gpt-5.4-nano`, cached going
forward): ~14 calls (category-breakdown extra strategies), ~6 calls
(abstention report), 450 calls (semantic_resolvable, first flawed run) + 450
calls (semantic_resolvable, corrected re-run — both cached and disclosed),
and the balanced-detector-benchmark's LLM-judge calls (48 pairs). No calls
were made for `ConfidenceRuleMerge` (deterministic) or for reusing existing
cached MAB/robustness results.

## Round 4 follow-up: page-budget pass (2026-07-04)

User asked to bring the body back toward the 8-page limit via trimming,
shorter captions, and moving detail to appendices. No content, numbers, or
claims were cut — only wording and placement.

- Moved the `semantic_resolvable` category's construct-validity-bug story
  and full transcript analysis to a new Appendix J
  (`sec:semantic-resolvable-detail`), leaving a 6-sentence summary with the
  headline `0.000` result in the body.
- Moved the RQ2 F1-drift bootstrap-CI paragraph to the existing Detector
  Threshold Calibration Sweep appendix, leaving a 2-sentence summary in
  Section 5.4.
- **Found and fixed a real formatting bug**: the abstention-metrics table
  (Table 5) used `[t]` float placement, which let LaTeX jump it to the top
  of a column ahead of the paragraph that introduces it, leaving a
  large blank gap in the middle of page 9. Changed to `[H]` (exact
  placement), matching every other new table in this round.
- Shortened 9 table/figure captions (confirmatory table, category-breakdown
  table and figure, robustness figure, MAB figure, abstention table and
  figure) to their essential claim + number, without dropping any
  qualification.
- Trimmed the Related Work section's 6 subsections (~30% shorter, no
  citations or distinctions dropped), the MAB section's commit/flag
  paragraph (now points to the new abstention table instead of repeating
  its numbers), the RQ1 "Reframing" callout (merged into the preceding
  paragraph), the RQ3 paragraph, and the resolvable-category
  power-expansion paragraph (headline numbers only, full table already in
  Appendix I).
- Removed one stale orphaned LaTeX comment (a leftover note about page
  fitting with no actual command attached).

**Result**: Conclusion moved from page 11 to page 10 (one full page
recovered). Recompiles clean (zero Overfull/error warnings, zero undefined
references), all 71 tests still pass. **Still 2 pages over the 8-page
limit** — the remaining gap is a genuinely large amount of new, substantive
content (two new experiments with their own subsections, a 6-subsection
Related Work, a new Abstention-aware evaluation section with table+figure,
an expanded 7-item contributions list) rather than removable padding.
Closing the rest of the gap would need either cutting one of these new
subsections down to a paragraph (losing detail, not just wording) or moving
a full experimental subsection (not just its supporting detail) to an
appendix — a scope/completeness tradeoff for the user to decide, not a
mechanical trim.

## Round 4 follow-up 2: abstract + Introduction conciseness pass (2026-07-04)

User asked for the abstract at 150-170 words and a more concise
Introduction, with every original idea preserved (wording/length only, no
content cuts).

- Abstract: 200 → **170 words exactly**. Every claim preserved (boundary-
  condition framing, `ConfidenceRuleMerge` promotion, the MAB inversion,
  abstention-aware metrics, the detector result, the release statement).
- Introduction: 845 → **684 words** (~19% shorter). Tightened the
  motivation paragraph, both "not X" distinction paragraphs (live
  coordination, single-timeline freshness, CRDT/local-first), the
  "why abstention matters" paragraph, the RQ list lead-in, the
  preregistration statement, and each of the 7 contribution items —
  every citation, mechanism, and distinct idea is still present, only the
  prose around them is tighter.
- Recompiles clean (zero Overfull/error/undefined-reference warnings), all
  71 tests still pass. Page position unchanged (Conclusion still page 10)
  since this pass targeted conciseness, not the remaining page-budget gap
  from the previous round.

## Round 4 follow-up 3: relocate 3 insignificant subsections to Appendix (2026-07-04)

User asked to move at least 3 low-significance subsections out of the main
body into the Appendix. Chose these three, deliberately avoiding any of the
three preregistered/confirmatory RQs so no confirmatory result loses body
visibility:

1. **Related Work §"Live multi-agent coordination vs.\ offline branch
   reconciliation"** — substantially redundant with the Introduction's own
   S-Bus/abort-model distinction (same citation, same argument), so the
   elaboration was moved and only a one-sentence pointer kept in the body.
2. **Related Work §"Single-timeline freshness and temporal conflict
   resolution"** — same reasoning, redundant with the Introduction's
   max-serial/no-shared-clock distinction (same `reddy2026freshness`
   citation).
3. **Results §"Post-hoc robustness checks: a more capable model, and a
   resolvable-category replication"** — entirely post-hoc/exploratory (not
   one of RQ1/RQ2/RQ3), already partially detailed in an appendix from an
   earlier round; moved the remaining body prose and Figure 2 wholesale,
   leaving only a 2-sentence summary with the headline numbers.

Both relocations preserve every idea and citation — nothing was deleted,
only moved to two new appendix sections ("Extended Related Work: Live
Coordination and Temporal Freshness"; "Post-hoc Robustness Checks Detail")
with a short body-side summary/pointer replacing each original subsection.
Added `\label{sec:intro}` and `\label{sec:related-work}` (didn't exist
before) so the new appendix cross-references resolve correctly.

**Result**: Conclusion moved from page 10 to **page 9** (another full page
recovered — two pages recovered total across this round's two trimming
passes, from page 11 down to page 9). Recompiles clean (zero
Overfull/error/undefined-reference warnings), all 71 tests still pass.
One page still over the 8-page target.

## Round 4 follow-up 4: relocate 3 more subsections to Appendix (2026-07-04)

User asked again to move at least 3 more low-significance subsections to
the Appendix. Again avoided all three preregistered/confirmatory RQs and
the core reframing sections (Ablations, MAB-negative, Abstention-aware
evaluation). Chose:

1. **Related Work §"Agent memory and memory benchmarks"** — its unique
   content (MemoryAgentBench's scope) is already covered in more detail in
   Method §3.5 "MemoryAgentBench extension"; moved in full.
2. **Related Work §"Conflict detection, NLI, and semantic similarity"** —
   generic background supporting RQ2/the detector work specifically, not
   the paper's central provenance/abstention claim; moved in full.
3. **Results §"Post-hoc: a balanced detector benchmark, with real negative
   examples"** — demoted from its own `\subsection` to a short
   `\paragraph` inside RQ2's subsection (2-3 sentences, headline numbers
   only); its full construction methodology and interpretive analysis
   moved into the existing Appendix I ("Balanced Detector Benchmark
   Detail"), which previously held only the table.

Extended the existing "Extended Related Work" appendix (renamed from
"...Live Coordination and Temporal Freshness" to just "Extended Related
Work" since it now holds four subsections, not two) rather than creating a
new one. Every citation and idea preserved — nothing deleted, only
relocated and, in the RQ2 case, shortened in the body with detail moved
alongside it in the appendix.

**Result**: recompiles clean (zero Overfull/error/undefined-reference
warnings), all 71 tests still pass. Conclusion remains on page 9 (this
round's moves reduced subsection *count* and body density further without
yet crossing another page boundary — real progress toward an eventual
further page recovery, even though the page number itself didn't move this
time).

## Round 4 follow-up 5: compact the contributions list (2026-07-04)

User noted the 7-item contributions `\enumerate` list in the Introduction
took up disproportionate vertical space. Converted it from a vertical
bulleted list (each item starting a new line, plus itemsep/topsep spacing
per item) to a single compact inline running-prose paragraph --- ``(i) ...;
(ii) ...; ... (vii) ...'' --- the standard ACL space-saving convention for
this exact situation. All 7 ideas, every citation/section cross-reference,
and every specific claim (RQ1/RQ3, `ConfidenceRuleMerge`, abstention
metrics, balanced detectors, MAB extension) are unchanged, only the
list-vs-paragraph formatting changed.

Recompiles clean (zero Overfull/error/undefined-reference warnings), all 71
tests still pass.

## Round 4 follow-up 6: standard ACL space-saving techniques (2026-07-04)

User asked for broad ACL-standard space-saving techniques applied to the
main content, targeting savings roughly equal to the Discussion section's
size. Applied, in order of impact:

1. **`titlesec` heading spacing** (biggest lever): added the package and
   set `\titlespacing*` for `\section`/`\subsection`/`\paragraph` to
   tighter before/after skips. This only changes whitespace, not ACL's
   required font/size/weight for these headings (`\titlespacing*`, not
   `\titleformat`) — safe against the venue's formatting requirements.
   With ~30 section/subsection headings and 14 `\paragraph` run-in headers
   in the body, this was the single highest-leverage change.
2. **`\setlength{\parskip}{0pt}`** — removes any residual inter-paragraph
   stretch.
3. **Converted the RQ1/RQ2/RQ3 `itemize` list to inline running prose**
   (same treatment as the earlier contributions-list conversion), removing
   per-item list spacing.

Iterated the `\titlespacing` values twice, tightening further after the
first pass showed the Discussion section moving mostly onto page 8 already.
Trimmed the last remaining Discussion paragraph ("The MAB negative result")
by another ~15%.

**Result**: Discussion (heading + first two of three paragraphs) now fits
entirely on page 8, versus starting near the bottom of page 9 before this
pass — a change of almost exactly one Discussion-section's worth of
vertical space, matching the user's stated target. Only the final
Discussion paragraph (~15 lines) plus Conclusion still spill onto page 9.
Recompiles clean (zero Overfull/error/undefined-reference warnings), all 71
tests still pass.

## Round 4 follow-up 7: further abstract trim (2026-07-04)

User asked to trim the abstract further while keeping it excellent. 170 →
**159 words**. Tightened wording throughout (e.g., "a preregistered
benchmark and harness, to identify" → "a preregistered benchmark and
harness to identify"; dropped "throughout" and "not a footnote" as the
least load-bearing words) without cutting any idea, number, or named
result — the full narrative arc (problem framing, `BranchMem` intro,
headline finding, `ConfidenceRuleMerge`'s strongest-baseline status, the
$230$-conflict MAB inversion, abstention-aware metrics, the detector
result, the release statement) is unchanged. Recompiles clean, all 71
tests still pass.

## Round 4 follow-up 8: further Introduction trim (2026-07-04)

User asked to trim the Introduction again while keeping it excellent.
683 → **622 words** (9% reduction). Two changes:

1. Merged the two "three distinguishing problems" paragraphs (live
   multi-agent coordination / single-timeline freshness / CRDT-style
   replication — previously 219 words across two paragraphs) into one
   165-word paragraph. Cut redundant framing ("has a specific structure,
   distinct from three neighbouring problems the literature addresses"
   → "differs from three related lines of work"; "First, ... Second, ...
   Third," → "It is not X ... Nor is it Y ... Nor is it Z") while keeping
   every citation (`khan2026sbus`, `reddy2026freshness`,
   `shapiro2011crdt`, `sarkar2026grite`), every named mechanism
   (`max(serial)`, optimistic concurrency control, commutative merge
   function), and the window-seat/aisle-seat example intact.
2. Tightened the "central finding" paragraph (74 → 70 words) — same
   content, fewer connective words.

Left the "Why abstention matters," RQ1/RQ2/RQ3, and
contributions-list paragraphs unchanged: each is already dense with a
distinct claim or numbered contribution (already compacted in earlier
rounds), and further cuts there risked losing an idea rather than just
wording. Recompiles clean (zero Overfull/error/undefined-reference
warnings); Conclusion still starts on page 9 (unchanged — this was a
prose-quality trim, not a page-budget push); all 71 tests still pass.

## Round 4 follow-up 9: Discussion section trim (2026-07-04)

User asked to trim Discussion again while keeping it excellent. 253 →
**233 words** (8% reduction) across the three paragraphs:

1. **"Construct validity is not free"** (103 → 88 words): tightened
   connective wording ("meant every conflict targeted" → "left every
   conflict targeting", "auditing transcripts, not just aggregate
   scores, caught it" → "only auditing transcripts, not aggregate
   scores, caught it") without dropping either piloting failure or the
   $15\%$-noise design detail.
2. **"Timestamps are a trap"** (already tight; ~33 words): dropped
   redundant "in both generator and extension" (the point — recency
   isn't evidence — doesn't depend on naming both places the fix
   landed).
3. **"The MAB negative result..."** (~109 → ~106 words): tightened
   sentence joins ("In piloting we could fix the benchmark; on real
   content we cannot, so..." → "We could fix this in piloting but not
   on real content, so..."); kept the research-integrity statement
   (that no post-hoc rule crediting calibrated abstention was added,
   since that would be exactly the result-shopping preregistration
   exists to prevent) fully intact, since cutting it would weaken a
   load-bearing epistemic claim, not just trim prose.

**Result**: the entire Discussion section, including the third paragraph
that previously spilled onto page 9, now fits on page 8 — and the
Conclusion moved from page 9 to **page 8** as a result. Recompiles clean
(zero Overfull/error/undefined-reference warnings), all 71 tests still
pass.

## Round 4 follow-up 10: trim other sections to pull Conclusion tail onto page 8 (2026-07-04)

The Conclusion's closing two sentences ("and loses to every naive baseline
once that signal is gone. We take this as a boundary condition on RQ1, not
a caveat to bury: deployment needs either a genuine reliability signal in
memory provenance, or a downstream consumer that treats
\textsc{flagged\_unresolved} as actionable, not a failure.") were still
spilling alone onto an otherwise-empty page 9 after the Discussion trim.
User asked to trim other sections to reclaim that space, not the
Conclusion itself. Tightened three earlier, already-dense paragraphs
without dropping any claim or number:

1. **MemoryAgentBench extension** (Problem Formulation,
   Section~\ref{sec:mab}): 172 → 149 words. Cut redundant framing
   ("Its published data exposes" → "Since its published data exposes",
   "extraction correctness" → "correctness", "drawn uniformly at random
   from" → "of") while keeping every count ($99.9\%$ coverage,
   $18{,}337$ lines, $50$/$51{,}314$ audited triples, $0$ errors,
   $49$--$56\%$ LWW range, $17{,}816$ pairs).
2. **RQ2 real-content MemoryAgentBench negative result**
   (Section~\ref{sec:mab-negative}, first paragraph): 145 → 121 words.
   Removed a genuinely redundant clause restating the $15\to50$ cap
   decision a second time ("a uniform scale/cost decision applied after
   seeing the $15$-cap number but identically to every strategy") that
   repeated what the prior sentence already said.
3. **RQ2 detector generalization-gap paragraph**
   (Section~\ref{sec:results}, "RQ2: cheap detectors vs.\ LLM judge"):
   161 → 146 words. Dropped one redundant "both" and tightened
   connectives; kept every F1 number, CI, and the calibration-set /
   locked-set comparison intact.

**Result**: the Conclusion's full text, including the quoted closing
sentences, now fits entirely on page 8; "Limitations" starts cleanly at
the top of page 9. Total PDF page count dropped from 16 to **15**.
Recompiles clean (zero Overfull/error/undefined-reference warnings), all
71 tests still pass.

## Round 4 follow-up 11: remove all "peer review" mentions (2026-07-04)

User asked to remove any text mentioning "peer review." Found and rewrote
7 occurrences across the main body, a table caption, and one appendix,
each time keeping the underlying disclosure (that a check was post-hoc /
added later / exploratory rather than confirmatory) and only dropping the
"peer review" framing itself:

- MemoryAgentBench extension (Section~\ref{sec:mab}): "so, in response to
  peer review, we ran an author self-audit" → "so we additionally ran an
  author self-audit."
- Experimental Setup, "Post-hoc robustness checks" paragraph: "In response
  to peer review, we ran two additional checks" → "We ran two additional
  checks after the locked analysis was complete."
- Ablations table caption: "added in response to peer review" → "added as
  a later post-hoc check."
- MAB-negative analysis ($15\to50$ cap decision): "in response to a later
  peer-review round asking for more power" → "to gain more statistical
  power."
- Limitations intro: "two added after post-hoc checks run in response to
  peer review" → "two added after later post-hoc checks."
- Limitations, "No independent validation" paragraph: "What we did add, in
  response to peer review: an author self-audit" → "What we did add: an
  author self-audit."
- Appendix~\ref{sec:calibration}, "Judge F1 drift, bootstrapped": "the
  independently-generated, three-times-larger locked set (added in
  response to peer review)" → "the independently-generated,
  three-times-larger locked set" (parenthetical dropped entirely, already
  redundant with the surrounding sentence).

Verified zero remaining matches for "peer review" / "peer-review" in
`paper/acl_latex.tex`. Recompiles clean (zero Overfull/error/undefined-
reference warnings); page count unchanged at 15; Conclusion still on page
8; all 71 tests still pass.
