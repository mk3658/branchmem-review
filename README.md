# BranchMem

Code, data, and cached model calls for **BranchMem**, a preregistered
benchmark and harness for asynchronous partition-and-merge in LLM-agent
memory: reconciling two branches of an agent's natural-language memory that
diverged while disconnected (no shared clock, no live negotiation), and
identifying the conditions under which that reconciliation can be trusted.

This repository accompanies an anonymous submission and is released for
double-blind review. It contains no author-identifying information.

## Headline finding

Ancestor-aware LLM merging (`ThreeWayLLMMerge`) beats naive baselines
(last-writer-wins, concatenation, branch-discard) by a wide margin — but
only when branch provenance carries a legible reliability signal. A
deterministic rule reading that same signal (`ConfidenceRuleMerge`) matches
or beats the LLM at zero API cost, and on real, branch-forked
MemoryAgentBench content — where no such signal exists — the ranking
inverts entirely: the LLM abstains on every one of 230 conflicts rather
than guess, which raw accuracy alone scores as failure. We report
abstention-aware metrics (commit rate, conditional accuracy, expected
utility) showing that calibrated abstention is the correct behavior once
the signal is absent, and a separate, harder, balanced conflict-detection
benchmark shows a genuine generalization gap for cheap detectors.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Real LLM calls (Anthropic or OpenAI-compatible backends) require an API key
in the environment:

```bash
export ANTHROPIC_API_KEY=...    # for llm.backend: anthropic
# or
export OPENAI_API_KEY=...       # for llm.backend: openai_compatible
```

No key is needed to run the test suite or use `llm.backend: mock`.

## Run

```bash
pytest tests/ -q                          # unit tests (no API key needed)
python scripts/run_pilot.py               # small real-LLM pilot (~few hundred calls)
python scripts/run_experiment.py          # the locked confirmatory run (n=60, real API calls)
```

Every real LLM call across every round of this project is cached
(content-hash keyed) under `llm_cache/` and committed as an auditable
artifact — re-running any script costs nothing once the cache is populated
for that exact scenario/prompt combination.

## Module map

```
branchmem/
  memory/
    schemas.py       MemoryFact, MemoryBranch, ConflictPair, MergeResult, Resolution
    store.py          SQLite-backed fact store with branch/provenance tracking
    branch_sim.py      BranchSimulator: fork a common ancestor, apply scripted
                         per-branch UpdateSpec streams, derive ground-truth conflicts
  llm/
    base.py            LLMBackend ABC; complete() wraps caching + logging
    cache.py           content-hash JSON cache for every real call
    mock_backend.py     deterministic, offline, dev/test-only backend
    anthropic_backend.py, openai_compatible_backend.py
                         real backends; read API keys from env, fail fast if missing
  conflict/
    base.py            ConflictDetector ABC
    embedding_detector.py   cosine-similarity threshold (local, no API cost)
    nli_detector.py          NLI cross-encoder contradiction probability (local)
    llm_judge_detector.py     LLM-as-judge (the RQ2 reference standard)
  merge/
    base.py            MergeStrategy ABC + shared key-collision helpers
    last_writer_wins.py   naive, no branch-awareness baseline
    naive_concat.py        keep-everything baseline
    branch_discard.py       OCC-abort analogue (always_b / fewer_updates policies)
    three_way_llm.py         the proposed method: ancestor + source/confidence-aware
                                LLM merge, batched per-scenario for cost control
    confidence_rule.py        deterministic, no-LLM baseline: keep higher-confidence
                                branch, flag on exact tie — the benchmark's
                                strongest baseline
    two_way_llm.py             ablation: LLM merge without ancestor context
    raw_text_llm.py            ablation: LLM merge with no metadata at all
  benchmark/
    scenario_generator.py   synthetic branch-divergence scenarios, ground truth
                              by construction (orthogonal / resolvable / ambiguous)
    semantic_resolvable_generator.py  ground truth recoverable only from
                              ancestor semantics, not source/confidence
    mab_extension.py         extends real MemoryAgentBench data with a branch-fork
                              step (validated against live data; see findings.md
                              for scope relative to the locked run)
    downstream_tasks.py       QA-style questions scored against generator ground
                                truth, never against a reference LLM
  evaluation/
    metrics.py         downstream accuracy (by category), detector P/R/F1
    stats.py            paired Wilcoxon, paired bootstrap CI, Holm-Bonferroni
    runner.py            orchestrates a full benchmark run (used by both scripts)
    plots.py              accuracy-by-strategy bar chart
  eval/
    abstention_metrics.py   commit rate, conditional accuracy, wrong-commit
                              rate, expected utility, coverage-risk
  utils/
    config.py, logging.py, seeding.py

configs/
  default.yaml        dev defaults (mock backend, calibrated detector thresholds)
  experiment.yaml       locked confirmatory-run overrides (model, n_scenarios, seed)

data/
  scenario_templates.json              entity/predicate/value pools for the generator
  semantic_resolvable_templates.json     ancestor-constraint templates
  balanced_detector_benchmark.json        6-category conflict/non-conflict pairs

annotation/
  audit_sample.csv, audit_instructions.md   independent-validation protocol
                                              (no human annotation performed here)

scripts/
  run_pilot.py          pilot run (historical record; thresholds now hardcoded
                          to the calibrated values it discovered)
  run_experiment.py       the locked confirmatory run
  run_unit_tests.py         thin pytest wrapper
  compute_category_breakdown_extra_strategies.py   ConfidenceRuleMerge/TwoWayLLM/
                                                      RawTextLLM category breakdown
  compute_abstention_report.py    abstention-aware metrics, synthetic + MAB
  run_semantic_resolvable_experiment.py   semantic_resolvable-category experiment
  run_balanced_detector_benchmark.py       balanced detector benchmark
  sample_for_audit.py                        builds annotation/audit_sample.csv

results/
  pilot/pilot_summary.json
  final/{results.csv,results.json,stats_output.json,accuracy_by_strategy.png,
          findings.md, semantic_resolvable.json, balanced_detector_benchmark.json,
          abstention_metrics_report.json, category_breakdown_extra_strategies.json}
```

## Results summary

The confirmatory result (locked `n=60`, Holm-Bonferroni corrected) shows
`ThreeWayLLMMerge` beating naive baselines by a wide, well-powered margin
overall. Decomposing by category shows this win is concentrated in
"resolvable" conflicts, where correctness is tied to an explicit
source/confidence signal — and a single deterministic rule
(`ConfidenceRuleMerge`) reading that same signal matches or slightly beats
the LLM (0.962 vs. 0.936). A separate ablation designed so the answer
requires genuine semantic reasoning over raw text (not the engineered
signal) produces a striking negative result: every confidence-independent
strategy, including the LLM, scores exactly 0.000. On real, branch-forked
MemoryAgentBench content — which has no engineered signal — the ranking
inverts completely (`ThreeWayLLMMerge`: 0.936 → 0.004), with the model
abstaining on every conflict rather than guessing.

Conflict detection (cheap embedding/NLI detectors vs. an LLM-judge
reference) is a genuine **mixed result**: the locked-set precision of 1.0
is a construction artifact of an all-positive test set, and both
detectors' rankings flip on a harder, balanced follow-up set with real
negative examples.

Full numbers, category breakdowns, and every caveat: `results/final/findings.md`,
`ANALYSIS_PLAN.md` (preregistered analysis), `ANALYSIS_PLAN_ADDENDUM.md`
(every post-hoc addition, disclosed separately), and `CHANGELOG.md`
(chronological index). Reproduction details: `REPRODUCIBILITY.md`.

## Known limitations (see findings.md and ANALYSIS_PLAN_ADDENDUM.md for the full list)

- Single, cheap LLM (`gpt-5.4-nano`) for nearly all results; the one
  robustness check is a same-provider swap to a slightly larger model on a
  small scenario subset — no different-vendor or open-weight model tested.
- The real-content negative result rests on 8 MemoryAgentBench rows (230
  conflicts after raising a per-row cap) — the full available dataset for
  this task, not a cost-driven subsample, but still a small sample.
- No independent (non-author) validation of any ground-truth label; an
  audit protocol and sampled item set are released (`annotation/`) to
  support independent validation, but none has been performed for this
  submission.
- `ThreeWayLLMMerge` and `LLMJudgeConflictDetector` were called once per
  scenario/pair (not majority-vote across repeats), a preregistered
  cost/variance tradeoff — see `ANALYSIS_PLAN.md` sec. 6.
