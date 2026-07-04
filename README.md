# BranchMem

Asynchronous partition-and-merge for semantic agent memory: an evaluation
harness identifying the boundary conditions under which reconciling two
branches of an agent's natural-language memory (diverged while
disconnected, no shared timeline, no option to abort or negotiate live) is
reliable — and reporting, honestly, where it is not.

See `ANALYSIS_PLAN.md` for the preregistered analysis,
`ANALYSIS_PLAN_ADDENDUM.md` for every post-hoc addition since, and
`CHANGELOG.md` for a chronological index of every substantive change. **Results:
`results/final/findings.md`; reproduction details: `REPRODUCIBILITY.md`.**

Headline finding: LLM-mediated merging beats naive baselines only when
branch provenance carries a legible reliability signal — a deterministic
rule reading that same signal (`ConfidenceRuleMerge`) matches or beats it,
and on real content where no such signal exists, the model correctly
abstains rather than guesses, which raw accuracy alone penalizes as error.

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
python scripts/run_experiment.py          # the locked Phase 6 run (n=60, real API calls)
```

All LLM calls are cached (content-hash keyed) under `llm_cache/`, which is
committed to git as an auditable artifact — re-running either script costs
nothing once the cache is populated for that exact scenario/prompt set.

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
                                strongest baseline, promoted out of ablation-only
    two_way_llm.py             ablation: LLM merge without ancestor context
    raw_text_llm.py            ablation: LLM merge with no metadata at all
  benchmark/
    scenario_generator.py   synthetic branch-divergence scenarios, ground truth
                              by construction (orthogonal / resolvable / ambiguous)
    semantic_resolvable_generator.py  post-hoc category: equal source/confidence,
                              ground truth recoverable only from ancestor semantics
    mab_extension.py         extends real MemoryAgentBench data with a branch-fork
                              step (validated against live data; not in the Phase 6
                              locked run — see findings.md limitations)
    downstream_tasks.py       QA-style questions scored against generator ground
                                truth, never against a reference LLM
  evaluation/
    metrics.py         downstream accuracy (by category), detector P/R/F1
    stats.py            paired Wilcoxon, paired bootstrap CI, Holm-Bonferroni
    runner.py            orchestrates a full benchmark run (used by both scripts)
    plots.py              accuracy-by-strategy bar chart
  eval/
    abstention_metrics.py   post-hoc: commit rate, conditional accuracy,
                              wrong-commit rate, expected utility, coverage-risk
  utils/
    config.py, logging.py, seeding.py

configs/
  default.yaml        dev defaults (mock backend, calibrated detector thresholds)
  experiment.yaml       Phase 6 locked overrides (model, n_scenarios, seed)

data/
  scenario_templates.json              entity/predicate/value pools for the generator
  semantic_resolvable_templates.json     ancestor-constraint templates (post-hoc)
  balanced_detector_benchmark.json        6-category conflict/non-conflict pairs (post-hoc)

annotation/
  audit_sample.csv, audit_instructions.md   independent-validation protocol
                                              (no human annotation performed here)

scripts/
  run_pilot.py          Phase 5 pilot (historical record; thresholds now hardcoded
                          to the calibrated values it discovered)
  run_experiment.py       Phase 6 locked run
  run_unit_tests.py         thin pytest wrapper
  compute_category_breakdown_extra_strategies.py   ConfidenceRuleMerge/TwoWayLLM/
                                                      RawTextLLM category breakdown
  compute_abstention_report.py    abstention-aware metrics, synthetic + MAB
  run_semantic_resolvable_experiment.py   post-hoc semantic_resolvable experiment
  run_balanced_detector_benchmark.py       post-hoc balanced detector benchmark
  sample_for_audit.py                        builds annotation/audit_sample.csv
results/
  pilot/pilot_summary.json
  final/{results.csv,results.json,stats_output.json,accuracy_by_strategy.png,
          findings.md, semantic_resolvable.json, balanced_detector_benchmark.json,
          abstention_metrics_report.json, category_breakdown_extra_strategies.json}
```

## Results summary

H1 (three-way merge beats naive baselines) and H2 (three-way merge beats
branch-discard) are both **supported**, with large, well-powered effect
sizes (paired Wilcoxon, Holm-Bonferroni corrected, all p < 1e-9). H3 (can
conflict detection be done cheaply) is a genuine **mixed result**: an
embedding-similarity detector stayed within the preregistered F1 tolerance
of an LLM-judge reference; an NLI cross-encoder detector did not, on the
locked Phase 6 scenario set. Full numbers, category breakdowns, and
limitations: `results/final/findings.md`.

## Known limitations (see findings.md for the full list)

- Single LLM (`gpt-5.4-nano`, chosen for cost, not capability) for all
  LLM-dependent results.
- `ThreeWayLLMMerge` and `LLMJudgeConflictDetector` were called once per
  scenario/pair (not majority-vote across repeats), per a preregistered
  cost/variance tradeoff — see ANALYSIS_PLAN.md sec.6.
- The MemoryAgentBench extension is validated (Phase 4) but not part of the
  Phase 6 locked run; scaling the benchmark to include it, and to a more
  capable model for confirmation, are the natural next steps.
- H2's effect size is partly a near-mechanical consequence of the benchmark
  always including orthogonal questions — disclosed in ANALYSIS_PLAN.md
  before Phase 6 ran, not discovered after the fact.
