# CHANGELOG.md

Chronological, file-level index of substantive changes to the BranchMem
codebase and paper.

## Round 4 (2026-07-04): Reframing around provenance + abstention boundary conditions

Triggered by an external reviewer verdict: the headline `ThreeWayLLMMerge`
result is fragile (a deterministic `ConfidenceRuleMerge` matches/beats it;
the advantage inverts on real MemoryAgentBench content). Rather than treat
this as a flaw to patch around, the paper's framing is being changed to make
the boundary condition itself the contribution.

- **Reframe**: new title, abstract, introduction, contributions list built
  around "merging works only when provenance carries a legible reliability
  signal; abstention-aware evaluation is required when it doesn't."
- **`ConfidenceRuleMerge` promoted** from an ablation-only strategy to a
  main-table baseline.
- **New**: `branchmem/eval/abstention_metrics.py` — commit rate, conditional
  accuracy, wrong-commit rate, expected utility under three cost regimes,
  coverage-risk, computed from already-cached result files (no new API
  calls).
- **New**: `semantic_resolvable` conflict category — equal source, equal
  confidence, uninformative timestamps; ground truth resolvable only from
  raw-text/ancestor semantics. New post-hoc experiment, real API calls,
  disjoint from the locked confirmatory run.
- **New**: balanced conflict-detector benchmark (contradiction, paraphrase,
  entailment, unrelated-same-entity, preference-refinement, ambiguous) to
  replace the all-positive locked-set precision of 1.0 with a real
  precision/recall/F1/AUROC picture.
- **New**: `annotation/` audit protocol (sample CSV + instructions) for
  independent validation — scaffolding only, no claim of human validation
  performed.
- **Docs**: this file, `REPRODUCIBILITY.md`, `ANALYSIS_PLAN_ADDENDUM.md`
  added; `README.md` updated.

## Round 5 (2026-07-04): Cross-vendor robustness check

Real Anthropic API access obtained after the locked run. Replicated three
existing post-hoc analyses with `claude-haiku-4-5-20251001` on identical
scenario constructions, directly testing whether the paper's effects are
OpenAI-specific:

- **RQ1** (same 20 scenarios as the `gpt-5.4-mini` check): `claude-haiku-4-5`
  scores 0.936 vs. 0.914 (nano) / 0.929 (mini) — effect holds across
  a genuinely different vendor.
- **`semantic_resolvable`** (identical 150-scenario construction): every
  confidence-independent strategy again scores exactly 0.000 — the
  paper's most striking negative result generalizes across vendors.
- **Real MemoryAgentBench content**: `ThreeWayLLMMerge` again scores
  exactly 0.000 — complete abstention on real content is a property of
  the data, not one provider's model.
- **New**: `scripts/run_cross_vendor_rq1.py`,
  `scripts/run_cross_vendor_semantic_resolvable.py`,
  `scripts/run_cross_vendor_mab.py`; results in `results/final/cross_vendor_*.json`.
- **Docs**: `ANALYSIS_PLAN_ADDENDUM.md` item A5; paper's Limitations,
  "Does raw text ever carry the resolving signal?", "Post-hoc robustness
  checks", "Post-hoc secondary analysis: real MemoryAgentBench content",
  and the "Post-hoc Robustness Checks Detail" appendix updated.

## Round 6 (2026-07-04): Reviewer-response analyses

Four post-hoc analyses answering a reviewer's questions; none change any
prior result:

- **Noise sensitivity** (`scripts/run_noise_sensitivity.py`): boundary
  condition holds across `source_signal_noise` 0.0-0.5; ConfidenceRule >=
  LLM at every level; both decay to chance as the signal becomes
  uninformative. Zero new API (cache reuse).
- **Semantic-resolvable failure modes**
  (`scripts/analyze_semantic_resolvable_failures.py`): 0 correct commits of
  150; 83% flagged; all active errors in constraint-exclusion/negation
  templates. Zero new API.
- **Risk-coverage frontier** (`scripts/compute_risk_coverage_frontier.py`):
  abstaining strategies reach risk 0.13-0.17 at coverage ~0.67 vs.
  0.63-0.70 for never-abstaining baselines. Zero new API.
- **Provenance proxy at branch granularity**
  (`scripts/run_provenance_proxy.py`): a per-replica reliability proxy is
  exploited as well as the per-fact signal (0.883 vs. 0.858/0.867).
- **New**: two truth-discovery/knowledge-fusion citations
  (`dong2009integrating`, `li2016survey`); main-text characterization +
  Extended Related Work subsection.
- **Docs**: `ANALYSIS_PLAN_ADDENDUM.md` item A6; paper Sections 2, 5.1, 5.5,
  5.6, 5.8, Limitations, and Appendices J/K/M updated; two new appendix
  figures (noise sweep, risk-coverage frontier).

## Earlier rounds

Phases 0-7 (initial build through `findings.md`), a first review-response
round (`ConfidenceRuleMerge`/`TwoWayLLMMerge` ablations, expanded MAB
sample), and a second review-response round (`RawTextLLMMerge`,
resolvable-category power expansion, parser-precision self-audit, RQ2 F1
bootstrap CIs).
