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

## Earlier rounds

Phases 0-7 (initial build through `findings.md`), a first review-response
round (`ConfidenceRuleMerge`/`TwoWayLLMMerge` ablations, expanded MAB
sample), and a second review-response round (`RawTextLLMMerge`,
resolvable-category power expansion, parser-precision self-audit, RQ2 F1
bootstrap CIs).
