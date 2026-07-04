"""Orchestrates a full benchmark run: generate scenarios, run every merge
strategy and conflict detector on each, score, and write raw + aggregated
results. Used by both scripts/run_pilot.py (informally) and
scripts/run_experiment.py (the locked Phase 6 run)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.scenario_generator import ScenarioConfig, ScenarioGenerator
from branchmem.conflict.embedding_detector import EmbeddingConflictDetector
from branchmem.conflict.llm_judge_detector import LLMJudgeConflictDetector
from branchmem.conflict.nli_detector import NLIConflictDetector
from branchmem.evaluation.metrics import score_detector, score_downstream
from branchmem.llm.base import LLMBackend
from branchmem.merge.branch_discard import BranchDiscard
from branchmem.merge.last_writer_wins import LastWriterWins
from branchmem.merge.naive_concat import NaiveConcat
from branchmem.merge.three_way_llm import ThreeWayLLMMerge
from branchmem.utils.logging import get_logger

logger = get_logger("runner")


@dataclass
class ScenarioRunResult:
    scenario_id: str
    divergence_span: float
    accuracy_by_strategy: dict[str, float] = field(default_factory=dict)
    category_accuracy_by_strategy: dict[str, dict[str, float]] = field(default_factory=dict)
    n_questions: int = 0
    n_conflict_pairs: int = 0


def build_strategies(backend: LLMBackend) -> dict:
    return {
        "last_writer_wins": LastWriterWins(),
        "naive_concat": NaiveConcat(),
        "branch_discard_always_b": BranchDiscard(policy="always_b"),
        "branch_discard_fewer_updates": BranchDiscard(policy="fewer_updates"),
        "three_way_llm": ThreeWayLLMMerge(backend=backend),
    }


def build_detectors(
    backend: LLMBackend, embedding_threshold: float = 0.80, nli_threshold: float = 0.20
) -> dict:
    return {
        "embedding_threshold": EmbeddingConflictDetector(threshold=embedding_threshold),
        "nli": NLIConflictDetector(contradiction_threshold=nli_threshold),
        "llm_judge": LLMJudgeConflictDetector(backend=backend),
    }


def run_full_benchmark(
    backend: LLMBackend,
    n_scenarios: int,
    seed: int,
    divergence_spans: list[float],
    scenario_config_kwargs: Optional[dict] = None,
    embedding_threshold: float = 0.80,
    nli_threshold: float = 0.20,
) -> tuple[list[ScenarioRunResult], dict, list]:
    """Run every merge strategy and detector over `n_scenarios`, split evenly
    across `divergence_spans`. Returns (per-scenario results, detector
    scores dict, raw conflict pairs list) for downstream analysis/reporting.
    """
    generator = ScenarioGenerator()
    strategies = build_strategies(backend)
    detectors = build_detectors(backend, embedding_threshold, nli_threshold)

    scenario_config_kwargs = scenario_config_kwargs or {}
    n_per_span = n_scenarios // len(divergence_spans)
    remainder = n_scenarios - n_per_span * len(divergence_spans)

    scenarios = []
    for i, span in enumerate(divergence_spans):
        n_this_span = n_per_span + (1 if i < remainder else 0)
        config = ScenarioConfig(divergence_span=span, **scenario_config_kwargs)
        span_seed = seed + i  # distinct sub-seed per span, deterministic overall
        scenarios.extend(generator.generate(n_this_span, config, seed=span_seed))
    logger.info("Generated %d scenarios across %d divergence spans", len(scenarios), len(divergence_spans))

    results: list[ScenarioRunResult] = []
    all_pairs = []
    all_judgments: dict[str, list] = {name: [] for name in detectors}
    fact_lookup: dict[str, object] = {}

    for idx, scenario in enumerate(scenarios):
        questions = generate_downstream_questions(scenario)
        run_result = ScenarioRunResult(
            scenario_id=scenario.scenario_id,
            divergence_span=scenario.metadata.get("divergence_span", 0.0),
            n_questions=len(questions),
            n_conflict_pairs=len(scenario.conflict_pairs),
        )
        for name, strategy in strategies.items():
            merge_result = strategy.merge(scenario.ancestor, scenario.branch_a, scenario.branch_b)
            score = score_downstream(questions, merge_result)
            run_result.accuracy_by_strategy[name] = score.accuracy
            run_result.category_accuracy_by_strategy[name] = {
                cat: score.accuracy_for(cat) for cat in ("orthogonal", "resolvable", "ambiguous")
            }
        results.append(run_result)

        for f in scenario.branch_a.facts + scenario.branch_b.facts:
            fact_lookup[f.fact_id] = f
        all_pairs.extend(scenario.conflict_pairs)
        for name, detector in detectors.items():
            for pair in scenario.conflict_pairs:
                fact_a, fact_b = fact_lookup[pair.fact_a_id], fact_lookup[pair.fact_b_id]
                all_judgments[name].append(detector.detect(fact_a, fact_b))

        if (idx + 1) % 10 == 0 or idx == len(scenarios) - 1:
            logger.info("Processed %d/%d scenarios", idx + 1, len(scenarios))

    detector_scores = {
        name: score_detector(all_pairs, judgments, name) for name, judgments in all_judgments.items()
    }
    return results, detector_scores, all_pairs


def write_results(
    results: list[ScenarioRunResult], detector_scores: dict, out_dir: Path, run_metadata: dict
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    results_json = {
        "metadata": run_metadata,
        "per_scenario": [asdict(r) for r in results],
        "detector_scores": {
            name: {
                "precision": s.precision, "recall": s.recall, "f1": s.f1,
                "mean_latency_s": s.mean_latency_s, "n_pairs": s.n_pairs,
            }
            for name, s in detector_scores.items()
        },
    }
    (out_dir / "results.json").write_text(json.dumps(results_json, indent=2))

    strategy_names = list(results[0].accuracy_by_strategy.keys()) if results else []
    csv_lines = ["scenario_id,divergence_span," + ",".join(strategy_names)]
    for r in results:
        row = [r.scenario_id, str(r.divergence_span)] + [str(r.accuracy_by_strategy[s]) for s in strategy_names]
        csv_lines.append(",".join(row))
    (out_dir / "results.csv").write_text("\n".join(csv_lines))

    logger.info("Wrote %s and %s", out_dir / "results.json", out_dir / "results.csv")
