#!/usr/bin/env python3
"""Phase 5 pilot: small-scale real-LLM run to estimate effect sizes and
variance for ANALYSIS_PLAN.md. Uses the cheapest available model
(gpt-5.4-nano) per the project's cost-control requirement. Every call is
cached under llm_cache/ (committed to git) so re-running this script costs
nothing after the first pass.

Where an LLM call feeds into a reported metric, we resample a subset with
use_cache=False to measure non-determinism (temperature=0 does not guarantee
identical output across calls) rather than pretending temp=0 means
deterministic.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from branchmem.benchmark.downstream_tasks import generate_downstream_questions
from branchmem.benchmark.scenario_generator import ScenarioConfig, ScenarioGenerator
from branchmem.conflict.embedding_detector import EmbeddingConflictDetector
from branchmem.conflict.llm_judge_detector import LLMJudgeConflictDetector
from branchmem.conflict.nli_detector import NLIConflictDetector
from branchmem.evaluation.metrics import score_detector, score_downstream
from branchmem.llm.base import build_backend
from branchmem.merge.branch_discard import BranchDiscard
from branchmem.merge.last_writer_wins import LastWriterWins
from branchmem.merge.naive_concat import NaiveConcat
from branchmem.merge.three_way_llm import ThreeWayLLMMerge
from branchmem.utils.logging import get_logger
from branchmem.utils.seeding import set_all_seeds

logger = get_logger("run_pilot")

PILOT_SEED = 2026
N_SCENARIOS = 20
N_VARIANCE_SCENARIOS = 5  # subset resampled 3x to measure LLM non-determinism
N_VARIANCE_REPEATS = 3
MODEL = "gpt-5.4-nano"
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "pilot"


def main() -> None:
    set_all_seeds(PILOT_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    llm_config = {"backend": "openai_compatible", "model": MODEL, "cache_dir": "llm_cache",
                  "temperature": 0.0, "max_tokens": 1024}
    backend = build_backend(llm_config)

    generator = ScenarioGenerator()
    config = ScenarioConfig(n_orthogonal_a=1, n_orthogonal_b=1, n_resolvable=2, n_ambiguous=1, divergence_span=8.0)
    scenarios = generator.generate(N_SCENARIOS, config, seed=PILOT_SEED)

    strategies = {
        "last_writer_wins": LastWriterWins(),
        "naive_concat": NaiveConcat(),
        "branch_discard_always_b": BranchDiscard(policy="always_b"),
        "branch_discard_fewer_updates": BranchDiscard(policy="fewer_updates"),
        "three_way_llm": ThreeWayLLMMerge(backend=backend),
    }

    per_scenario_accuracy: dict[str, list[float]] = {name: [] for name in strategies}
    per_scenario_category_acc: dict[str, dict[str, list[float]]] = {name: {} for name in strategies}
    all_questions_count = 0

    for scenario in scenarios:
        questions = generate_downstream_questions(scenario)
        all_questions_count += len(questions)
        for name, strategy in strategies.items():
            result = strategy.merge(scenario.ancestor, scenario.branch_a, scenario.branch_b)
            score = score_downstream(questions, result)
            per_scenario_accuracy[name].append(score.accuracy)
            for cat in ("orthogonal", "resolvable", "ambiguous"):
                acc = score.accuracy_for(cat)
                if acc == acc:  # not NaN
                    per_scenario_category_acc[name].setdefault(cat, []).append(acc)

    logger.info("Ran %d scenarios (%d downstream questions total)", N_SCENARIOS, all_questions_count)

    # --- conflict detectors, over every conflict pair in the pilot set ---
    fact_lookup = {}
    for s in scenarios:
        for f in s.branch_a.facts + s.branch_b.facts:
            fact_lookup[f.fact_id] = f
    all_pairs = [p for s in scenarios for p in s.conflict_pairs]

    # Calibrated in ANALYSIS_PLAN.md sec.3 from an earlier pilot pass; locked
    # in configs/default.yaml. Left hardcoded here (rather than reading the
    # config) since this script predates the calibration and re-running it
    # is just a historical record at this point -- Phase 6's run_experiment.py
    # reads these from configs/default.yaml.
    embedding_detector = EmbeddingConflictDetector(threshold=0.80)
    nli_detector = NLIConflictDetector(contradiction_threshold=0.20)
    llm_judge_detector = LLMJudgeConflictDetector(backend=backend)

    detector_scores = {}
    detector_judgments_cache = {}
    for detector in (embedding_detector, nli_detector, llm_judge_detector):
        judgments = []
        for pair in all_pairs:
            fact_a, fact_b = fact_lookup[pair.fact_a_id], fact_lookup[pair.fact_b_id]
            judgments.append(detector.detect(fact_a, fact_b))
        score = score_detector(all_pairs, judgments, detector.name)
        detector_scores[detector.name] = score
        detector_judgments_cache[detector.name] = judgments
        logger.info(
            "detector %-15s P=%.3f R=%.3f F1=%.3f latency=%.4fs n=%d",
            detector.name, score.precision, score.recall, score.f1, score.mean_latency_s, score.n_pairs,
        )

    # --- variance estimation: resample three_way_llm and llm_judge on a subset ---
    variance_scenarios = scenarios[:N_VARIANCE_SCENARIOS]
    three_way_variance: dict[str, list[float]] = {}
    three_way_merge_for_variance = ThreeWayLLMMerge(backend=backend)
    for scenario in variance_scenarios:
        questions = generate_downstream_questions(scenario)
        accs = []
        for _ in range(N_VARIANCE_REPEATS):
            # use_cache=False: a genuinely fresh call each time, not a cache hit
            # replaying the same response 3x (which would falsely show zero variance).
            result = three_way_merge_for_variance.merge(
                scenario.ancestor, scenario.branch_a, scenario.branch_b, use_cache=False
            )
            accs.append(score_downstream(questions, result).accuracy)
        three_way_variance[scenario.scenario_id] = accs

    llm_judge_flip_count = 0
    llm_judge_variance_pairs = all_pairs[:10]
    llm_judge_for_variance = LLMJudgeConflictDetector(backend=backend)
    for pair in llm_judge_variance_pairs:
        fact_a, fact_b = fact_lookup[pair.fact_a_id], fact_lookup[pair.fact_b_id]
        judgments = []
        for _ in range(N_VARIANCE_REPEATS):
            j = llm_judge_for_variance.detect(fact_a, fact_b, use_cache=False)
            judgments.append(j.is_conflict)
        if len(set(judgments)) > 1:
            llm_judge_flip_count += 1

    # --- effect sizes among the 5 strategies (informs H1's minimum effect size) ---
    strategy_names = list(strategies.keys())
    pairwise_diffs = {}
    for i, name_a in enumerate(strategy_names):
        for name_b in strategy_names[i + 1:]:
            diffs = [
                a - b for a, b in zip(per_scenario_accuracy[name_a], per_scenario_accuracy[name_b])
            ]
            pairwise_diffs[f"{name_a}_vs_{name_b}"] = {
                "mean_diff": statistics.mean(diffs),
                "sd_diff": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
            }

    total_input_tokens = 0
    total_output_tokens = 0
    n_llm_calls = 0
    for path in (Path("llm_cache")).glob("*.json"):
        data = json.loads(path.read_text())
        total_input_tokens += data.get("input_tokens", 0)
        total_output_tokens += data.get("output_tokens", 0)
        n_llm_calls += 1

    summary = {
        "seed": PILOT_SEED,
        "n_scenarios": N_SCENARIOS,
        "model": MODEL,
        "mean_accuracy_by_strategy": {
            name: statistics.mean(accs) for name, accs in per_scenario_accuracy.items()
        },
        "sd_accuracy_by_strategy": {
            name: statistics.stdev(accs) if len(accs) > 1 else 0.0 for name, accs in per_scenario_accuracy.items()
        },
        "mean_accuracy_by_strategy_and_category": {
            name: {cat: statistics.mean(accs) for cat, accs in cats.items()}
            for name, cats in per_scenario_category_acc.items()
        },
        "pairwise_accuracy_diffs": pairwise_diffs,
        "detector_scores": {
            name: {"precision": s.precision, "recall": s.recall, "f1": s.f1,
                   "mean_latency_s": s.mean_latency_s, "n_pairs": s.n_pairs}
            for name, s in detector_scores.items()
        },
        "three_way_llm_merge_variance": {
            "per_scenario_accuracy_across_3_repeats": three_way_variance,
            "note": "each list is 3 independent (use_cache=False) merge runs' downstream accuracy on the same scenario",
        },
        "llm_judge_detector_flip_rate": {
            "n_pairs_tested": len(llm_judge_variance_pairs),
            "n_pairs_with_disagreement_across_3_repeats": llm_judge_flip_count,
        },
        "token_usage_and_cost": {
            "n_cached_llm_calls_total": n_llm_calls,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "note": "Dollar cost intentionally omitted -- verify against current published "
                    "OpenAI pricing for the exact model rather than trusting a hardcoded rate.",
        },
    }

    (OUT_DIR / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", OUT_DIR / "pilot_summary.json")
    logger.info("Mean accuracy by strategy: %s", summary["mean_accuracy_by_strategy"])
    logger.info("Token usage: %d in / %d out across %d cached calls", total_input_tokens, total_output_tokens, n_llm_calls)


if __name__ == "__main__":
    main()
