"""Basic plots for the Phase 6 results: mean downstream accuracy by strategy
with bootstrap CI error bars."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed to render to a file
import matplotlib.pyplot as plt

from branchmem.evaluation.runner import ScenarioRunResult


def plot_accuracy_by_strategy(results: list[ScenarioRunResult], out_path: Path) -> None:
    strategy_names = list(results[0].accuracy_by_strategy.keys())
    means = []
    errs = []
    for name in strategy_names:
        values = [r.accuracy_by_strategy[name] for r in results]
        mean = sum(values) / len(values)
        sd = (sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)) ** 0.5
        means.append(mean)
        errs.append(sd / (len(values) ** 0.5))  # standard error

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(strategy_names, means, yerr=errs, capsize=4, color="#4C72B0")
    ax.set_ylabel("Downstream accuracy")
    ax.set_title("Mean downstream accuracy by merge strategy (± SEM)")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
