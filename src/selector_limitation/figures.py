from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
FIGURES = Path("figures")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _frontier() -> None:
    rows = [
        row
        for row in read_rows(RESULTS / "synthetic_sweep.csv")
        if float(row["validation_noise"]) == 2.0
    ]
    x = np.array([int(row["candidate_count"]) for row in rows])
    oracle = np.array([float(row["mean_oracle_quality"]) for row in rows])
    selected = np.array([float(row["mean_selected_quality"]) for row in rows])

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, oracle, marker="o", label="reachable frontier")
    ax.plot(x, selected, marker="o", label="selected quality")
    ax.set_xscale("log")
    ax.set_xlabel("candidates searched")
    ax.set_ylabel("latent quality")
    ax.set_title("Search expands faster than selection")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "frontier.svg")
    plt.close(fig)


def _regret_phase() -> None:
    rows = [
        row
        for row in read_rows(RESULTS / "robustness_sweep.csv")
        if row["world"] == "heavy_tail"
    ]
    counts = sorted({int(row["candidate_count"]) for row in rows})
    noise = sorted({float(row["validation_noise"]) for row in rows})
    lookup = {
        (float(row["validation_noise"]), int(row["candidate_count"])): float(
            row["mean_regret"]
        )
        for row in rows
    }
    values = np.array([[lookup[(level, count)] for count in counts] for level in noise])

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    image = ax.imshow(values, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(counts)), [str(count) for count in counts])
    ax.set_yticks(np.arange(len(noise)), [str(level) for level in noise])
    ax.set_xlabel("candidates searched")
    ax.set_ylabel("validation noise")
    ax.set_title("Selection regret under heavy-tailed errors")
    fig.colorbar(image, ax=ax, label="mean regret")
    fig.tight_layout()
    fig.savefig(FIGURES / "regret_phase.svg")
    plt.close(fig)


def _search_policy() -> None:
    rows = read_rows(RESULTS / "family_search_sweep.csv")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for mode in sorted({row["mode"] for row in rows}):
        group = [row for row in rows if row["mode"] == mode]
        x = np.array([int(row["budget"]) for row in group])
        y = np.array([float(row["mean_oracle_quality"]) for row in group])
        ax.plot(x, y, marker="o", label=mode.replace("_", " "))

    ax.set_xscale("log")
    ax.set_xlabel("candidate budget")
    ax.set_ylabel("reachable quality")
    ax.set_title("Search policy changes the frontier")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "search_policy.svg")
    plt.close(fig)


def _market() -> None:
    rows = read_rows(RESULTS / "market_summary.csv")
    ensembles = {
        int(row["budget"]): float(row["mean_ensemble_test_sharpe"])
        for row in read_rows(RESULTS / "market_ensemble_summary.csv")
    }

    budgets = sorted({int(row["budget"]) for row in rows})
    by_selector = {
        selector: {
            int(row["budget"]): float(row["mean_selected_test_sharpe"])
            for row in rows
            if row["selector"] == selector
        }
        for selector in ("single_fold", "mean_fold")
    }
    oracle = {
        int(row["budget"]): float(row["mean_oracle_test_sharpe"])
        for row in rows
        if row["selector"] == "mean_fold"
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(budgets, [oracle[b] for b in budgets], marker="o", label="ex-post frontier")
    ax.plot(
        budgets,
        [by_selector["single_fold"][b] for b in budgets],
        marker="o",
        label="single fold",
    )
    ax.plot(
        budgets,
        [by_selector["mean_fold"][b] for b in budgets],
        marker="o",
        label="four-fold mean",
    )
    ax.plot(
        budgets,
        [ensembles[b] for b in budgets],
        marker="o",
        label="5-candidate ensemble",
    )
    ax.axhline(0, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("candidate budget")
    ax.set_ylabel("2018–2019 Sharpe")
    ax.set_title("Historical replication")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "market_replication.svg")
    plt.close(fig)


def render_all() -> None:
    FIGURES.mkdir(exist_ok=True)
    _frontier()
    _regret_phase()
    _search_policy()
    _market()


if __name__ == "__main__":
    render_all()
