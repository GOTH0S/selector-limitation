from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metrics import evaluate_selection
from .population import CandidatePopulation
from .search import FamilyWorldSpec, SearchMode, SearchTrace, run_family_search
from .selectors import ValidationWinner

DEFAULT_BUDGETS = (16, 25, 50, 100, 250, 500, 1000)
DEFAULT_MODES: tuple[SearchMode, ...] = ("uniform", "winner_following")


@dataclass(frozen=True)
class FamilyTrialResult:
    mode: SearchMode
    budget: int
    selected_quality: float
    oracle_quality: float
    regret: float
    oracle_hit: bool
    selected_family_is_best: bool
    selected_family_true_mean: float
    best_family_true_mean: float
    allocation_hhi: float
    effective_family_count: float


def evaluate_family_trace(trace: SearchTrace, *, budget: int) -> FamilyTrialResult:
    if budget < 1 or budget > trace.population.n_candidates:
        raise ValueError("budget outside search trace")

    population = CandidatePopulation(
        true_quality=trace.population.true_quality[:budget],
        validation_scores=trace.population.validation_scores[:budget],
    )
    selected_index = ValidationWinner().select(population)
    outcome = evaluate_selection(population, selected_index)

    family_ids = trace.family_ids[:budget]
    family_counts = np.bincount(
        family_ids, minlength=trace.family_true_quality.shape[0]
    )
    weights = family_counts / budget
    allocation_hhi = float(np.square(weights).sum())

    selected_family = int(family_ids[selected_index])
    best_family = int(trace.family_true_quality.argmax())

    return FamilyTrialResult(
        mode=trace.mode,
        budget=budget,
        selected_quality=outcome.selected_quality,
        oracle_quality=outcome.oracle_quality,
        regret=outcome.regret,
        oracle_hit=outcome.selected_index == outcome.oracle_index,
        selected_family_is_best=selected_family == best_family,
        selected_family_true_mean=float(trace.family_true_quality[selected_family]),
        best_family_true_mean=float(trace.family_true_quality[best_family]),
        allocation_hhi=allocation_hhi,
        effective_family_count=1.0 / allocation_hhi,
    )


def run_family_trial(
    spec: FamilyWorldSpec,
    *,
    budget: int,
    mode: SearchMode,
    seed: int,
) -> FamilyTrialResult:
    trace = run_family_search(spec, budget=budget, mode=mode, seed=seed)
    return evaluate_family_trace(trace, budget=budget)


def summarize_family_trials(trials: list[FamilyTrialResult]) -> dict[str, float]:
    if not trials:
        raise ValueError("trials cannot be empty")

    fields = (
        "selected_quality",
        "oracle_quality",
        "regret",
        "oracle_hit",
        "selected_family_is_best",
        "selected_family_true_mean",
        "best_family_true_mean",
        "allocation_hhi",
        "effective_family_count",
    )
    return {
        f"mean_{field}": float(np.mean([getattr(trial, field) for trial in trials]))
        for field in fields
    }


def run_family_sweep(
    *,
    seeds: int,
    spec: FamilyWorldSpec | None = None,
) -> list[dict[str, float | int | str]]:
    if seeds < 1:
        raise ValueError("seeds must be positive")
    spec = spec or FamilyWorldSpec()

    trials_by_cell: dict[tuple[SearchMode, int], list[FamilyTrialResult]] = {
        (mode, budget): [] for mode in DEFAULT_MODES for budget in DEFAULT_BUDGETS
    }
    max_budget = max(DEFAULT_BUDGETS)

    for mode in DEFAULT_MODES:
        for seed in range(seeds):
            trace = run_family_search(spec, budget=max_budget, mode=mode, seed=seed)
            for budget in DEFAULT_BUDGETS:
                trials_by_cell[(mode, budget)].append(
                    evaluate_family_trace(trace, budget=budget)
                )

    rows: list[dict[str, float | int | str]] = []
    for mode in DEFAULT_MODES:
        for budget in DEFAULT_BUDGETS:
            rows.append(
                {
                    "mode": mode,
                    "budget": budget,
                    "seeds": seeds,
                    "n_families": spec.n_families,
                    "family_validation_bias_scale": spec.family_validation_bias_scale,
                    "validation_noise": spec.validation_noise,
                    "exploit_probability": spec.exploit_probability,
                    **summarize_family_trials(trials_by_cell[(mode, budget)]),
                }
            )
    return rows


def write_csv(rows: list[dict[str, float | int | str]], out: Path) -> None:
    if not rows:
        raise ValueError("rows cannot be empty")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run correlated-family search experiments")
    parser.add_argument("--seeds", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("results/family_search_sweep.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_csv(run_family_sweep(seeds=args.seeds), args.out)


if __name__ == "__main__":
    main()
