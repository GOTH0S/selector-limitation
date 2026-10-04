from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metrics import evaluate_selection
from .population import CandidatePopulation
from .search import FamilyWorldSpec, SearchMode, SearchTrace, run_family_search
from .selectors import (
    FamilyBalancedSelector,
    HierarchicalShrinkageSelector,
    MeanRankSelector,
    Selector,
    SingleFoldWinner,
    StabilityPenaltySelector,
    ValidationWinner,
)

DEFAULT_BUDGETS = (16, 25, 50, 100, 250, 500, 1000)
DEFAULT_MODES: tuple[SearchMode, ...] = (
    "uniform",
    "winner_following",
)
DEFAULT_SELECTORS: tuple[Selector, ...] = (
    SingleFoldWinner(),
    ValidationWinner(),
    MeanRankSelector(),
    StabilityPenaltySelector(),
    HierarchicalShrinkageSelector(),
    FamilyBalancedSelector(),
)


@dataclass(frozen=True)
class SelectorTrialResult:
    mode: SearchMode
    selector: str
    budget: int
    selected_quality: float
    oracle_quality: float
    baseline_quality: float
    regret: float
    efficiency: float
    oracle_hit: bool
    selected_family_is_best: bool


def evaluate_selector(
    trace: SearchTrace,
    *,
    budget: int,
    selector: Selector,
) -> SelectorTrialResult:
    population = CandidatePopulation(
        true_quality=trace.population.true_quality[:budget],
        validation_scores=trace.population.validation_scores[:budget],
    )
    family_ids = trace.family_ids[:budget]
    selected_index = selector.select(population, family_ids=family_ids)
    outcome = evaluate_selection(population, selected_index)
    selected_family = int(family_ids[selected_index])
    best_family = int(trace.family_true_quality.argmax())

    return SelectorTrialResult(
        mode=trace.mode,
        selector=selector.name,
        budget=budget,
        selected_quality=outcome.selected_quality,
        oracle_quality=outcome.oracle_quality,
        baseline_quality=outcome.baseline_quality,
        regret=outcome.regret,
        efficiency=outcome.efficiency,
        oracle_hit=outcome.selected_index == outcome.oracle_index,
        selected_family_is_best=selected_family == best_family,
    )


def summarize_selector_trials(
    trials: list[SelectorTrialResult],
) -> dict[str, float]:
    if not trials:
        raise ValueError("trials cannot be empty")
    return {
        "mean_selected_quality": float(
            np.mean([trial.selected_quality for trial in trials])
        ),
        "mean_oracle_quality": float(
            np.mean([trial.oracle_quality for trial in trials])
        ),
        "mean_regret": float(np.mean([trial.regret for trial in trials])),
        "mean_efficiency": float(
            np.mean([trial.efficiency for trial in trials])
        ),
        "oracle_hit_rate": float(
            np.mean([trial.oracle_hit for trial in trials])
        ),
        "best_family_hit_rate": float(
            np.mean([trial.selected_family_is_best for trial in trials])
        ),
    }


def run_selector_sweep(
    *,
    seeds: int,
    spec: FamilyWorldSpec | None = None,
    selectors: tuple[Selector, ...] = DEFAULT_SELECTORS,
) -> list[dict[str, float | int | str]]:
    if seeds < 1:
        raise ValueError("seeds must be positive")
    spec = spec or FamilyWorldSpec(validation_folds=5)
    if spec.validation_folds < 2:
        raise ValueError(
            "selector comparison requires at least two validation folds"
        )

    trials: dict[
        tuple[SearchMode, str, int], list[SelectorTrialResult]
    ] = {
        (mode, selector.name, budget): []
        for mode in DEFAULT_MODES
        for selector in selectors
        for budget in DEFAULT_BUDGETS
    }
    max_budget = max(DEFAULT_BUDGETS)

    for mode in DEFAULT_MODES:
        for seed in range(seeds):
            trace = run_family_search(
                spec, budget=max_budget, mode=mode, seed=seed
            )
            for budget in DEFAULT_BUDGETS:
                for selector in selectors:
                    trials[(mode, selector.name, budget)].append(
                        evaluate_selector(
                            trace,
                            budget=budget,
                            selector=selector,
                        )
                    )

    rows: list[dict[str, float | int | str]] = []
    for mode in DEFAULT_MODES:
        for selector in selectors:
            for budget in DEFAULT_BUDGETS:
                rows.append(
                    {
                        "mode": mode,
                        "selector": selector.name,
                        "budget": budget,
                        "seeds": seeds,
                        "validation_folds": spec.validation_folds,
                        "family_validation_bias_scale": (
                            spec.family_validation_bias_scale
                        ),
                        "validation_noise": spec.validation_noise,
                        **summarize_selector_trials(
                            trials[(mode, selector.name, budget)]
                        ),
                    }
                )
    return rows


def write_csv(
    rows: list[dict[str, float | int | str]], out: Path
) -> None:
    if not rows:
        raise ValueError("rows cannot be empty")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare selectors on identical search traces"
    )
    parser.add_argument("--seeds", type=int, default=1000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/selector_sweep.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_csv(run_selector_sweep(seeds=args.seeds), args.out)


if __name__ == "__main__":
    main()
