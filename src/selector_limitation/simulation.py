from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import evaluate_selection
from .population import PopulationSpec, generate_population
from .selectors import ValidationWinner


@dataclass(frozen=True)
class TrialResult:
    candidate_count: int
    validation_noise: float
    oracle_quality: float
    selected_quality: float
    regret: float
    efficiency: float
    selected_is_oracle: bool


def run_trial(spec: PopulationSpec, *, seed: int) -> TrialResult:
    population = generate_population(spec, seed=seed)
    selector = ValidationWinner()
    selected_index = selector.select(population)
    outcome = evaluate_selection(population, selected_index)

    return TrialResult(
        candidate_count=spec.n_candidates,
        validation_noise=spec.validation_noise,
        oracle_quality=outcome.oracle_quality,
        selected_quality=outcome.selected_quality,
        regret=outcome.regret,
        efficiency=outcome.efficiency,
        selected_is_oracle=outcome.selected_index == outcome.oracle_index,
    )


def summarize_trials(trials: list[TrialResult]) -> dict[str, float]:
    if not trials:
        raise ValueError("trials cannot be empty")

    oracle = np.array([trial.oracle_quality for trial in trials], dtype=np.float64)
    selected = np.array([trial.selected_quality for trial in trials], dtype=np.float64)
    regret = np.array([trial.regret for trial in trials], dtype=np.float64)
    efficiency = np.array([trial.efficiency for trial in trials], dtype=np.float64)
    hit = np.array([trial.selected_is_oracle for trial in trials], dtype=np.float64)

    return {
        "mean_oracle_quality": float(oracle.mean()),
        "mean_selected_quality": float(selected.mean()),
        "mean_regret": float(regret.mean()),
        "median_regret": float(np.median(regret)),
        "p90_regret": float(np.quantile(regret, 0.90)),
        "mean_efficiency": float(efficiency.mean()),
        "oracle_hit_rate": float(hit.mean()),
    }
