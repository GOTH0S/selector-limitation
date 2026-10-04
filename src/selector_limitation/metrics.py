from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .population import CandidatePopulation


@dataclass(frozen=True)
class SelectionOutcome:
    selected_index: int
    oracle_index: int
    selected_quality: float
    oracle_quality: float
    baseline_quality: float
    regret: float
    efficiency: float


def evaluate_selection(population: CandidatePopulation, selected_index: int) -> SelectionOutcome:
    if selected_index < 0 or selected_index >= population.n_candidates:
        raise IndexError("selected_index outside candidate population")

    oracle_index = int(population.true_quality.argmax())
    selected_quality = float(population.true_quality[selected_index])
    oracle_quality = float(population.true_quality[oracle_index])
    baseline_quality = float(population.true_quality.mean())
    regret = oracle_quality - selected_quality

    available_improvement = oracle_quality - baseline_quality
    if np.isclose(available_improvement, 0.0):
        efficiency = 1.0 if np.isclose(regret, 0.0) else 0.0
    else:
        efficiency = (selected_quality - baseline_quality) / available_improvement

    return SelectionOutcome(
        selected_index=selected_index,
        oracle_index=oracle_index,
        selected_quality=selected_quality,
        oracle_quality=oracle_quality,
        baseline_quality=baseline_quality,
        regret=regret,
        efficiency=float(efficiency),
    )
