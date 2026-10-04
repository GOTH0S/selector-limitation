import numpy as np

from selector_limitation.metrics import evaluate_selection
from selector_limitation.population import CandidatePopulation, PopulationSpec, generate_population
from selector_limitation.selectors import ValidationWinner


def test_validation_winner_is_oracle_when_validation_is_exact() -> None:
    population = generate_population(
        PopulationSpec(n_candidates=100, validation_noise=0.0, validation_folds=2),
        seed=31,
    )
    selected = ValidationWinner().select(population)
    outcome = evaluate_selection(population, selected)

    assert outcome.selected_index == outcome.oracle_index
    assert outcome.regret == 0.0
    assert outcome.efficiency == 1.0


def test_regret_uses_latent_quality_not_validation_score() -> None:
    population = CandidatePopulation(
        true_quality=np.array([0.0, 2.0, 1.0]),
        validation_scores=np.array([[4.0], [1.0], [0.0]]),
    )
    selected = ValidationWinner().select(population)
    outcome = evaluate_selection(population, selected)

    assert selected == 0
    assert outcome.oracle_index == 1
    assert outcome.regret == 2.0
