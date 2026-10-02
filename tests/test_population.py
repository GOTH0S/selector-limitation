import numpy as np
import pytest

from selector_limitation.population import PopulationSpec, generate_population


def test_generation_is_deterministic() -> None:
    spec = PopulationSpec(n_candidates=20, validation_noise=0.5, validation_folds=3)
    left = generate_population(spec, seed=17)
    right = generate_population(spec, seed=17)

    np.testing.assert_array_equal(left.true_quality, right.true_quality)
    np.testing.assert_array_equal(left.validation_scores, right.validation_scores)


def test_zero_noise_exposes_true_quality() -> None:
    spec = PopulationSpec(n_candidates=20, validation_noise=0.0, validation_folds=4)
    population = generate_population(spec, seed=7)

    expected = np.repeat(population.true_quality[:, None], 4, axis=1)
    np.testing.assert_array_equal(population.validation_scores, expected)


def test_invalid_spec_rejected() -> None:
    with pytest.raises(ValueError):
        PopulationSpec(n_candidates=1, validation_noise=1.0)
