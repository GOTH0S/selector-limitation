import numpy as np

from selector_limitation.robustness_experiment import generate_world


def test_robustness_worlds_are_deterministic() -> None:
    worlds = ("gaussian", "heavy_tail", "family_bias", "regime_shift")
    for world in worlds:
        left = generate_world(
            world=world,
            n_candidates=50,
            validation_noise=1.0,
            seed=17,
        )
        right = generate_world(
            world=world,
            n_candidates=50,
            validation_noise=1.0,
            seed=17,
        )
        np.testing.assert_array_equal(left.true_quality, right.true_quality)
        np.testing.assert_array_equal(left.validation_scores, right.validation_scores)


def test_regime_shift_changes_validation_ordering() -> None:
    gaussian = generate_world(
        world="gaussian",
        n_candidates=100,
        validation_noise=0.0,
        seed=9,
    )
    shifted = generate_world(
        world="regime_shift",
        n_candidates=100,
        validation_noise=0.0,
        seed=9,
    )
    assert int(gaussian.validation_mean.argmax()) != int(
        shifted.validation_mean.argmax()
    )
