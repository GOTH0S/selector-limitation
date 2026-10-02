import numpy as np
import pytest

from selector_limitation.family_experiment import (
    evaluate_family_trace,
    run_family_sweep,
    run_family_trial,
)
from selector_limitation.search import FamilyWorldSpec, run_family_search


def test_family_search_is_deterministic() -> None:
    spec = FamilyWorldSpec()
    left = run_family_search(spec, budget=100, mode="winner_following", seed=17)
    right = run_family_search(spec, budget=100, mode="winner_following", seed=17)

    np.testing.assert_array_equal(left.population.true_quality, right.population.true_quality)
    np.testing.assert_array_equal(
        left.population.validation_scores, right.population.validation_scores
    )
    np.testing.assert_array_equal(left.family_ids, right.family_ids)


def test_initial_budget_samples_every_family_equally() -> None:
    spec = FamilyWorldSpec(n_families=5, initial_per_family=3)
    trace = run_family_search(spec, budget=15, mode="winner_following", seed=3)

    np.testing.assert_array_equal(trace.family_counts, np.full(5, 3))
    assert trace.allocation_hhi == pytest.approx(0.2)
    assert trace.effective_family_count == pytest.approx(5.0)


def test_full_exploitation_concentrates_after_seed_round() -> None:
    spec = FamilyWorldSpec(
        n_families=4,
        initial_per_family=1,
        exploit_probability=1.0,
    )
    trace = run_family_search(spec, budget=20, mode="winner_following", seed=11)

    assert trace.family_counts.max() == 17
    assert np.count_nonzero(trace.family_counts == 1) == 3


def test_budget_below_seed_round_is_rejected() -> None:
    spec = FamilyWorldSpec(n_families=8, initial_per_family=2)
    with pytest.raises(ValueError):
        run_family_search(spec, budget=15, mode="uniform", seed=0)


def test_family_trial_metrics_are_bounded() -> None:
    trial = run_family_trial(
        FamilyWorldSpec(), budget=100, mode="winner_following", seed=29
    )

    assert trial.regret >= 0.0
    assert 0.0 < trial.allocation_hhi <= 1.0
    assert 1.0 <= trial.effective_family_count <= 8.0


def test_family_sweep_shape() -> None:
    rows = run_family_sweep(seeds=2)
    assert len(rows) == 14
    assert {row["mode"] for row in rows} == {"uniform", "winner_following"}


def test_prefix_evaluation_matches_standalone_run() -> None:
    spec = FamilyWorldSpec()
    long_trace = run_family_search(spec, budget=1000, mode="winner_following", seed=41)
    prefix = run_family_trial(spec, budget=100, mode="winner_following", seed=41)

    assert evaluate_family_trace(long_trace, budget=100) == prefix
