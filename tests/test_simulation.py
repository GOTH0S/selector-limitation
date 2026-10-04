from selector_limitation.experiment import run_sweep
from selector_limitation.population import PopulationSpec
from selector_limitation.simulation import run_trial, summarize_trials


def test_trial_is_reproducible() -> None:
    spec = PopulationSpec(n_candidates=50, validation_noise=1.0)
    assert run_trial(spec, seed=99) == run_trial(spec, seed=99)


def test_summary_preserves_nonnegative_regret() -> None:
    spec = PopulationSpec(n_candidates=50, validation_noise=1.0)
    trials = [run_trial(spec, seed=seed) for seed in range(20)]
    summary = summarize_trials(trials)

    assert summary["mean_regret"] >= 0.0
    assert 0.0 <= summary["oracle_hit_rate"] <= 1.0


def test_sweep_shape() -> None:
    rows = run_sweep(seeds=2, validation_folds=1)
    assert len(rows) == 28
    assert {row["candidate_count"] for row in rows} == {10, 25, 50, 100, 250, 500, 1000}
