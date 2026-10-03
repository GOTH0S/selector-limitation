from selector_limitation.search import FamilyWorldSpec, run_family_search
from selector_limitation.selector_experiment import (
    DEFAULT_SELECTORS,
    evaluate_selector,
    run_selector_sweep,
)


def test_all_default_selectors_return_valid_results() -> None:
    spec = FamilyWorldSpec(validation_folds=5)
    trace = run_family_search(spec, budget=100, mode="winner_following", seed=13)

    for selector in DEFAULT_SELECTORS:
        result = evaluate_selector(trace, budget=100, selector=selector)
        assert result.regret >= 0.0
        assert result.oracle_quality >= result.selected_quality


def test_selector_sweep_shape() -> None:
    rows = run_selector_sweep(seeds=2)
    assert len(rows) == 2 * len(DEFAULT_SELECTORS) * 7


def test_selector_sweep_rejects_single_fold_spec() -> None:
    try:
        run_selector_sweep(seeds=1, spec=FamilyWorldSpec(validation_folds=1))
    except ValueError as exc:
        assert "at least two" in str(exc)
    else:
        raise AssertionError("single-fold selector comparison should fail")
