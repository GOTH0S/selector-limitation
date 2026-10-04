import numpy as np
import pytest

from selector_limitation.market_data import UNIVERSE, PricePanel
from selector_limitation.stress_experiment import (
    PERIODS,
    UNIVERSES,
    _stable_argmax,
    build_failure_map,
    classify_expansion,
    failure_status,
    subset_panel,
    validate_period,
)


def toy_panel() -> PricePanel:
    dates = np.arange(
        np.datetime64("2007-01-01"),
        np.datetime64("2007-01-11"),
    )
    prices = np.arange(80, dtype=float).reshape(10, 8) + 100.0
    return PricePanel(dates=dates, tickers=UNIVERSE, prices=prices)


def test_all_stress_universes_keep_spy_first() -> None:
    assert len(UNIVERSES) == 12
    assert all(tickers[0] == "SPY" for tickers in UNIVERSES.values())


def test_subset_panel_preserves_requested_order() -> None:
    panel = subset_panel(toy_panel(), ("SPY", "TLT", "GLD"))
    assert panel.tickers == ("SPY", "TLT", "GLD")
    np.testing.assert_array_equal(panel.prices[:, 0], toy_panel().prices[:, 0])
    np.testing.assert_array_equal(panel.prices[:, 1], toy_panel().prices[:, 5])


def test_subset_panel_rejects_changed_state_reference() -> None:
    with pytest.raises(ValueError):
        subset_panel(toy_panel(), ("QQQ", "SPY"))


def test_periods_are_chronological_and_nonoverlapping() -> None:
    for period in PERIODS:
        validate_period(period)


def test_failure_bins_are_deterministic() -> None:
    assert failure_status(-0.1, -0.2, -1.0) == "NO_POSITIVE_FRONTIER"
    assert failure_status(1.0, -0.1, -0.2) == "NEGATIVE_SELECTION"
    assert failure_status(1.0, 0.1, -0.01) == "INVERTED"
    assert failure_status(1.0, 0.1, 0.10) == "WEAK_CAPTURE"
    assert failure_status(1.0, 0.3, 0.30) == "PARTIAL_CAPTURE"
    assert failure_status(1.0, 0.6, 0.60) == "STRONG_CAPTURE"


def test_stable_argmax_ignores_subset_order_for_ties() -> None:
    scores = np.array([0.0, 2.0, 2.0, 1.0])
    left = np.array([2, 1, 3], dtype=np.int64)
    right = np.array([1, 3, 2], dtype=np.int64)
    assert _stable_argmax(left, scores) == 1
    assert _stable_argmax(right, scores) == 1


def test_stress_universes_are_unique_asset_sets() -> None:
    assert len(set(UNIVERSES.values())) == len(UNIVERSES)


def test_expansion_classification_is_sign_based() -> None:
    assert classify_expansion(1.0, -0.1) == "SELECTOR_LIMITED"
    assert classify_expansion(1.0, 0.1) == "FRONTIER_AND_SELECTION_UP"
    assert classify_expansion(-1.0, -0.1) == "BROAD_DETERIORATION"
    assert classify_expansion(0.0, -0.1) == "FLAT_FRONTIER_SELECTION_DOWN"


def test_failure_map_excludes_ensemble_mitigation() -> None:
    common = {
        "period": "P18_19",
        "universe": "FULL8",
        "budget": 2670,
        "search_seeds": 100,
        "mean_oracle_test_sharpe": 1.8,
        "mean_candidate_test_sharpe": 0.0,
        "mean_selection_regret": 1.2,
        "mean_frontier_efficiency": 0.3,
        "selected_positive_rate": 1.0,
        "mean_validation_test_rank_corr": 0.05,
    }
    rows = [
        {
            **common,
            "selector": "mean_fold",
            "mean_selected_test_sharpe": 0.6,
        },
        {
            **common,
            "selector": "diverse_ensemble",
            "mean_selected_test_sharpe": 0.8,
        },
    ]
    result = build_failure_map(rows, 2670)
    assert [row["selector"] for row in result] == ["mean_fold"]
