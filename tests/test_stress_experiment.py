import numpy as np
import pytest

from selector_limitation.market_data import UNIVERSE, PricePanel
from selector_limitation.stress_experiment import (
    PERIODS,
    UNIVERSES,
    _stable_argmax,
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
    assert len(UNIVERSES) == 13
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
