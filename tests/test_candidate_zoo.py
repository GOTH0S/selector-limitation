import numpy as np

from selector_limitation.candidate_zoo import (
    build_candidate_zoo,
    candidate_returns,
)
from selector_limitation.market_data import PricePanel, UNIVERSE


def toy_panel(rows: int = 320) -> PricePanel:
    dates = np.arange(
        np.datetime64("2010-01-01"),
        np.datetime64("2010-01-01") + rows,
    )
    t = np.arange(rows)[:, None]
    slopes = np.linspace(0.0002, 0.0010, 8)[None, :]
    prices = 100.0 * np.exp(t * slopes + 0.01 * np.sin(t / 11.0))
    return PricePanel(dates=dates, tickers=UNIVERSE, prices=prices)


def test_zoo_is_large_and_deterministic() -> None:
    left = build_candidate_zoo()
    right = build_candidate_zoo()
    assert left == right
    assert len(left) >= 2000
    assert len({spec.candidate_id for spec in left}) == len(left)


def test_candidate_returns_are_finite_and_aligned() -> None:
    panel = toy_panel()
    specs = build_candidate_zoo()[:20]
    returns = candidate_returns(panel, specs)
    assert returns.shape == (panel.dates.size, len(specs))
    assert np.isfinite(returns).all()


def test_future_price_change_does_not_change_earlier_pnl() -> None:
    panel = toy_panel()
    spec = build_candidate_zoo()[0:1]
    original = candidate_returns(panel, spec)
    changed_prices = panel.prices.copy()
    changed_prices[-1] *= 5.0
    changed = candidate_returns(
        PricePanel(panel.dates, panel.tickers, changed_prices),
        spec,
    )
    np.testing.assert_array_equal(original[:-2], changed[:-2])
