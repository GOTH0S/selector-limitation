import numpy as np
import pytest

from selector_limitation.market_data import UNIVERSE, PricePanel, parse_prices


def test_parse_prices_enforces_cutoff() -> None:
    text = "Date," + ",".join(UNIVERSE) + "\n"
    text += "2019-12-31," + ",".join(["1"] * 8) + "\n"
    text += "2020-01-02," + ",".join(["2"] * 8) + "\n"
    panel = parse_prices(text, end_date="2019-12-31")
    assert panel.dates.tolist() == [np.datetime64("2019-12-31")]


def test_price_panel_rejects_nonpositive_prices() -> None:
    with pytest.raises(ValueError):
        PricePanel(
            dates=np.array(["2019-01-01"], dtype="datetime64[D]"),
            tickers=UNIVERSE,
            prices=np.zeros((1, 8)),
        )
