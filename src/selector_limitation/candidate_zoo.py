from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from .market_data import PricePanel

FloatArray = NDArray[np.float64]

LOOKBACKS = (5, 10, 20, 40, 60, 120, 252)
VOL_WINDOWS = (10, 20, 40, 60)
STATES = ("all", "risk_on", "risk_off", "high_vol", "low_vol")
REBALANCE_DAYS = (1, 5, 20)
TRANSFORMS = ("sign", "linear")


@dataclass(frozen=True, order=True)
class CandidateSpec:
    family: str
    lookback: int
    secondary: int
    state: str
    rebalance_days: int
    transform: str
    complexity: int

    @property
    def candidate_id(self) -> str:
        return (
            f"{self.family}:l{self.lookback}:s{self.secondary}:"
            f"{self.state}:r{self.rebalance_days}:{self.transform}"
        )


def build_candidate_zoo() -> tuple[CandidateSpec, ...]:
    specs: list[CandidateSpec] = []

    def add(
        family: str,
        lookback: int,
        secondary: int,
        base_complexity: int,
    ) -> None:
        for state in STATES:
            for rebalance in REBALANCE_DAYS:
                for transform in TRANSFORMS:
                    specs.append(
                        CandidateSpec(
                            family=family,
                            lookback=lookback,
                            secondary=secondary,
                            state=state,
                            rebalance_days=rebalance,
                            transform=transform,
                            complexity=base_complexity
                            + (state != "all")
                            + (transform == "linear"),
                        )
                    )

    for lookback in LOOKBACKS:
        add("ts_momentum", lookback, 0, 2)
        add("ts_reversal", lookback, 0, 2)
        add("ma_gap", lookback, 0, 3)
        add("cs_momentum", lookback, 0, 3)
        add("cs_reversal", lookback, 0, 3)

    for lookback in (20, 40, 60, 120, 252):
        add("breakout", lookback, 0, 4)

    for lookback in LOOKBACKS:
        for vol_window in VOL_WINDOWS:
            add("vol_adjusted_momentum", lookback, vol_window, 4)

    for short, long in combinations(LOOKBACKS, 2):
        add("trend_acceleration", short, long, 4)

    return tuple(specs)


def _rolling_return(prices: FloatArray, lookback: int) -> FloatArray:
    out = np.full_like(prices, np.nan)
    out[lookback:] = prices[lookback:] / prices[:-lookback] - 1.0
    return out


def _rolling_mean(prices: FloatArray, window: int) -> FloatArray:
    out = np.full_like(prices, np.nan)
    csum = np.vstack([np.zeros((1, prices.shape[1])), np.cumsum(prices, axis=0)])
    out[window - 1 :] = (csum[window:] - csum[:-window]) / window
    return out


def _rolling_vol(simple_returns: FloatArray, window: int) -> FloatArray:
    out = np.full_like(simple_returns, np.nan)
    for end in range(window, simple_returns.shape[0] + 1):
        sample = simple_returns[end - window : end]
        out[end - 1] = np.std(sample, axis=0, ddof=1)
    return out


def _breakout_position(prices: FloatArray, window: int) -> FloatArray:
    out = np.full_like(prices, np.nan)
    for end in range(window, prices.shape[0] + 1):
        sample = prices[end - window : end]
        low = sample.min(axis=0)
        high = sample.max(axis=0)
        spread = high - low
        out[end - 1] = np.divide(
            2.0 * (prices[end - 1] - low),
            spread,
            out=np.zeros_like(spread),
            where=spread > 0,
        ) - 1.0
    return out


def _rank_cross_section(values: FloatArray) -> FloatArray:
    ranks = np.empty_like(values)
    for row in range(values.shape[0]):
        if not np.isfinite(values[row]).all():
            ranks[row] = np.nan
            continue
        order = np.argsort(values[row])
        raw = np.empty(values.shape[1], dtype=np.float64)
        raw[order] = np.arange(values.shape[1], dtype=np.float64)
        ranks[row] = raw - raw.mean()
    return ranks


def _state_mask(prices: FloatArray, daily_returns: FloatArray, state: str) -> FloatArray:
    if state == "all":
        return np.ones(prices.shape[0], dtype=np.float64)
    spy_60 = _rolling_return(prices[:, :1], 60)[:, 0]
    spy_vol_20 = _rolling_vol(daily_returns[:, :1], 20)[:, 0]
    spy_vol_120 = _rolling_vol(daily_returns[:, :1], 120)[:, 0]
    if state == "risk_on":
        return (spy_60 > 0).astype(np.float64)
    if state == "risk_off":
        return (spy_60 <= 0).astype(np.float64)
    if state == "high_vol":
        return (spy_vol_20 > spy_vol_120).astype(np.float64)
    if state == "low_vol":
        return (spy_vol_20 <= spy_vol_120).astype(np.float64)
    raise ValueError(f"unknown state: {state}")


def _raw_signal_for_key(
    family: str,
    lookback: int,
    secondary: int,
    prices: FloatArray,
    ret_cache: dict[int, FloatArray],
    mean_cache: dict[int, FloatArray],
    vol_cache: dict[int, FloatArray],
    breakout_cache: dict[int, FloatArray],
) -> FloatArray:
    lookback_return = ret_cache[lookback]
    if family == "ts_momentum":
        return lookback_return
    if family == "ts_reversal":
        return -lookback_return
    if family == "ma_gap":
        return prices / mean_cache[lookback] - 1.0
    if family == "breakout":
        return breakout_cache[lookback]
    if family == "cs_momentum":
        return _rank_cross_section(lookback_return)
    if family == "cs_reversal":
        return -_rank_cross_section(lookback_return)
    if family == "vol_adjusted_momentum":
        return np.divide(
            lookback_return,
            vol_cache[secondary],
            out=np.full_like(lookback_return, np.nan),
            where=vol_cache[secondary] > 0,
        )
    if family == "trend_acceleration":
        return ret_cache[lookback] - ret_cache[secondary]
    raise ValueError(f"unknown family: {family}")


def _apply_transform(raw: FloatArray, transform: str) -> FloatArray:
    if transform == "sign":
        return np.sign(raw)
    if transform == "linear":
        finite_rows = np.isfinite(raw).any(axis=1)
        scale = np.ones((raw.shape[0], 1), dtype=np.float64)
        scale[finite_rows] = np.nanmedian(
            np.abs(raw[finite_rows]), axis=1, keepdims=True
        )
        transformed = np.divide(
            raw, scale, out=np.zeros_like(raw), where=scale > 0
        )
        return np.clip(transformed, -3.0, 3.0)
    raise ValueError(f"unknown transform: {transform}")


def _normalize_weights(signal: FloatArray) -> FloatArray:
    clean = np.where(np.isfinite(signal), signal, 0.0)
    gross = np.abs(clean).sum(axis=1, keepdims=True)
    return np.divide(clean, gross, out=np.zeros_like(clean), where=gross > 0)


def _hold_between_rebalances(weights: FloatArray, rebalance_days: int) -> FloatArray:
    if rebalance_days == 1:
        return weights
    held = np.zeros_like(weights)
    current = np.zeros(weights.shape[1], dtype=np.float64)
    for row in range(weights.shape[0]):
        if row % rebalance_days == 0:
            current = weights[row]
        held[row] = current
    return held


def candidate_returns(
    panel: PricePanel,
    specs: tuple[CandidateSpec, ...],
    *,
    cost_bps: float = 5.0,
) -> FloatArray:
    prices = panel.prices
    daily_returns = np.zeros_like(prices)
    daily_returns[1:] = prices[1:] / prices[:-1] - 1.0

    needed_returns = sorted(
        {spec.lookback for spec in specs}
        | {spec.secondary for spec in specs if spec.secondary in LOOKBACKS}
    )
    needed_means = sorted(
        {spec.lookback for spec in specs if spec.family == "ma_gap"}
    )
    needed_vols = sorted(
        {spec.secondary for spec in specs if spec.family == "vol_adjusted_momentum"}
        | {20, 120}
    )
    needed_breakouts = sorted(
        {spec.lookback for spec in specs if spec.family == "breakout"}
    )

    ret_cache = {
        window: _rolling_return(prices, window) for window in needed_returns
    }
    mean_cache = {
        window: _rolling_mean(prices, window) for window in needed_means
    }
    vol_cache = {
        window: _rolling_vol(daily_returns, window) for window in needed_vols
    }
    breakout_cache = {
        window: _breakout_position(prices, window) for window in needed_breakouts
    }

    state_cache = {
        state: _state_mask(prices, daily_returns, state) for state in STATES
    }
    raw_cache: dict[tuple[str, int, int], FloatArray] = {}
    for spec in specs:
        key = (spec.family, spec.lookback, spec.secondary)
        if key not in raw_cache:
            raw_cache[key] = _raw_signal_for_key(
                spec.family,
                spec.lookback,
                spec.secondary,
                prices,
                ret_cache,
                mean_cache,
                vol_cache,
                breakout_cache,
            )

    output = np.zeros((prices.shape[0], len(specs)), dtype=np.float64)
    for column, spec in enumerate(specs):
        raw = raw_cache[(spec.family, spec.lookback, spec.secondary)]
        signal = _apply_transform(raw, spec.transform)
        signal = signal * state_cache[spec.state][:, None]
        weights = _hold_between_rebalances(
            _normalize_weights(signal), spec.rebalance_days
        )
        turnover = np.abs(
            weights - np.vstack([np.zeros((1, weights.shape[1])), weights[:-1]])
        ).sum(axis=1)
        pnl = np.zeros(prices.shape[0], dtype=np.float64)
        pnl[1:] = np.sum(weights[:-1] * daily_returns[1:], axis=1)
        pnl[1:] -= turnover[:-1] * (cost_bps / 10_000.0)
        output[:, column] = pnl
    return output
