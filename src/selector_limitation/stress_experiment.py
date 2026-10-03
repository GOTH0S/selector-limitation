from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .candidate_zoo import CandidateSpec, build_candidate_zoo, candidate_returns
from .market_data import SOURCE_COMMIT, SOURCE_SHA256, PricePanel, load_snapshot
from .market_experiment import (
    COST_BPS,
    MarketScores,
    _date_mask,
    _selected_metrics,
    annualized_sharpe,
    rank_correlation,
    selector_scores,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

STRESS_BUDGETS = (25, 100, 500, 1000)
SEARCH_SEEDS = 100
EXPECTED_PHASE4_ORACLE = 1.8589463473419494
EXPECTED_PHASE4_MEAN_FOLD = 0.5618176772530484

PHASE4_DEV_FOLDS = (
    ("2010-01-04", "2011-12-30"),
    ("2012-01-03", "2013-12-31"),
    ("2014-01-02", "2015-12-31"),
    ("2016-01-04", "2017-12-29"),
)
PHASE4_TEST = ("2018-01-02", "2019-12-31")


@dataclass(frozen=True)
class PeriodSpec:
    name: str
    development_folds: tuple[tuple[str, str], ...]
    test_period: tuple[str, str]


PERIODS = (
    PeriodSpec(
        "P12_13",
        (
            ("2008-01-02", "2009-12-31"),
            ("2010-01-04", "2011-12-30"),
        ),
        ("2012-01-03", "2013-12-31"),
    ),
    PeriodSpec(
        "P14_15",
        (
            ("2010-01-04", "2011-12-30"),
            ("2012-01-03", "2013-12-31"),
        ),
        ("2014-01-02", "2015-12-31"),
    ),
    PeriodSpec(
        "P16_17",
        (
            ("2012-01-03", "2013-12-31"),
            ("2014-01-02", "2015-12-31"),
        ),
        ("2016-01-04", "2017-12-29"),
    ),
    PeriodSpec(
        "P18_19",
        (
            ("2014-01-02", "2015-12-31"),
            ("2016-01-04", "2017-12-29"),
        ),
        ("2018-01-02", "2019-12-31"),
    ),
)

BASE_UNIVERSES: dict[str, tuple[str, ...]] = {
    "FULL8": ("SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "DBC"),
    "US3": ("SPY", "QQQ", "IWM"),
    "GLOBAL_EQ5": ("SPY", "QQQ", "IWM", "EFA", "EEM"),
    "CROSS_ASSET4": ("SPY", "TLT", "GLD", "DBC"),
    "EQ_BOND6": ("SPY", "QQQ", "IWM", "EFA", "EEM", "TLT"),
    "EQ_REAL7_LOO_TLT": ("SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "DBC"),
}

FULL8 = BASE_UNIVERSES["FULL8"]
UNIVERSES: dict[str, tuple[str, ...]] = dict(BASE_UNIVERSES)
for ticker in FULL8[1:]:
    candidate = tuple(x for x in FULL8 if x != ticker)
    if candidate not in UNIVERSES.values():
        UNIVERSES[f"LOO_{ticker}"] = candidate


def subset_panel(panel: PricePanel, tickers: tuple[str, ...]) -> PricePanel:
    if not tickers or tickers[0] != "SPY":
        raise ValueError("stress universes must retain SPY as the first asset")
    missing = set(tickers) - set(panel.tickers)
    if missing:
        raise ValueError(f"unknown tickers: {sorted(missing)}")
    indices = [panel.tickers.index(ticker) for ticker in tickers]
    return PricePanel(
        dates=panel.dates.copy(),
        tickers=tickers,
        prices=panel.prices[:, indices].copy(),
    )


def validate_period(period: PeriodSpec) -> None:
    if len(period.development_folds) < 2:
        raise ValueError("stress periods require at least two development folds")
    previous_end = None
    for start, end in period.development_folds:
        if start > end:
            raise ValueError(f"invalid development fold {start}..{end}")
        if previous_end is not None and start <= previous_end:
            raise ValueError("development folds overlap or are out of order")
        previous_end = end
    if previous_end is None or period.test_period[0] <= previous_end:
        raise ValueError("test period must begin after all development folds")
    if period.test_period[0] > period.test_period[1]:
        raise ValueError("invalid test period")


def score_period(
    panel: PricePanel,
    returns: FloatArray,
    period: PeriodSpec,
) -> MarketScores:
    validate_period(period)
    folds: list[FloatArray] = []
    dev_masks: list[NDArray[np.bool_]] = []
    for start, end in period.development_folds:
        mask = _date_mask(panel.dates, start, end)
        if mask.sum() < 250:
            raise ValueError(
                f"{period.name} development fold {start}..{end} is too short"
            )
        folds.append(annualized_sharpe(returns[mask]))
        dev_masks.append(mask)

    test_mask = _date_mask(panel.dates, *period.test_period)
    if test_mask.sum() < 250:
        raise ValueError(f"{period.name} test period is too short")
    dev_mask = np.logical_or.reduce(dev_masks)
    return MarketScores(
        fold_scores=np.stack(folds, axis=1),
        test_scores=annualized_sharpe(returns[test_mask]),
        development_returns=returns[dev_mask],
        test_returns=returns[test_mask],
    )




def _stable_argmax(indices: IntArray, scores: FloatArray) -> int:
    values = scores[indices]
    best = np.max(values)
    tied = indices[values == best]
    return int(np.min(tied))


def _stable_diverse_ensemble(
    subset: IntArray,
    mean_fold: FloatArray,
    development_returns: FloatArray,
    *,
    k: int = 5,
    correlation_penalty: float = 0.25,
) -> IntArray:
    order = np.lexsort((subset, -mean_fold[subset]))
    pool = subset[order][: min(100, subset.size)]
    chosen: list[int] = []
    for _ in range(min(k, pool.size)):
        best_index = None
        best_score = -np.inf
        for candidate in pool:
            idx = int(candidate)
            if idx in chosen:
                continue
            penalty = 0.0
            if chosen:
                corr = np.corrcoef(
                    development_returns[:, idx],
                    development_returns[:, chosen].mean(axis=1),
                )[0, 1]
                penalty = correlation_penalty * (
                    0.0 if np.isnan(corr) else corr
                )
            score = float(mean_fold[idx] - penalty)
            if (
                score > best_score
                or (
                    score == best_score
                    and (best_index is None or idx < best_index)
                )
            ):
                best_score = score
                best_index = idx
        if best_index is not None:
            chosen.append(best_index)
    return np.asarray(chosen, dtype=np.int64)


def _record(
    *,
    period: str,
    universe: str,
    seed: int,
    budget: int,
    selector: str,
    selected_quality: float,
    subset_test: FloatArray,
    rank_corr: float,
    selected_family: str,
    selected_complexity: int,
) -> dict[str, float | int | str]:
    regret, efficiency = _selected_metrics(selected_quality, subset_test)
    return {
        "period": period,
        "universe": universe,
        "search_seed": seed,
        "budget": budget,
        "selector": selector,
        "selected_test_sharpe": selected_quality,
        "oracle_test_sharpe": float(np.max(subset_test)),
        "mean_candidate_test_sharpe": float(np.mean(subset_test)),
        "selection_regret": regret,
        "frontier_efficiency": efficiency,
        "selected_positive": int(selected_quality > 0),
        "validation_test_rank_corr": rank_corr,
        "selected_family": selected_family,
        "selected_complexity": selected_complexity,
    }


def run_cell(
    period: PeriodSpec,
    universe_name: str,
    panel: PricePanel,
    specs: tuple[CandidateSpec, ...],
    returns: FloatArray,
    *,
    search_seeds: int,
) -> list[dict[str, float | int | str]]:
    scores = score_period(panel, returns, period)
    selections = selector_scores(scores, specs, include_bootstrap=False)
    mean_fold = selections["mean_fold"]
    candidate_count = len(specs)
    budgets = tuple(x for x in STRESS_BUDGETS if x < candidate_count) + (
        candidate_count,
    )

    rows: list[dict[str, float | int | str]] = []
    for seed in range(search_seeds):
        permutation = np.random.default_rng(seed).permutation(candidate_count)
        for budget in budgets:
            subset = permutation[:budget]
            subset_test = scores.test_scores[subset]
            rank_corr = rank_correlation(mean_fold[subset], subset_test)

            for selector, all_scores in selections.items():
                selected = _stable_argmax(subset, all_scores)
                rows.append(
                    _record(
                        period=period.name,
                        universe=universe_name,
                        seed=seed,
                        budget=budget,
                        selector=selector,
                        selected_quality=float(scores.test_scores[selected]),
                        subset_test=subset_test,
                        rank_corr=rank_corr,
                        selected_family=specs[selected].family,
                        selected_complexity=specs[selected].complexity,
                    )
                )

            members = _stable_diverse_ensemble(
                subset,
                mean_fold,
                scores.development_returns,
            )
            ensemble_returns = scores.test_returns[:, members].mean(axis=1)
            ensemble_quality = float(
                annualized_sharpe(ensemble_returns[:, None])[0]
            )
            rows.append(
                _record(
                    period=period.name,
                    universe=universe_name,
                    seed=seed,
                    budget=budget,
                    selector="diverse_ensemble",
                    selected_quality=ensemble_quality,
                    subset_test=subset_test,
                    rank_corr=rank_corr,
                    selected_family="ensemble",
                    selected_complexity=-1,
                )
            )
    return rows


def summarize(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    groups: dict[
        tuple[str, str, int, str],
        list[dict[str, float | int | str]],
    ] = {}
    for row in rows:
        key = (
            str(row["period"]),
            str(row["universe"]),
            int(row["budget"]),
            str(row["selector"]),
        )
        groups.setdefault(key, []).append(row)

    output: list[dict[str, float | int | str]] = []
    for key, group in sorted(groups.items()):
        period, universe, budget, selector = key

        output.append(
            {
                "period": period,
                "universe": universe,
                "budget": budget,
                "selector": selector,
                "search_seeds": len(group),
                "mean_selected_test_sharpe": float(
                    np.mean(
                        [float(row["selected_test_sharpe"]) for row in group]
                    )
                ),
                "mean_oracle_test_sharpe": float(
                    np.mean([float(row["oracle_test_sharpe"]) for row in group])
                ),
                "mean_candidate_test_sharpe": float(
                    np.mean(
                        [
                            float(row["mean_candidate_test_sharpe"])
                            for row in group
                        ]
                    )
                ),
                "mean_selection_regret": float(
                    np.mean([float(row["selection_regret"]) for row in group])
                ),
                "mean_frontier_efficiency": float(
                    np.mean(
                        [float(row["frontier_efficiency"]) for row in group]
                    )
                ),
                "selected_positive_rate": float(
                    np.mean([float(row["selected_positive"]) for row in group])
                ),
                "mean_validation_test_rank_corr": float(
                    np.mean(
                        [
                            float(row["validation_test_rank_corr"])
                            for row in group
                        ]
                    )
                ),
            }
        )
    return output


def failure_status(oracle: float, selected: float, efficiency: float) -> str:
    if oracle <= 0:
        return "NO_POSITIVE_FRONTIER"
    if selected <= 0:
        return "NEGATIVE_SELECTION"
    if efficiency < 0:
        return "INVERTED"
    if efficiency < 0.25:
        return "WEAK_CAPTURE"
    if efficiency < 0.50:
        return "PARTIAL_CAPTURE"
    return "STRONG_CAPTURE"


def build_failure_map(
    summary: list[dict[str, float | int | str]],
    full_budget: int,
) -> list[dict[str, float | int | str]]:
    rows = []
    for row in summary:
        if int(row["budget"]) != full_budget:
            continue
        oracle = float(row["mean_oracle_test_sharpe"])
        selected = float(row["mean_selected_test_sharpe"])
        efficiency = float(row["mean_frontier_efficiency"])
        rank_corr = float(row["mean_validation_test_rank_corr"])
        rows.append(
            {
                "period": row["period"],
                "universe": row["universe"],
                "selector": row["selector"],
                "selected_test_sharpe": selected,
                "oracle_test_sharpe": oracle,
                "selection_regret": float(row["mean_selection_regret"]),
                "frontier_efficiency": efficiency,
                "validation_test_rank_corr": rank_corr,
                "low_identifiability": int(abs(rank_corr) < 0.10),
                "status": failure_status(oracle, selected, efficiency),
            }
        )
    return rows


def summarize_selectors(
    failure_map: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    groups: dict[str, list[dict[str, float | int | str]]] = {}
    for row in failure_map:
        groups.setdefault(str(row["selector"]), []).append(row)

    output = []
    for selector, group in sorted(groups.items()):
        selected = np.array(
            [float(row["selected_test_sharpe"]) for row in group]
        )
        regret = np.array([float(row["selection_regret"]) for row in group])
        efficiency = np.array(
            [float(row["frontier_efficiency"]) for row in group]
        )
        statuses = [str(row["status"]) for row in group]
        output.append(
            {
                "selector": selector,
                "cells": len(group),
                "mean_selected_test_sharpe": float(selected.mean()),
                "median_selected_test_sharpe": float(np.median(selected)),
                "positive_cell_rate": float(np.mean(selected > 0)),
                "negative_selection_rate": (
                    statuses.count("NEGATIVE_SELECTION") / len(statuses)
                ),
                "inverted_rate": statuses.count("INVERTED") / len(statuses),
                "material_capture_rate": float(np.mean(efficiency >= 0.25)),
                "strong_capture_rate": float(np.mean(efficiency >= 0.50)),
                "median_selection_regret": float(np.median(regret)),
                "median_frontier_efficiency": float(np.median(efficiency)),
            }
        )
    return output


def bootstrap_period_audit(
    panel: PricePanel,
    specs: tuple[CandidateSpec, ...],
    returns: FloatArray,
) -> list[dict[str, float | int | str]]:
    rows = []
    for period in PERIODS:
        scores = score_period(panel, returns, period)
        selection = selector_scores(scores, specs, include_bootstrap=True)[
            "bootstrap_lcb"
        ]
        selected = int(np.argmax(selection))
        selected_quality = float(scores.test_scores[selected])
        regret, efficiency = _selected_metrics(
            selected_quality, scores.test_scores
        )
        rows.append(
            {
                "period": period.name,
                "universe": "FULL8",
                "selector": "bootstrap_lcb",
                "selected_test_sharpe": selected_quality,
                "oracle_test_sharpe": float(scores.test_scores.max()),
                "selection_regret": regret,
                "frontier_efficiency": efficiency,
                "selected_positive": int(selected_quality > 0),
                "validation_test_rank_corr": rank_correlation(
                    scores.fold_scores.mean(axis=1),
                    scores.test_scores,
                ),
                "selected_family": specs[selected].family,
            }
        )
    return rows


def phase4_reproduction(
    panel: PricePanel,
    specs: tuple[CandidateSpec, ...],
    returns: FloatArray,
) -> dict[str, float]:
    period = PeriodSpec(
        "PHASE4_CONTROL",
        PHASE4_DEV_FOLDS,
        PHASE4_TEST,
    )
    scores = score_period(panel, returns, period)
    selection = selector_scores(scores, specs, include_bootstrap=False)
    selected = int(np.argmax(selection["mean_fold"]))
    selected_quality = float(scores.test_scores[selected])
    oracle = float(scores.test_scores.max())
    if not np.isclose(oracle, EXPECTED_PHASE4_ORACLE, atol=1e-12, rtol=0):
        raise AssertionError(
            f"Phase 4 oracle drifted: {oracle} != {EXPECTED_PHASE4_ORACLE}"
        )
    if not np.isclose(
        selected_quality,
        EXPECTED_PHASE4_MEAN_FOLD,
        atol=1e-12,
        rtol=0,
    ):
        raise AssertionError(
            "Phase 4 mean-fold selection drifted: "
            f"{selected_quality} != {EXPECTED_PHASE4_MEAN_FOLD}"
        )
    return {
        "oracle_test_sharpe": oracle,
        "mean_fold_selected_test_sharpe": selected_quality,
    }


def _write_csv(
    rows: list[dict[str, float | int | str]], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run period/universe selector stress tests"
    )
    parser.add_argument(
        "--cache", type=Path, default=Path(".cache/prices.csv")
    )
    parser.add_argument(
        "--ledger-out",
        type=Path,
        default=Path("results/stress_ledger.csv"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("results/stress_summary.csv"),
    )
    parser.add_argument(
        "--failure-map-out",
        type=Path,
        default=Path("results/failure_map.csv"),
    )
    parser.add_argument(
        "--selector-summary-out",
        type=Path,
        default=Path("results/stress_selector_summary.csv"),
    )
    parser.add_argument(
        "--bootstrap-out",
        type=Path,
        default=Path("results/bootstrap_period_stress.csv"),
    )
    parser.add_argument(
        "--run-out",
        type=Path,
        default=Path("results/stress_run.json"),
    )
    parser.add_argument("--search-seeds", type=int, default=SEARCH_SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_panel = load_snapshot(args.cache, end_date="2019-12-31")
    specs = build_candidate_zoo()

    full_returns = candidate_returns(full_panel, specs, cost_bps=COST_BPS)
    benchmark = phase4_reproduction(full_panel, specs, full_returns)
    bootstrap_rows = bootstrap_period_audit(full_panel, specs, full_returns)

    rows: list[dict[str, float | int | str]] = []
    for universe_name, tickers in UNIVERSES.items():
        panel = subset_panel(full_panel, tickers)
        if universe_name == "FULL8":
            returns = full_returns
        else:
            returns = candidate_returns(panel, specs, cost_bps=COST_BPS)
        for period in PERIODS:
            rows.extend(
                run_cell(
                    period,
                    universe_name,
                    panel,
                    specs,
                    returns,
                    search_seeds=args.search_seeds,
                )
            )

    summary = summarize(rows)
    failure_map = build_failure_map(summary, len(specs))
    selector_summary = summarize_selectors(failure_map)

    _write_csv(rows, args.ledger_out)
    _write_csv(summary, args.summary_out)
    _write_csv(failure_map, args.failure_map_out)
    _write_csv(selector_summary, args.selector_summary_out)
    _write_csv(bootstrap_rows, args.bootstrap_out)

    payload = {
        "data_source": {
            "repository": "marcoreyess22/jump-risk-engine",
            "commit": SOURCE_COMMIT,
            "path": "data/prices.csv",
            "sha256": SOURCE_SHA256,
        },
        "candidate_count": len(specs),
        "cost_bps": COST_BPS,
        "search_seeds": args.search_seeds,
        "budgets": [25, 100, 500, 1000, len(specs)],
        "periods": [
            {
                "name": period.name,
                "development_folds": period.development_folds,
                "test_period": period.test_period,
            }
            for period in PERIODS
        ],
        "universes": {name: list(tickers) for name, tickers in UNIVERSES.items()},
        "phase4_reproduction": benchmark,
        "failure_bins": {
            "low_identifiability_abs_rank_corr_lt": 0.10,
            "weak_capture_efficiency_lt": 0.25,
            "strong_capture_efficiency_gte": 0.50,
        },
    }
    args.run_out.parent.mkdir(parents=True, exist_ok=True)
    args.run_out.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print("STRESS_RUN_JSON=" + json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()
