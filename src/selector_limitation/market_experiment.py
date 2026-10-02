from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .candidate_zoo import CandidateSpec, build_candidate_zoo, candidate_returns
from .market_data import PricePanel, load_snapshot

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

BUDGETS = (25, 50, 100, 250, 500, 1000, 2000)
DEV_FOLDS = (
    ("2010-01-04", "2011-12-30"),
    ("2012-01-03", "2013-12-31"),
    ("2014-01-02", "2015-12-31"),
    ("2016-01-04", "2017-12-29"),
)
TEST_PERIOD = ("2018-01-02", "2019-12-31")
PENALTY_GRID = (0.0, 0.025, 0.05, 0.10, 0.20)


@dataclass(frozen=True)
class MarketScores:
    fold_scores: FloatArray
    test_scores: FloatArray
    development_returns: FloatArray
    test_returns: FloatArray


def _date_mask(
    dates: NDArray[np.datetime64], start: str, end: str
) -> NDArray[np.bool_]:
    return (dates >= np.datetime64(start, "D")) & (
        dates <= np.datetime64(end, "D")
    )


def annualized_sharpe(returns: FloatArray) -> FloatArray:
    mean = np.mean(returns, axis=0)
    std = np.std(returns, axis=0, ddof=1)
    return np.divide(
        np.sqrt(252.0) * mean,
        std,
        out=np.zeros_like(mean, dtype=np.float64),
        where=std > 0,
    )


def score_market(panel: PricePanel, returns: FloatArray) -> MarketScores:
    folds = []
    dev_masks = []
    for start, end in DEV_FOLDS:
        mask = _date_mask(panel.dates, start, end)
        if mask.sum() < 250:
            raise ValueError(f"development fold {start}..{end} is too short")
        folds.append(annualized_sharpe(returns[mask]))
        dev_masks.append(mask)

    test_mask = _date_mask(panel.dates, *TEST_PERIOD)
    if test_mask.sum() < 250:
        raise ValueError("test period is too short")
    dev_mask = np.logical_or.reduce(dev_masks)
    return MarketScores(
        fold_scores=np.stack(folds, axis=1),
        test_scores=annualized_sharpe(returns[test_mask]),
        development_returns=returns[dev_mask],
        test_returns=returns[test_mask],
    )


def _ranks(values: FloatArray) -> FloatArray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def rank_correlation(left: FloatArray, right: FloatArray) -> float:
    if left.size < 2:
        return 0.0
    return float(np.corrcoef(_ranks(left), _ranks(right))[0, 1])


def _mean_rank(fold_scores: FloatArray) -> FloatArray:
    ranks = np.empty_like(fold_scores)
    for fold in range(fold_scores.shape[1]):
        order = np.argsort(-fold_scores[:, fold])
        ranks[order, fold] = np.arange(
            fold_scores.shape[0], dtype=np.float64
        )
    return -ranks.mean(axis=1)


def _normalized_complexity(complexity: FloatArray) -> FloatArray:
    spread = complexity.max() - complexity.min()
    if spread == 0:
        return np.zeros_like(complexity)
    return (complexity - complexity.min()) / spread


def _nested_complexity_penalty(
    fold_scores: FloatArray, complexity: FloatArray
) -> float:
    normalized = _normalized_complexity(complexity)
    validation_by_penalty = []
    for penalty in PENALTY_GRID:
        forward_scores = []
        for heldout in range(1, fold_scores.shape[1]):
            train_mean = fold_scores[:, :heldout].mean(axis=1)
            selected = int(np.argmax(train_mean - penalty * normalized))
            forward_scores.append(float(fold_scores[selected, heldout]))
        validation_by_penalty.append(float(np.mean(forward_scores)))
    return PENALTY_GRID[int(np.argmax(validation_by_penalty))]


def _bootstrap_lcb(
    development_returns: FloatArray,
    *,
    seed: int = 20261002,
    block: int = 20,
    draws: int = 128,
) -> FloatArray:
    rng = np.random.default_rng(seed)
    n_rows, n_candidates = development_returns.shape
    means = np.empty((draws, n_candidates), dtype=np.float64)
    blocks_needed = int(np.ceil(n_rows / block))
    max_start = n_rows - block
    for draw in range(draws):
        starts = rng.integers(0, max_start + 1, size=blocks_needed)
        sample = np.concatenate(
            [
                development_returns[start : start + block]
                for start in starts
            ],
            axis=0,
        )[:n_rows]
        means[draw] = sample.mean(axis=0)
    return np.quantile(means, 0.10, axis=0) * 252.0


def _family_balanced_score(
    fold_scores: FloatArray, family_ids: NDArray[np.str_]
) -> FloatArray:
    mean_score = fold_scores.mean(axis=1)
    result = np.full(mean_score.shape, -np.inf, dtype=np.float64)
    for family in np.unique(family_ids):
        mask = family_ids == family
        family_median = float(np.median(mean_score[mask]))
        indices = np.flatnonzero(mask)
        result[indices] = family_median + 1e-9 * mean_score[indices]
    return result


def selector_scores(
    scores: MarketScores,
    specs: tuple[CandidateSpec, ...],
) -> dict[str, FloatArray]:
    folds = scores.fold_scores
    complexity = np.array(
        [spec.complexity for spec in specs], dtype=np.float64
    )
    families = np.array([spec.family for spec in specs])
    normalized_complexity = _normalized_complexity(complexity)
    nested_penalty = _nested_complexity_penalty(folds, complexity)

    standard_error = folds.std(axis=1, ddof=1) / np.sqrt(folds.shape[1])
    return {
        "single_fold": folds[:, 0],
        "mean_fold": folds.mean(axis=1),
        "mean_rank": _mean_rank(folds),
        "stability_lcb": folds.mean(axis=1) - standard_error,
        "family_balanced": _family_balanced_score(folds, families),
        "nested_complexity": (
            folds.mean(axis=1) - nested_penalty * normalized_complexity
        ),
        "bootstrap_lcb": _bootstrap_lcb(scores.development_returns),
    }


def _greedy_diverse_ensemble(
    subset: IntArray,
    mean_fold: FloatArray,
    development_returns: FloatArray,
    *,
    k: int = 5,
    correlation_penalty: float = 0.25,
) -> IntArray:
    ranked = subset[np.argsort(-mean_fold[subset])]
    pool = ranked[: min(100, ranked.size)]
    chosen: list[int] = []
    for _ in range(min(k, pool.size)):
        best = None
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
            if score > best_score:
                best_score = score
                best = idx
        if best is not None:
            chosen.append(best)
    return np.asarray(chosen, dtype=np.int64)


def _selected_metrics(
    selected_quality: float,
    subset_test: FloatArray,
) -> tuple[float, float]:
    oracle = float(np.max(subset_test))
    baseline = float(np.mean(subset_test))
    regret = oracle - selected_quality
    denominator = oracle - baseline
    efficiency = (
        1.0
        if denominator == 0
        else (selected_quality - baseline) / denominator
    )
    return regret, efficiency


def run_market_sweep(
    panel: PricePanel,
    specs: tuple[CandidateSpec, ...],
    returns: FloatArray,
    *,
    search_seeds: int = 100,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
]:
    scores = score_market(panel, returns)
    selection_scores = selector_scores(scores, specs)
    mean_fold = selection_scores["mean_fold"]
    candidate_count = len(specs)
    budgets = tuple(b for b in BUDGETS if b <= candidate_count) + (
        candidate_count,
    )

    rows: list[dict[str, float | int | str]] = []
    ensemble_rows: list[dict[str, float | int | str]] = []
    for seed in range(search_seeds):
        permutation = np.random.default_rng(seed).permutation(candidate_count)
        for budget in budgets:
            subset = permutation[:budget]
            subset_test = scores.test_scores[subset]
            rank_corr = rank_correlation(mean_fold[subset], subset_test)
            oracle = float(np.max(subset_test))
            baseline = float(np.mean(subset_test))

            for selector, all_scores in selection_scores.items():
                selected = int(subset[np.argmax(all_scores[subset])])
                selected_quality = float(scores.test_scores[selected])
                regret, efficiency = _selected_metrics(
                    selected_quality, subset_test
                )
                rows.append(
                    {
                        "search_seed": seed,
                        "budget": budget,
                        "selector": selector,
                        "selected_test_sharpe": selected_quality,
                        "oracle_test_sharpe": oracle,
                        "mean_candidate_test_sharpe": baseline,
                        "selection_regret": regret,
                        "frontier_efficiency": efficiency,
                        "selected_positive": int(selected_quality > 0),
                        "validation_test_rank_corr": rank_corr,
                        "selected_family": specs[selected].family,
                        "selected_complexity": specs[selected].complexity,
                    }
                )

            members = _greedy_diverse_ensemble(
                subset,
                mean_fold,
                scores.development_returns,
            )
            ensemble_test = scores.test_returns[:, members].mean(axis=1)
            ensemble_quality = float(
                annualized_sharpe(ensemble_test[:, None])[0]
            )
            ensemble_rows.append(
                {
                    "search_seed": seed,
                    "budget": budget,
                    "members": len(members),
                    "ensemble_test_sharpe": ensemble_quality,
                    "oracle_candidate_test_sharpe": oracle,
                    "ensemble_minus_mean_candidate": (
                        ensemble_quality - baseline
                    ),
                    "ensemble_positive": int(ensemble_quality > 0),
                }
            )
    return rows, ensemble_rows


def summarize(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    grouped: dict[
        tuple[int, str], list[dict[str, float | int | str]]
    ] = {}
    for row in rows:
        grouped.setdefault(
            (int(row["budget"]), str(row["selector"])), []
        ).append(row)

    output = []
    for (budget, selector), group in sorted(grouped.items()):
        output.append(
            {
                "budget": budget,
                "selector": selector,
                "search_seeds": len(group),
                "mean_selected_test_sharpe": float(
                    np.mean([float(row["selected_test_sharpe"]) for row in group])
                ),
                "mean_oracle_test_sharpe": float(
                    np.mean([float(row["oracle_test_sharpe"]) for row in group])
                ),
                "mean_selection_regret": float(
                    np.mean([float(row["selection_regret"]) for row in group])
                ),
                "mean_frontier_efficiency": float(
                    np.mean([float(row["frontier_efficiency"]) for row in group])
                ),
                "selected_positive_rate": float(
                    np.mean([float(row["selected_positive"]) for row in group])
                ),
                "mean_validation_test_rank_corr": float(
                    np.mean(
                        [float(row["validation_test_rank_corr"]) for row in group]
                    )
                ),
            }
        )
    return output


def _write_csv(
    rows: list[dict[str, float | int | str]], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _write_candidate_manifest(
    specs: tuple[CandidateSpec, ...], path: Path
) -> None:
    rows = [
        {
            "candidate_id": spec.candidate_id,
            "family": spec.family,
            "lookback": spec.lookback,
            "secondary": spec.secondary,
            "state": spec.state,
            "rebalance_days": spec.rebalance_days,
            "transform": spec.transform,
            "complexity": spec.complexity,
        }
        for spec in specs
    ]
    _write_csv(rows, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the stale-market selector replication"
    )
    parser.add_argument(
        "--cache", type=Path, default=Path(".cache/prices.csv")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/market_sweep.csv")
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("results/market_summary.csv"),
    )
    parser.add_argument(
        "--ensemble-out",
        type=Path,
        default=Path("results/market_ensemble.csv"),
    )
    parser.add_argument(
        "--zoo-out",
        type=Path,
        default=Path("results/candidate_zoo.csv"),
    )
    parser.add_argument("--search-seeds", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = load_snapshot(args.cache, end_date="2019-12-31")
    specs = build_candidate_zoo()
    returns = candidate_returns(panel, specs, cost_bps=5.0)
    rows, ensemble_rows = run_market_sweep(
        panel,
        specs,
        returns,
        search_seeds=args.search_seeds,
    )
    summary = summarize(rows)
    _write_csv(rows, args.out)
    _write_csv(summary, args.summary_out)
    _write_csv(ensemble_rows, args.ensemble_out)
    _write_candidate_manifest(specs, args.zoo_out)

    payload = {
        "data_start": str(panel.dates[0]),
        "data_end": str(panel.dates[-1]),
        "candidates": len(specs),
        "development_folds": DEV_FOLDS,
        "test_period": TEST_PERIOD,
        "search_seeds": args.search_seeds,
        "max_budget_summary": [
            row for row in summary if row["budget"] == len(specs)
        ],
    }
    print(
        "MARKET_SUMMARY_JSON="
        + json.dumps(payload, separators=(",", ":"))
    )


if __name__ == "__main__":
    main()
