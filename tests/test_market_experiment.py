import numpy as np

from selector_limitation.candidate_zoo import CandidateSpec
from selector_limitation.market_experiment import (
    _nested_complexity_penalty,
    annualized_sharpe,
    rank_correlation,
)


def test_annualized_sharpe_orders_candidates() -> None:
    returns = np.column_stack(
        [
            np.tile([0.001, 0.0012], 100),
            np.tile([0.0, 0.002], 100),
        ]
    )
    scores = annualized_sharpe(returns)
    assert scores[0] > scores[1]


def test_rank_correlation_is_one_for_same_order() -> None:
    values = np.array([3.0, 1.0, 2.0])
    assert rank_correlation(values, values) == 1.0


def test_nested_penalty_returns_grid_value() -> None:
    folds = np.array(
        [
            [2.0, 2.0, -2.0, -2.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    complexity = np.array([10.0, 1.0])
    penalty = _nested_complexity_penalty(folds, complexity)
    assert penalty in (0.0, 0.025, 0.05, 0.10, 0.20)


def test_candidate_spec_id_is_stable() -> None:
    spec = CandidateSpec(
        "ts_momentum", 20, 0, "all", 5, "sign", 2
    )
    assert spec.candidate_id == "ts_momentum:l20:s0:all:r5:sign"
