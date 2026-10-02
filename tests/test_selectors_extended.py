import numpy as np
import pytest

from selector_limitation.population import CandidatePopulation
from selector_limitation.selectors import (
    FamilyBalancedSelector,
    HierarchicalShrinkageSelector,
    MeanRankSelector,
    SingleFoldWinner,
    StabilityPenaltySelector,
    ValidationWinner,
)


def population(
    scores: list[list[float]], quality: list[float] | None = None
) -> CandidatePopulation:
    matrix = np.array(scores, dtype=float)
    if quality is None:
        quality = [0.0] * matrix.shape[0]
    return CandidatePopulation(np.array(quality, dtype=float), matrix)


def test_single_fold_and_mean_fold_can_disagree() -> None:
    p = population([[10.0, 0.0, 0.0], [4.0, 4.0, 4.0]])
    assert SingleFoldWinner().select(p) == 0
    assert ValidationWinner().select(p) == 1


def test_mean_rank_rewards_repeated_high_rank() -> None:
    p = population([[10.0, 0.0, 0.0], [4.0, 4.0, 4.0], [3.0, 3.0, 3.0]])
    assert MeanRankSelector().select(p) == 1


def test_stability_penalty_can_reject_volatile_winner() -> None:
    p = population([[10.0, 0.0, 0.0], [3.5, 3.5, 3.5]])
    assert StabilityPenaltySelector(penalty=1.0).select(p) == 1


def test_family_balancing_removes_maximum_by_family_size() -> None:
    p = population([[8.0], [0.0], [0.0], [0.0], [5.0], [5.0]])
    families = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
    assert FamilyBalancedSelector().select(p, family_ids=families) == 4


def test_hierarchical_shrinkage_requires_family_ids() -> None:
    p = population([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError):
        HierarchicalShrinkageSelector().select(p)
