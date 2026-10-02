from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .population import CandidatePopulation

IntArray = NDArray[np.int64]


class Selector(Protocol):
    name: str

    def select(
        self,
        population: CandidatePopulation,
        *,
        family_ids: IntArray | None = None,
    ) -> int: ...


@dataclass(frozen=True)
class ValidationWinner:
    """Choose the candidate with the highest mean validation score."""

    name: str = "mean_fold_winner"

    def select(
        self,
        population: CandidatePopulation,
        *,
        family_ids: IntArray | None = None,
    ) -> int:
        del family_ids
        return int(population.validation_mean.argmax())


@dataclass(frozen=True)
class SingleFoldWinner:
    fold: int = 0
    name: str = "single_fold_winner"

    def select(
        self,
        population: CandidatePopulation,
        *,
        family_ids: IntArray | None = None,
    ) -> int:
        del family_ids
        if self.fold < 0 or self.fold >= population.validation_scores.shape[1]:
            raise ValueError("fold outside validation score matrix")
        return int(population.validation_scores[:, self.fold].argmax())


@dataclass(frozen=True)
class MeanRankSelector:
    """Prefer candidates that rank well repeatedly rather than winning on magnitude."""

    name: str = "mean_rank"

    def select(
        self,
        population: CandidatePopulation,
        *,
        family_ids: IntArray | None = None,
    ) -> int:
        del family_ids
        order = np.argsort(-population.validation_scores, axis=0)
        ranks = np.empty_like(order)
        columns = np.arange(order.shape[1])
        ranks[order, columns] = np.arange(order.shape[0])[:, None]
        return int(ranks.mean(axis=1).argmin())


@dataclass(frozen=True)
class StabilityPenaltySelector:
    penalty: float = 1.0
    name: str = "stability_penalty"

    def __post_init__(self) -> None:
        if self.penalty < 0:
            raise ValueError("penalty must be non-negative")

    def select(
        self,
        population: CandidatePopulation,
        *,
        family_ids: IntArray | None = None,
    ) -> int:
        del family_ids
        scores = population.validation_scores
        if scores.shape[1] == 1:
            return int(scores[:, 0].argmax())
        standard_error = scores.std(axis=1, ddof=1) / np.sqrt(scores.shape[1])
        robust_score = scores.mean(axis=1) - self.penalty * standard_error
        return int(robust_score.argmax())


@dataclass(frozen=True)
class HierarchicalShrinkageSelector:
    strength: float = 4.0
    name: str = "hierarchical_shrinkage"

    def __post_init__(self) -> None:
        if self.strength < 0:
            raise ValueError("strength must be non-negative")

    def select(
        self,
        population: CandidatePopulation,
        *,
        family_ids: IntArray | None = None,
    ) -> int:
        if family_ids is None:
            raise ValueError("family_ids are required for hierarchical shrinkage")
        if family_ids.shape != (population.n_candidates,):
            raise ValueError("family_ids do not match population")

        candidate_mean = population.validation_mean
        family_center = np.empty(population.n_candidates, dtype=np.float64)
        for family in np.unique(family_ids):
            mask = family_ids == family
            family_center[mask] = candidate_mean[mask].mean()

        folds = population.validation_scores.shape[1]
        weight = folds / (folds + self.strength)
        shrunk = family_center + weight * (candidate_mean - family_center)
        return int(shrunk.argmax())


@dataclass(frozen=True)
class FamilyBalancedSelector:
    """Select a family first so a large family gets no extra shots at the maximum."""

    name: str = "family_balanced"

    def select(
        self,
        population: CandidatePopulation,
        *,
        family_ids: IntArray | None = None,
    ) -> int:
        if family_ids is None:
            raise ValueError("family_ids are required for family-balanced selection")
        if family_ids.shape != (population.n_candidates,):
            raise ValueError("family_ids do not match population")

        candidate_mean = population.validation_mean
        families = np.unique(family_ids)
        family_scores = np.array(
            [np.median(candidate_mean[family_ids == family]) for family in families]
        )
        selected_family = families[int(family_scores.argmax())]
        indices = np.flatnonzero(family_ids == selected_family)
        return int(indices[candidate_mean[indices].argmax()])
