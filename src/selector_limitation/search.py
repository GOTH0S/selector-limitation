from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .population import CandidatePopulation

SearchMode = Literal["uniform", "winner_following", "diversity_preserving"]
IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FamilyWorldSpec:
    n_families: int = 8
    initial_per_family: int = 2
    validation_folds: int = 1
    family_quality_scale: float = 1.0
    within_family_quality_scale: float = 0.35
    validation_noise: float = 1.0
    family_validation_bias_scale: float = 2.0
    exploit_probability: float = 0.95
    diversity_probability: float = 0.95

    def __post_init__(self) -> None:
        if self.n_families < 2:
            raise ValueError("n_families must be at least 2")
        if self.initial_per_family < 1:
            raise ValueError("initial_per_family must be positive")
        if self.validation_folds < 1:
            raise ValueError("validation_folds must be positive")
        if self.family_quality_scale <= 0:
            raise ValueError("family_quality_scale must be positive")
        if self.within_family_quality_scale < 0:
            raise ValueError("within_family_quality_scale must be non-negative")
        if self.validation_noise < 0:
            raise ValueError("validation_noise must be non-negative")
        if self.family_validation_bias_scale < 0:
            raise ValueError("family_validation_bias_scale must be non-negative")
        if not 0.0 <= self.exploit_probability <= 1.0:
            raise ValueError("exploit_probability must lie in [0, 1]")
        if not 0.0 <= self.diversity_probability <= 1.0:
            raise ValueError("diversity_probability must lie in [0, 1]")

    @property
    def initial_budget(self) -> int:
        return self.n_families * self.initial_per_family


@dataclass(frozen=True)
class SearchTrace:
    population: CandidatePopulation
    family_ids: IntArray
    family_true_quality: FloatArray
    family_validation_bias: FloatArray
    family_counts: IntArray
    mode: SearchMode

    @property
    def allocation_hhi(self) -> float:
        weights = self.family_counts / self.family_counts.sum()
        return float(np.square(weights).sum())

    @property
    def effective_family_count(self) -> float:
        return 1.0 / self.allocation_hhi


def _draw_candidate(
    rng: np.random.Generator,
    family: int,
    family_true_quality: FloatArray,
    family_validation_bias: FloatArray,
    spec: FamilyWorldSpec,
) -> tuple[float, FloatArray]:
    quality = float(
        family_true_quality[family]
        + rng.normal(0.0, spec.within_family_quality_scale)
    )
    scores = (
        quality
        + family_validation_bias[family]
        + rng.normal(0.0, spec.validation_noise, size=spec.validation_folds)
    )
    return quality, scores.astype(np.float64, copy=False)


def run_family_search(
    spec: FamilyWorldSpec,
    *,
    budget: int,
    mode: SearchMode,
    seed: int,
) -> SearchTrace:
    if budget < spec.initial_budget:
        raise ValueError(f"budget must be at least {spec.initial_budget}")
    if mode not in ("uniform", "winner_following", "diversity_preserving"):
        raise ValueError(f"unknown search mode: {mode}")

    rng = np.random.default_rng(seed)
    family_true_quality = rng.normal(
        0.0, spec.family_quality_scale, size=spec.n_families
    )
    family_validation_bias = rng.normal(
        0.0, spec.family_validation_bias_scale, size=spec.n_families
    )

    true_quality = np.empty(budget, dtype=np.float64)
    validation_scores = np.empty(
        (budget, spec.validation_folds), dtype=np.float64
    )
    family_ids = np.empty(budget, dtype=np.int64)
    family_counts = np.zeros(spec.n_families, dtype=np.int64)

    cursor = 0
    best_validation = -np.inf
    best_family = 0

    def append_candidate(family: int) -> None:
        nonlocal cursor, best_validation, best_family
        quality, scores = _draw_candidate(
            rng,
            family,
            family_true_quality,
            family_validation_bias,
            spec,
        )
        true_quality[cursor] = quality
        validation_scores[cursor] = scores
        family_ids[cursor] = family
        family_counts[family] += 1
        mean_score = float(scores.mean())
        if mean_score > best_validation:
            best_validation = mean_score
            best_family = family
        cursor += 1

    for family in range(spec.n_families):
        for _ in range(spec.initial_per_family):
            append_candidate(family)

    while cursor < budget:
        if mode == "uniform":
            family = int(rng.integers(spec.n_families))
        elif mode == "winner_following":
            family = (
                best_family
                if rng.random() < spec.exploit_probability
                else int(rng.integers(spec.n_families))
            )
        elif rng.random() < spec.diversity_probability:
            least_used = np.flatnonzero(family_counts == family_counts.min())
            family = int(rng.choice(least_used))
        else:
            family = int(rng.integers(spec.n_families))
        append_candidate(family)

    return SearchTrace(
        population=CandidatePopulation(
            true_quality=true_quality,
            validation_scores=validation_scores,
        ),
        family_ids=family_ids,
        family_true_quality=family_true_quality.astype(
            np.float64, copy=False
        ),
        family_validation_bias=family_validation_bias.astype(
            np.float64, copy=False
        ),
        family_counts=family_counts,
        mode=mode,
    )
