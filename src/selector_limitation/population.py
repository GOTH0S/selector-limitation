from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PopulationSpec:
    n_candidates: int
    validation_noise: float
    validation_folds: int = 1
    quality_mean: float = 0.0
    quality_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.n_candidates < 2:
            raise ValueError("n_candidates must be at least 2")
        if self.validation_noise < 0:
            raise ValueError("validation_noise must be non-negative")
        if self.validation_folds < 1:
            raise ValueError("validation_folds must be positive")
        if self.quality_scale <= 0:
            raise ValueError("quality_scale must be positive")


@dataclass(frozen=True)
class CandidatePopulation:
    true_quality: FloatArray
    validation_scores: FloatArray

    def __post_init__(self) -> None:
        if self.true_quality.ndim != 1:
            raise ValueError("true_quality must be one-dimensional")
        if self.validation_scores.ndim != 2:
            raise ValueError("validation_scores must be two-dimensional")
        if self.validation_scores.shape[0] != self.true_quality.shape[0]:
            raise ValueError("candidate axis does not match")

    @property
    def n_candidates(self) -> int:
        return int(self.true_quality.shape[0])

    @property
    def validation_mean(self) -> FloatArray:
        return self.validation_scores.mean(axis=1)


def generate_population(spec: PopulationSpec, *, seed: int) -> CandidatePopulation:
    rng = np.random.default_rng(seed)
    true_quality = rng.normal(
        loc=spec.quality_mean,
        scale=spec.quality_scale,
        size=spec.n_candidates,
    )
    noise = rng.normal(
        loc=0.0,
        scale=spec.validation_noise,
        size=(spec.n_candidates, spec.validation_folds),
    )
    validation_scores = true_quality[:, None] + noise
    return CandidatePopulation(
        true_quality=true_quality.astype(np.float64, copy=False),
        validation_scores=validation_scores.astype(np.float64, copy=False),
    )
