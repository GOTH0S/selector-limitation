from __future__ import annotations

from dataclasses import dataclass

from .population import CandidatePopulation


@dataclass(frozen=True)
class ValidationWinner:
    """Choose the candidate with the highest mean validation score."""

    def select(self, population: CandidatePopulation) -> int:
        return int(population.validation_mean.argmax())
