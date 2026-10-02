"""Selection-regret experiments with known latent candidate quality."""

from .metrics import SelectionOutcome, evaluate_selection
from .population import CandidatePopulation, PopulationSpec, generate_population
from .selectors import ValidationWinner

__all__ = [
    "CandidatePopulation",
    "PopulationSpec",
    "SelectionOutcome",
    "ValidationWinner",
    "evaluate_selection",
    "generate_population",
]
