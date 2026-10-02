"""Selection-regret experiments with known latent candidate quality."""

from .metrics import SelectionOutcome, evaluate_selection
from .population import CandidatePopulation, PopulationSpec, generate_population
from .search import FamilyWorldSpec, SearchTrace, run_family_search
from .selectors import ValidationWinner

__all__ = [
    "CandidatePopulation",
    "FamilyWorldSpec",
    "PopulationSpec",
    "SearchTrace",
    "SelectionOutcome",
    "ValidationWinner",
    "evaluate_selection",
    "generate_population",
    "run_family_search",
]
