import numpy as np

from selector_limitation.search import FamilyWorldSpec, run_family_search


def test_diversity_search_stays_balanced() -> None:
    spec = FamilyWorldSpec(
        n_families=8,
        initial_per_family=1,
        diversity_probability=1.0,
    )
    trace = run_family_search(
        spec,
        budget=80,
        mode="diversity_preserving",
        seed=12,
    )
    assert int(trace.family_counts.max() - trace.family_counts.min()) <= 1


def test_diversity_search_is_deterministic() -> None:
    spec = FamilyWorldSpec()
    left = run_family_search(
        spec,
        budget=100,
        mode="diversity_preserving",
        seed=41,
    )
    right = run_family_search(
        spec,
        budget=100,
        mode="diversity_preserving",
        seed=41,
    )
    np.testing.assert_array_equal(left.family_ids, right.family_ids)
