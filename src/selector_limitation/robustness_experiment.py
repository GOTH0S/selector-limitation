from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Literal

import numpy as np

from .metrics import evaluate_selection
from .population import CandidatePopulation
from .selectors import ValidationWinner

World = Literal["gaussian", "heavy_tail", "family_bias", "regime_shift"]

COUNTS = (10, 25, 50, 100, 250, 500, 1000)
NOISE = (0.25, 0.5, 1.0, 2.0)
WORLDS: tuple[World, ...] = (
    "gaussian",
    "heavy_tail",
    "family_bias",
    "regime_shift",
)


def generate_world(
    *,
    world: World,
    n_candidates: int,
    validation_noise: float,
    seed: int,
) -> CandidatePopulation:
    rng = np.random.default_rng(seed)
    true_quality = rng.normal(size=n_candidates)

    if world == "gaussian":
        observed = true_quality + rng.normal(
            scale=validation_noise, size=n_candidates
        )
    elif world == "heavy_tail":
        noise = rng.standard_t(df=3, size=n_candidates) / np.sqrt(3.0)
        observed = true_quality + validation_noise * noise
    elif world == "family_bias":
        family_ids = np.arange(n_candidates) % 8
        rng.shuffle(family_ids)
        family_bias = rng.normal(size=8)
        idiosyncratic = rng.normal(size=n_candidates)
        shock = (
            0.75 * family_bias[family_ids]
            + np.sqrt(1.0 - 0.75**2) * idiosyncratic
        )
        observed = true_quality + validation_noise * shock
    elif world == "regime_shift":
        innovation = rng.normal(size=n_candidates)
        development_quality = (
            0.5 * true_quality + np.sqrt(0.75) * innovation
        )
        observed = development_quality + rng.normal(
            scale=validation_noise, size=n_candidates
        )
    else:
        raise ValueError(f"unknown world: {world}")

    return CandidatePopulation(
        true_quality=true_quality.astype(np.float64, copy=False),
        validation_scores=observed[:, None].astype(np.float64, copy=False),
    )


def run_robustness_sweep(*, seeds: int) -> list[dict[str, float | int | str]]:
    if seeds < 1:
        raise ValueError("seeds must be positive")

    selector = ValidationWinner()
    rows: list[dict[str, float | int | str]] = []

    for world in WORLDS:
        for validation_noise in NOISE:
            for candidate_count in COUNTS:
                outcomes = []
                for seed in range(seeds):
                    population = generate_world(
                        world=world,
                        n_candidates=candidate_count,
                        validation_noise=validation_noise,
                        seed=seed,
                    )
                    selected = selector.select(population)
                    outcomes.append(evaluate_selection(population, selected))

                rows.append(
                    {
                        "world": world,
                        "candidate_count": candidate_count,
                        "validation_noise": validation_noise,
                        "seeds": seeds,
                        "mean_oracle_quality": float(
                            np.mean([outcome.oracle_quality for outcome in outcomes])
                        ),
                        "mean_selected_quality": float(
                            np.mean([outcome.selected_quality for outcome in outcomes])
                        ),
                        "mean_regret": float(
                            np.mean([outcome.regret for outcome in outcomes])
                        ),
                        "mean_efficiency": float(
                            np.mean([outcome.efficiency for outcome in outcomes])
                        ),
                        "oracle_hit_rate": float(
                            np.mean(
                                [
                                    outcome.selected_index == outcome.oracle_index
                                    for outcome in outcomes
                                ]
                            )
                        ),
                    }
                )
    return rows


def write_csv(rows: list[dict[str, float | int | str]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic robustness sweeps")
    parser.add_argument("--seeds", type=int, default=2000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/robustness_sweep.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_csv(run_robustness_sweep(seeds=args.seeds), args.out)


if __name__ == "__main__":
    main()
