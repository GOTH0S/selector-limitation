from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .population import PopulationSpec
from .simulation import run_trial, summarize_trials

DEFAULT_COUNTS = (10, 25, 50, 100, 250, 500, 1000)
DEFAULT_NOISE = (0.25, 0.5, 1.0, 2.0)


def run_sweep(*, seeds: int, validation_folds: int) -> list[dict[str, float]]:
    if seeds < 1:
        raise ValueError("seeds must be positive")

    rows: list[dict[str, float]] = []
    for validation_noise in DEFAULT_NOISE:
        for candidate_count in DEFAULT_COUNTS:
            spec = PopulationSpec(
                n_candidates=candidate_count,
                validation_noise=validation_noise,
                validation_folds=validation_folds,
            )
            trials = [run_trial(spec, seed=seed) for seed in range(seeds)]
            summary = summarize_trials(trials)
            rows.append(
                {
                    "candidate_count": candidate_count,
                    "validation_noise": validation_noise,
                    "validation_folds": validation_folds,
                    "seeds": seeds,
                    **summary,
                }
            )
    return rows


def write_csv(rows: list[dict[str, float]], out: Path) -> None:
    if not rows:
        raise ValueError("rows cannot be empty")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic selection-regret sweep")
    parser.add_argument("--seeds", type=int, default=2000)
    parser.add_argument("--validation-folds", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results/synthetic_sweep.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_sweep(seeds=args.seeds, validation_folds=args.validation_folds)
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()
