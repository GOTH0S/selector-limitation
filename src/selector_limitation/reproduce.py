from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .experiment import run_sweep
from .experiment import write_csv as write_synthetic
from .family_experiment import run_family_sweep
from .family_experiment import write_csv as write_family
from .figures import render_all
from .robustness_experiment import run_robustness_sweep
from .robustness_experiment import write_csv as write_robustness
from .selector_experiment import run_selector_sweep
from .selector_experiment import write_csv as write_selectors


def run_module(module: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, "-m", module, *args],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the main results")
    parser.add_argument(
        "--market",
        action="store_true",
        help="also rerun the historical market and stress experiments",
    )
    args = parser.parse_args()

    write_synthetic(
        run_sweep(seeds=2000, validation_folds=1),
        Path("results/synthetic_sweep.csv"),
    )
    write_robustness(
        run_robustness_sweep(seeds=2000),
        Path("results/robustness_sweep.csv"),
    )
    write_family(
        run_family_sweep(seeds=500),
        Path("results/family_search_sweep.csv"),
    )
    write_selectors(
        run_selector_sweep(seeds=1000),
        Path("results/selector_sweep.csv"),
    )

    if args.market:
        run_module(
            "selector_limitation.market_experiment",
            "--search-seeds",
            "100",
        )
        run_module(
            "selector_limitation.stress_experiment",
            "--search-seeds",
            "100",
        )

    render_all()


if __name__ == "__main__":
    main()
