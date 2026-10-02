# selector-limitation

More search is only useful if selection keeps up.

Suppose a research process generates `N` candidate signals. Candidate `i` has an unknown true quality `q_i`, but the researcher only sees a noisy validation estimate `v_i`. As `N` grows, the best candidate the process *could* have selected can improve while the candidate it *actually* selects becomes increasingly dominated by validation noise.

This repository studies that gap directly.

The first experiment is synthetic so the latent quality of every candidate is known. That lets us separate two quantities that are usually conflated in backtests:

- **reachable frontier** — the true quality of the best candidate generated;
- **selected quality** — the true quality of the candidate chosen using validation data only.

Their difference is **selection regret**.

The initial sweep varies candidate count and validation noise under a fixed data-generating process. Later experiments will add correlated candidate families, adaptive search, alternative selectors, and a stale public-market replication.

## First result

With validation noise fixed at `2.0`, a 2,000-seed sweep gives:

| candidates | oracle quality | selected quality | regret | oracle hit rate |
|---:|---:|---:|---:|---:|
| 10 | 1.529 | 0.720 | 0.809 | 27.5% |
| 100 | 2.501 | 1.133 | 1.368 | 9.5% |
| 1,000 | 3.239 | 1.445 | 1.794 | 4.0% |

The selector still benefits from more search in absolute terms, but it captures the expanding frontier much more slowly. The gap is the object of study here; later experiments will test when it becomes large enough that additional search actually degrades realised selection quality and which selectors recover the lost frontier.

## Run

```bash
python -m pip install -e ".[dev]"
pytest -q
python -m selector_limitation.experiment --out results/synthetic_sweep.csv
```

The experiment writes one row per `(noise, candidate_count)` cell with Monte Carlo means and quantiles. No test-set or latent-quality information is available to the selector.
