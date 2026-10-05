# selector-limitation

If you test more ideas, you should have a better chance of finding a good one. But if your evaluation is noisy, a larger search also gives you more chances to pick a lucky-looking bad idea.

This repo measures that gap.

## 30-second version

The experiment is:

```text
generate candidates -> score them on development data -> pick one -> reveal future/true quality
```

There are two versions of the same setup.

- **Synthetic:** a candidate is just an idea with a hidden true quality and a noisy development score. This lets us know exactly which idea was really best.
- **Historical:** a candidate is one of 2,670 simple ETF trading rules. The selector sees only pre-2018 performance; 2018–2019 is used to judge what it picked.

As the search budget grows, I compare:

- **best available** — the best candidate the larger search happened to contain;
- **selected** — the candidate a realistic selector chose from development evidence.

Their difference is **selection regret**.

The main result: larger searches reliably contain better candidates, but the selector often fails to keep up. With sufficiently heavy-tailed noise, searching more can actually make the chosen candidate worse.

![Frontier versus selected quality](figures/frontier.svg)

## Synthetic result

With ordinary Gaussian noise, searching more still improves the chosen candidate, just more slowly than it improves the best available candidate.

With heavy-tailed noise, the result can reverse. At noise `2.0`:

| candidates searched | best available | selected |
|---:|---:|---:|
| 10 | 1.53 | 0.74 |
| 100 | 2.50 | 0.60 |
| 1,000 | 3.24 | 0.30 |

The search finds better ideas while the selection rule makes worse choices.

![Heavy-tailed regret surface](figures/regret_phase.svg)

## Search policy

Candidates can also belong to related families. A winner-following search keeps proposing from whichever family currently looks best; a diversity-preserving search keeps exploring the others.

At 1,000 proposals, winner-following collapses to **1.15 effective families** and reaches a best-available quality of **2.22**. Uniform and diversity-preserving search keep roughly eight families alive and reach **2.36**.

![Search policy comparison](figures/search_policy.svg)

## Historical check

The market experiment uses 2,670 generic price-only strategies on eight liquid ETFs.

A search budget of 25 means the selector only gets the first 25 candidates in a seeded search order; a budget of 2,670 means it gets the whole zoo. Candidate construction never uses the test period.

- development evidence: 2010–2017
- test period: 2018–2019
- trading cost: 5 bps per unit of turnover

| candidates searched | best future Sharpe | single-fold choice | four-fold choice | 5-candidate ensemble |
|---:|---:|---:|---:|---:|
| 25 | 1.05 | -0.09 | 0.10 | -0.16 |
| 500 | 1.57 | -0.27 | 0.41 | 0.33 |
| 1,000 | 1.70 | -0.37 | 0.53 | 0.41 |
| 2,670 | 1.86 | -0.38 | 0.56 | **0.73** |

“Best future Sharpe” is hindsight only. It tells us what the search contained, not what could have been known at selection time.

![Historical replication](figures/market_replication.svg)

Across four test periods and 12 fixed universes, the best available candidate improves with larger search in all 48 cases. Single-fold selection gets worse in 27; four-fold selection gets worse in 25. So the effect is real, but not universal.

[RESULTS.md](RESULTS.md) has the remaining setup and tables.

## Related

See: [concept-recomposition](https://github.com/GOTH0S/concept-recomposition), a separate experiment on whether reusable intermediate representations expand what a bounded search can reach.

## Reproduce

```bash
python -m pip install -e ".[dev]"
python -m selector_limitation.reproduce
```

Add `--market` to rerun the historical experiments. The price file is pinned to a public commit and hash-checked before use.

Python 3.11+; NumPy, Matplotlib, pytest and Ruff. Apache-2.0.
