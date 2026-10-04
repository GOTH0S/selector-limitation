# Results

## What is being measured?

Every experiment has the same structure:

1. produce a set of candidate ideas;
2. give each candidate some development evidence;
3. choose a candidate without seeing its future/true outcome;
4. compare that choice with the best candidate that was actually available.

The last comparison is **selection regret**.

In the synthetic experiments, “true quality” is generated directly, so the best candidate is known exactly. In the market experiments, the analogue is future test-period Sharpe.

## Synthetic search

Each candidate gets:

- a hidden true quality;
- a noisy development score.

The selector picks the highest development score.

With Gaussian noise `2.0`, more search still improves the selected candidate, but the best available candidate improves faster:

| candidates | best available | selected | regret |
|---:|---:|---:|---:|
| 10 | 1.529 | 0.720 | 0.809 |
| 100 | 2.501 | 1.133 | 1.368 |
| 1,000 | 3.239 | 1.445 | 1.794 |

The robustness sweep changes how development scores can be wrong: Gaussian noise, Student-t heavy tails, correlated family bias, and a development/future regime shift.

Heavy-tailed errors give the clearest failure case:

| candidates | best available | selected | regret |
|---:|---:|---:|---:|
| 10 | 1.529 | 0.742 | 0.787 |
| 100 | 2.501 | 0.600 | 1.901 |
| 1,000 | 3.239 | 0.302 | 2.937 |

Here, searching more genuinely makes the final choice worse.

## Correlated search

Candidates are split into eight families. Members of a family share a true family mean and a persistent validation bias.

Three proposal rules are compared:

- **uniform:** choose families at random;
- **winner following:** spend 95% of new proposals in the family containing the current validation winner;
- **diversity preserving:** mostly propose from the least-sampled families.

At 1,000 proposals:

| search policy | selected | best available | effective families |
|---|---:|---:|---:|
| uniform | 0.935 | 2.358 | 7.94 |
| winner following | 1.013 | 2.222 | 1.15 |
| diversity preserving | 0.973 | 2.360 | 8.00 |

Winner following slightly improves the candidate picked by its noisy score, but reduces what the search discovers by collapsing onto one family.

The selector comparison then holds the search trace fixed and changes only the choice rule: one fold, repeated-fold mean, mean rank, stability penalty, hierarchical shrinkage and family balancing.

## Historical data

The candidate set is 2,670 deterministic price-only trading rules built from simple momentum/reversal, moving-average, breakout, volatility-adjusted and trend-acceleration variants.

Assets:

`SPY QQQ IWM EFA EEM TLT GLD DBC`

A signal using date-`t` data earns only date-`t+1` return. Trading costs are 5 bps per unit of turnover.

The selector sees four development folds covering 2010–2017. The 2018–2019 period is held back for evaluation.

At the full 2,670-candidate search:

- best candidate in hindsight: **1.859 Sharpe**;
- candidate chosen from one development fold: **-0.379**;
- candidate chosen from four-fold mean performance: **0.562**;
- five-candidate diversity ensemble: **0.725**.

The hindsight best is not a tradable result; it measures the quality of the opportunity set the search produced.

The adjusted-price source is pinned to `marcoreyess22/jump-risk-engine@3187e756957af549e54e470e8064dbe782d319b8` and hash-checked.

## Period and universe stress

The same candidate language is rerun across 2012–13, 2014–15, 2016–17 and 2018–19, crossed with 12 fixed universes.

From a search budget of 25 to 2,670, the best available candidate improves in all 48 period/universe cells. The selected candidate gets worse in:

| selector | cells |
|---|---:|
| single fold | 27 / 48 |
| four-fold mean | 25 / 48 |
| mean rank | 20 / 48 |
| stability LCB | 20 / 48 |
| family balanced | 17 / 48 |

2016–2017 is the hardest period: four-fold selection deteriorates in all 12 universes. In 2018–2019 it improves in 10 of 12.

The point is not that more search is always harmful. It is that search quality and selection quality are separate bottlenecks: a research process can get better at finding promising possibilities faster than it gets better at recognizing which one will hold up.
