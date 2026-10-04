# Results

## Synthetic search

The baseline draws latent candidate quality from a standard normal and observes it through noisy validation estimates. With Gaussian noise `2.0`, search still helps in absolute terms:

| candidates | oracle | selected | regret |
|---:|---:|---:|---:|
| 10 | 1.529 | 0.720 | 0.809 |
| 100 | 2.501 | 1.133 | 1.368 |
| 1,000 | 3.239 | 1.445 | 1.794 |

The robustness sweep keeps the same candidate counts and noise levels but changes the error process: Gaussian, Student-t with three degrees of freedom, family-level correlated bias, and a development/future regime shift.

Heavy-tailed errors are the clear failure case. At validation noise `2.0`:

| candidates | oracle | selected | regret |
|---:|---:|---:|---:|
| 10 | 1.529 | 0.742 | 0.787 |
| 100 | 2.501 | 0.600 | 1.901 |
| 1,000 | 3.239 | 0.302 | 2.937 |

More search finds better candidates and produces worse choices.

## Correlated search

Eight candidate families have different true means and persistent validation biases. The search policies share the same starting round.

| policy, 1,000 proposals | selected | oracle | effective families |
|---|---:|---:|---:|
| uniform | 0.935 | 2.358 | 7.94 |
| winner following | 1.013 | 2.222 | 1.15 |
| diversity preserving | 0.973 | 2.360 | 8.00 |

Winner-following slightly improves the candidate selected by the noisy validator, but it suppresses the frontier by concentrating the search. Diversity preservation keeps the frontier close to uniform search.

The selector experiment then freezes search traces and compares one-fold selection, repeated-fold averaging, mean rank, stability penalties, hierarchical shrinkage and family balancing. Repeated validation recovers part of the gap; no selector closes it.

## Historical data

The market experiment uses adjusted daily prices for `SPY QQQ IWM EFA EEM TLT GLD DBC`, hard-stopped at 2019-12-31. The source file is pinned to `marcoreyess22/jump-risk-engine@3187e756957af549e54e470e8064dbe782d319b8` and accepted only if its SHA-256 is:

```text
0f1c7534aed1afc5d431be99f5c0357243e7357905e60b74b40a8711689f71bd
```

The candidate zoo contains 2,670 deterministic price-only hypotheses built from simple momentum/reversal, moving-average, breakout, volatility-adjusted and trend-acceleration variants. A signal using date-`t` data earns only date-`t+1` return. Costs are 5 bps per unit of turnover.

Four development folds cover 2010–2017. The 2018–2019 test period is not used in candidate construction or selector calibration.

At full search the ex-post frontier is **1.859 Sharpe**. Single-fold selection chooses a candidate with **-0.379** test Sharpe; the four-fold mean chooses **0.562**; the five-candidate diversity ensemble reaches **0.725**.

## Period and universe stress

The same candidate language is tested across 2012–13, 2014–15, 2016–17 and 2018–19, crossed with 12 fixed universes.

The mean ex-post frontier rises from budget 25 to 2,670 in all 48 cells. Search expansion nevertheless lowers selected future Sharpe in:

| selector | cells |
|---|---:|
| single fold | 27 / 48 |
| four-fold mean | 25 / 48 |
| mean rank | 20 / 48 |
| stability LCB | 20 / 48 |
| family balanced | 17 / 48 |

2016–2017 is the hardest period: four-fold mean selection deteriorates in all 12 universes. In 2018–2019 it improves in 10 of 12.

The useful conclusion is narrower than a universal multiple-testing rule: a larger hypothesis space can reliably improve the opportunity set while making the selection problem harder, and the severity of that bottleneck depends on both the error structure and the regime.
