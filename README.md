# selector-limitation

More search is only useful if selection keeps up.

Suppose a research process generates `N` candidate signals. Candidate `i` has an unknown true quality `q_i`, while the researcher sees only a noisy validation estimate. As the search expands, the best candidate *discovered* can improve much faster than the candidate the research process can reliably *identify*.

This repository separates those two problems:

- **reachable frontier** — the true quality of the best candidate generated;
- **selected quality** — the true quality of the candidate chosen from development evidence;
- **selection regret** — the gap between them.

The first three experiments use synthetic worlds where latent quality is known. The fourth moves the same question to a frozen historical ETF panel.

## Experiment 1 — independent candidates

With independent candidate quality and noisy validation, increasing search creates a steadily better oracle frontier while a naive selector captures a shrinking share of it.

At validation noise `2.0`, across 2,000 seeds:

| candidates | oracle quality | selected quality | regret | oracle hit rate |
|---:|---:|---:|---:|---:|
| 10 | 1.529 | 0.720 | 0.809 | 27.5% |
| 100 | 2.501 | 1.133 | 1.368 | 9.5% |
| 1,000 | 3.239 | 1.445 | 1.794 | 4.0% |

More search still helps the selected candidate in absolute terms, but the selector falls further behind the frontier.

## Experiment 2 — correlated families and adaptive search

Real research produces related variants rather than independent draws. Each synthetic family therefore has a latent true mean and a persistent validation bias:

```text
candidate quality = family quality + within-family variation
validation score  = candidate quality + family bias + observation noise
```

Two budget-matched search policies are compared: uniform allocation and 95% winner-following exploitation.

At 1,000 candidates across 2,000 seeds:

| search | selected quality | oracle quality | regret | true-best-family hit | effective families |
|---|---:|---:|---:|---:|---:|
| uniform | 0.913 | 2.340 | 1.428 | 28.5% | 7.94 |
| winner-following | 0.982 | 2.190 | 1.208 | 29.6% | 1.16 |

Winner-following does **not** make selected quality worse in this model. It does something different: it collapses search breadth and lowers the frontier that can be reached. The negative result is retained rather than tuning the simulation until search degradation appears.

## Experiment 3 — selector recovery

The search trace is then frozen and competing selectors receive the same evidence. The comparison includes one-fold selection, repeated-fold averaging, mean rank, a stability penalty, hierarchical shrinkage and family-balanced selection.

At a 1,000-candidate budget:

| search | selector | selected quality | regret | frontier efficiency |
|---|---|---:|---:|---:|
| uniform | single fold | 0.902 | 1.420 | 0.386 |
| uniform | five-fold mean | 1.160 | 1.162 | 0.500 |
| uniform | family-balanced | **1.177** | **1.145** | **0.507** |
| winner-following | single fold | 0.953 | 1.224 | 0.282 |
| winner-following | five-fold mean | **1.268** | **0.909** | **0.500** |
| winner-following | family-balanced | 1.226 | 0.951 | 0.451 |

Repeated validation repairs a meaningful amount of noisy selection. It cannot recreate candidates suppressed upstream by concentrated search.

## Experiment 4 — stale-market replication

The synthetic result is then tested on real historical prices without introducing production signals.

**Data.** Eight liquid ETFs — `SPY QQQ IWM EFA EEM TLT GLD DBC` — from a fixed public adjusted-close snapshot. The source file is downloaded from `marcoreyess22/jump-risk-engine` at commit `3187e75` and accepted only if its SHA-256 is

```text
0f1c7534aed1afc5d431be99f5c0357243e7357905e60b74b40a8711689f71bd
```

The source history starts in 2007; this experiment hard-stops it on **2019-12-31**. The dataset is not vendored here.

**Candidate zoo.** 2,670 deterministic, generic price-only hypotheses assembled from textbook primitives: time-series and cross-sectional momentum/reversal, moving-average gaps, breakouts, volatility-adjusted momentum and trend acceleration; multiple lookbacks, simple market-state gates, 1/5/20-day rebalance intervals and sign/linear transforms. Each candidate carries family and complexity metadata. Trading costs are 5 bps per unit of turnover.

A signal computed with date-`t` information earns only date-`t+1` return, and P&L is labelled on the realization date. The boundary convention is regression-tested.

**Protocol.**

- development: four chronological two-year folds covering 2010–2017;
- untouched future evaluation: **2018-01-02 to 2019-12-31**;
- search budgets: 25, 50, 100, 250, 500, 1,000, 2,000 and all 2,670 candidates;
- 100 seeded random search orders for partial-budget comparisons;
- the test period is never available to candidate construction or selector calibration.

The future-period oracle below is deliberately **ex post**. It is a diagnostic of what the search happened to contain, not a selectable strategy.

| budget | future oracle | single-fold selector | four-fold mean | 5-candidate ensemble | dev→future rank corr. |
|---:|---:|---:|---:|---:|---:|
| 25 | 1.052 | -0.087 | 0.104 | -0.161 | 0.040 |
| 100 | 1.331 | -0.135 | 0.206 | 0.034 | 0.052 |
| 500 | 1.566 | -0.267 | 0.413 | 0.332 | 0.057 |
| 1,000 | 1.697 | -0.369 | 0.535 | 0.409 | 0.058 |
| 2,000 | 1.822 | -0.392 | 0.564 | 0.562 | 0.055 |
| 2,670 | 1.859 | -0.379 | 0.562 | **0.725** | 0.055 |

Three points survive the move from simulation to historical data.

1. **Discovery capacity and selection capacity separate sharply.** The ex-post reachable frontier rises from about 1.05 to 1.86 as the candidate pool expands, while development/future rank correlation remains only about 0.04–0.06.
2. **A weak selector can become actively worse as search expands.** The one-fold winner moves from -0.09 at 25 candidates to roughly -0.38 at the full zoo, despite the reachable frontier improving.
3. **Better selection recovers only part of the frontier.** Four-fold selection reaches about 0.56 at full search. A simple diversity-aware five-candidate ensemble reaches 0.73, but is inferior at small budgets; diversification is not free when the underlying candidate set is weak.

At the full 2,670-candidate budget several deterministic selectors happen to select the same candidate, so their identical rows are not independent confirmation. The nested complexity penalty also does not improve on ordinary multi-fold selection in this run.

This is one historical period and one public universe, not evidence of a universal selector. The adjusted-price snapshot is pinned and reproducible, but was assembled retrospectively by the upstream source; that is different from a contemporaneous point-in-time market database. The next stress phase varies time periods and universes rather than treating 2018–2019 as a final answer.

Canonical compact outputs are committed in `results/market_summary.csv`, `results/market_ensemble_summary.csv` and `results/market_run.json`. The full per-seed ledger and candidate manifest are produced as GitHub Actions artifacts.

## Experiment 5 — period and universe stress

Phase 4's 2018–2019 result is not treated as the answer. The same frozen
2,670-candidate search language is stressed across four rolling two-year test
periods and 12 fixed asset universes.

Every one of the 48 cells has a higher mean ex-post oracle frontier at the full
2,670-candidate budget than at 25 candidates. But selection frequently fails to
keep up:

| selector | selector-limited cells | mean selected change | paired degradation |
|---|---:|---:|---:|
| family-balanced | 17 / 48 | **+0.104** | 43.1% |
| mean rank | 20 / 48 | -0.018 | 47.8% |
| stability LCB | 20 / 48 | -0.031 | 47.8% |
| mean fold | 25 / 48 | -0.183 | 53.5% |
| single fold | **27 / 48** | **-0.233** | **58.5%** |

The failure is strongly regime-dependent. In 2016–2017, mean-fold selection is
selector-limited in **12 / 12** universes and its selected future Sharpe falls
by 0.945 on average as search expands. In 2018–2019, the same selector improves
in **10 / 12** universes. That contrast is the key qualification to Experiment
4: repeated validation can help substantially, but not as a universal rule.

The five-candidate diversity ensemble is reported separately as a mitigation,
not as a selector-regret datapoint. It is positive in 33 / 48 full-budget
cells, but is negative in every 2016–2017 universe.

**Phase 5 verdict: conditional support.** Search expansion robustly improves the
reachable set; selector capacity is often the bottleneck, but the sign and size
of that bottleneck depend on regime and selector.

See [PHASE5_RESULT.md](PHASE5_RESULT.md) for the full audit and result.

## Reproduce

```bash
python -m pip install -e ".[dev]"
pytest -q

python -m selector_limitation.experiment --out results/synthetic_sweep.csv
python -m selector_limitation.family_experiment --out results/family_search_sweep.csv
python -m selector_limitation.selector_experiment --out results/selector_sweep.csv
python -m selector_limitation.market_experiment --search-seeds 100
python -m selector_limitation.stress_experiment --search-seeds 100
```

The market command downloads only the pinned snapshot, verifies its hash, truncates it to the frozen cutoff and regenerates the candidate zoo and summaries.
