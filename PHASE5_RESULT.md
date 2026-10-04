# Phase 5 result — period/universe stress and failure mapping

**Status: COMPLETE**

**Classification: CONDITIONAL SUPPORT.**

Phase 5 strengthens the selector-limitation result, but it also rules out the
stronger claim that more search mechanically worsens every selector in every
market regime.

## Frozen experiment

The Phase 4 candidate language is unchanged: 2,670 generic price-only
candidates, 5 bps turnover costs, date-`t` information earning only
date-`t+1` returns, and the same pinned adjusted-close snapshot truncated at
2019-12-31.

Phase 5 evaluates four rolling two-year test periods after four years of
development evidence:

- P12_13: test 2012–2013;
- P14_15: test 2014–2015;
- P16_17: test 2016–2017;
- P18_19: test 2018–2019.

Each period is crossed with 12 fixed universes: the full eight-asset panel,
several conceptual subsets, and one-at-a-time non-SPY removals. Partial search
uses 100 paired candidate orderings at budgets 25, 100, 500, 1,000 and the full
2,670 candidates.

The original Phase 4 setup is retained as a reproduction control. It reproduces
the full-zoo values to floating-point tolerance:

- oracle test Sharpe: **1.8589463473**;
- mean-fold selected test Sharpe: **0.5618176773**.

## Result 1 — the reachable frontier expands robustly

Across **all 48 period × universe cells**, the mean ex-post oracle frontier is
higher at 2,670 candidates than at 25 candidates.

Across the paired search paths, the frontier expands on **98.8%** of
seed/scenario comparisons on average. The mean cell-level increase in oracle
test Sharpe is **+0.841**.

This is the cleanest Phase 5 result: the larger search space really does contain
better candidates ex post. The open question is whether a selector can identify
them from development evidence.

## Result 2 — selection often fails to keep up

The sign-based expansion map labels a cell `SELECTOR_LIMITED` when the ex-post
frontier rises but the selector's mean future Sharpe falls between budget 25 and
the full zoo.

| selector | selector-limited cells | mean selected Sharpe change | paired degradation rate | positive full-budget cells |
|---|---:|---:|---:|---:|
| family_balanced | 17 / 48 | **+0.104** | 43.1% | 31 / 48 |
| mean_rank | 20 / 48 | -0.018 | 47.8% | **32 / 48** |
| stability_lcb | 20 / 48 | -0.031 | 47.8% | 30 / 48 |
| mean_fold | 25 / 48 | -0.183 | 53.5% | 28 / 48 |
| nested_complexity | 25 / 48 | -0.183 | 53.5% | 28 / 48 |
| single_fold | **27 / 48** | **-0.233** | **58.5%** | 25 / 48 |

Two conclusions follow.

First, the Phase 4 one-fold failure is not a one-period curiosity: single-fold
selection is selector-limited in a majority of the stress cells.

Second, repeated validation is not a universal cure. Mean-fold selection is
better than one-fold selection in some regimes, but the larger search still
reduces its selected future Sharpe in 25 of 48 cells.

Family balancing is the most robust of the tested corrections by this specific
expansion criterion, but it still degrades on 43% of paired paths and fails in
17 cells. No tested selector removes the bottleneck.

## Result 3 — the failure is strongly regime-dependent

The most important negative result is P16_17.

For **mean-fold selection**, all 12 P16_17 universes are
`SELECTOR_LIMITED`. The mean selected-Sharpe change from 25 to 2,670
candidates is **-0.945**, and selection deteriorates on **84.3%** of paired
search paths. The mean development-to-test rank correlation at full search is
only **0.113**.

Family balancing reduces but does not remove the problem: 8 of 12 P16_17 cells
remain selector-limited.

P18_19 is almost the mirror image and explains why Phase 4 looked more
encouraging. In that period:

- single-fold selection is selector-limited in **11 / 12** universes and
  deteriorates on **80.9%** of paired paths;
- mean-fold selection improves in **10 / 12** universes, with mean selected
  Sharpe change **+0.474** and only **21.1%** paired deterioration;
- mean-rank selection also improves in **10 / 12**, with mean change **+0.522**.

So repeated validation genuinely helps in the 2018–2019 regime, but Phase 5
shows that this cannot be generalized into a universal selector claim.

## Result 4 — global rank correlation is informative but insufficient

Full-budget development-to-test rank correlation varies strongly by period:

- P12_13: ~0.327;
- P14_15: ~0.382;
- P16_17: ~0.113;
- P18_19: ~0.054.

Yet P18_19 gives good mean-fold selection while P16_17 gives very poor
mean-fold selection. A single global rank-correlation statistic therefore does
not fully characterize top-of-list selection quality. The failure mechanism
must depend on more than overall ordering fidelity.

## Result 5 — the ensemble is a mitigation, not a solution

The diversity-aware five-candidate ensemble is kept separate from
single-candidate selection regret because it is a different decision object.

At the full search budget it:

- is positive in **33 / 48** cells;
- beats mean-fold selection in **24 / 48** cells;
- has mean advantage over mean-fold of **+0.074 Sharpe** and median advantage
  of only **+0.008**.

Its failure is also regime-specific. In P16_17 it is negative in **all 12**
universes, with mean test Sharpe **-1.364**. In P18_19 it is positive in
11 of 12 universes but trails mean-fold selection by **0.174 Sharpe on average**.

Diversification therefore helps in some cells but is not a generic repair for
selection failure.

## Result 6 — bootstrap confidence does not solve the regime problem

The block-bootstrap LCB selector was separately stressed on FULL8 using the same
fixed resampling design as Phase 4:

| period | selected test Sharpe |
|---|---:|
| P12_13 | +0.541 |
| P14_15 | +0.029 |
| P16_17 | -0.779 |
| P18_19 | -0.580 |

The heavier uncertainty-aware selector therefore also fails to provide a
period-invariant solution.

## Interpretation

The evidence now supports a narrower and more useful claim:

> **Expanding a hypothesis space can reliably improve what is reachable while
> making the research problem harder to solve. Whether realised decisions
> improve depends on the selector and the regime.**

This is stronger than saying "multiple testing is bad" because the experiment
separates three objects: search capacity, reachable quality and selected
quality. It is weaker than claiming a universal monotonic law: several
selectors improve in many cells, and family balancing has positive mean
expansion performance across the grid.

The practical bottleneck has therefore moved again. The next research question
is not simply how to generate more candidates, but **what observable properties
of a development regime predict which selector will generalize**.

## Audit trail

The original Phase 5 grid and failure-bin thresholds were frozen before the
initial stress result. A 2026-10-04 audit then added a paired, sign-based
expansion map and separated ensemble mitigation from single-candidate selection
regret. That audit did not alter the candidate zoo, data, returns, search
orderings or selector scores.

Canonical outputs:

- `results/stress_summary.csv`
- `results/expansion_map.csv`
- `results/failure_map.csv`
- `results/stress_selector_summary.csv`
- `results/ensemble_stress.csv`
- `results/bootstrap_period_stress.csv`
- `results/stress_run.json`

The full per-seed ledger remains a workflow artifact rather than Git history.
