# Phase 5 protocol — period/universe stress and failure mapping

This phase is specified before inspecting any Phase 5 result.

## Question

Phase 4 showed that expanding a fixed candidate zoo could improve the ex-post
reachable frontier while leaving the validation-to-future ordering weak. Phase 5
asks whether that separation is specific to one historical window or one asset
mix.

The unit of analysis is **selector behaviour conditional on a fixed search
language**. Phase 5 does not add new signal families, tune thresholds against a
new period, or search for a more favourable universe.

## Frozen objects

The following remain unchanged from Phase 4:

- the 2,670 deterministic candidate specifications;
- signal definitions, lookbacks, transforms, market-state gates and rebalance
  intervals;
- 5 bps cost per unit of turnover;
- date-t information earning only date-t+1 returns;
- the pinned public price snapshot and SHA-256 check;
- selector implementations;
- random candidate ordering by integer seed.

No Phase 5 result may change those objects.

## Period stress

To avoid giving later tests more development evidence simply because more
history exists, the comparative regime grid uses a **constant four calendar
years of development data followed by two calendar years of unseen test data**.
Each development window is split into two chronological two-year folds.

| label | development fold 1 | development fold 2 | unseen test |
|---|---|---|---|
| P12_13 | 2008-01-02–2009-12-31 | 2010-01-04–2011-12-30 | 2012-01-03–2013-12-31 |
| P14_15 | 2010-01-04–2011-12-30 | 2012-01-03–2013-12-31 | 2014-01-02–2015-12-31 |
| P16_17 | 2012-01-03–2013-12-31 | 2014-01-02–2015-12-31 | 2016-01-04–2017-12-29 |
| P18_19 | 2014-01-02–2015-12-31 | 2016-01-04–2017-12-29 | 2018-01-02–2019-12-31 |

The original Phase 4 2018–2019 experiment, which used four development folds
covering 2010–2017, is retained separately as a **reproduction control**. It is
not mixed into the fixed-window regime aggregates.

## Universe stress

Every universe keeps SPY as its first element so the already-frozen market-state
gates retain exactly the same reference asset semantics.

Conceptual slices:

- FULL8: SPY, QQQ, IWM, EFA, EEM, TLT, GLD, DBC
- US3: SPY, QQQ, IWM
- GLOBAL_EQ5: SPY, QQQ, IWM, EFA, EEM
- CROSS_ASSET4: SPY, TLT, GLD, DBC
- EQ_BOND6: SPY, QQQ, IWM, EFA, EEM, TLT
- EQ_REAL7_LOO_TLT: SPY, QQQ, IWM, EFA, EEM, GLD, DBC

Systematic sensitivity slices additionally remove one non-SPY asset at a time
from FULL8. The TLT removal is identical to `EQ_REAL7_LOO_TLT`, so that unique
asset set is counted once while serving both roles. SPY is never removed because
doing so would alter the meaning of the frozen state gates.

## Search budgets

Partial search uses 100 seeded candidate orderings at budgets:

`25, 100, 500, 1,000, 2,670`.

The full 2,670-candidate budget is deterministic with respect to candidate
membership; repeated seeded rows are retained only so aggregation code has the
same shape at every budget.

## Selectors

The factorial period × universe grid evaluates the single-candidate selectors:

- single_fold
- mean_fold
- mean_rank
- stability_lcb
- family_balanced
- nested_complexity

The diversity-aware ensemble is retained as a **separate mitigation diagnostic**
at the full candidate budget. It is not classified with the single-candidate
selectors because an ensemble is a different decision object and its Sharpe
should not be interpreted as selection regret against a single-candidate
oracle.

The block-bootstrap LCB is computationally heavier and stochastic conditional
on its resampling seed. It is therefore stress-tested across all four **period**
windows on FULL8 with the same 128-draw implementation and fixed bootstrap seed
used in Phase 4, but it is not included in the 12-universe factorial grid. This
is an explicit scope choice, not a result-driven omission.

## Expansion map

The central Phase 5 diagnostic pairs every search seed at budget 25 with the
same seed at the full 2,670-candidate budget. For each period, universe and
single-candidate selector it records:

- the change in the ex-post reachable frontier;
- the change in selected test Sharpe;
- the fraction of paired search paths on which selection deteriorates;
- the change in selection regret;
- development-to-test rank correlation at both endpoints.

The classification is deliberately sign-based rather than threshold-tuned.
`SELECTOR_LIMITED` means the reachable frontier increased while selected
quality fell. `FRONTIER_AND_SELECTION_UP` means both improved. Other labels
cover the remaining sign combinations.

## Failure map

A complementary failure map uses only the full 2,670-candidate budget and the
four fixed rolling period windows. It covers **single-candidate selectors
only**. Bins are descriptive diagnostics, not significance tests:

- `NO_POSITIVE_FRONTIER`: ex-post oracle Sharpe <= 0;
- `NEGATIVE_SELECTION`: oracle > 0 but selected Sharpe <= 0;
- `INVERTED`: selected Sharpe is below the mean candidate, so frontier
  efficiency < 0;
- `WEAK_CAPTURE`: 0 <= efficiency < 0.25;
- `PARTIAL_CAPTURE`: 0.25 <= efficiency < 0.50;
- `STRONG_CAPTURE`: efficiency >= 0.50.

A separate `low_identifiability` flag is true when the absolute development
score / future score rank correlation is below 0.10.

The ex-post oracle is never a selectable strategy. It exists only to measure
what the frozen search happened to contain.

## Phase 4 reproduction invariant

On FULL8 with the original 2010–2017 four-fold development window and the
2018–2019 test, Phase 5 must reproduce the Phase 4 full-zoo values within
floating-point tolerance:

- oracle test Sharpe: 1.8589463473419494
- mean-fold selected test Sharpe: 0.5618176772530484

Failure of this invariant invalidates the Phase 5 run.

## Outputs

Canonical compact outputs:

- `results/stress_summary.csv`
- `results/expansion_map.csv`
- `results/failure_map.csv`
- `results/stress_selector_summary.csv`
- `results/ensemble_stress.csv`
- `results/bootstrap_period_stress.csv`
- `results/stress_run.json`

The per-seed stress ledger is a workflow artifact rather than permanent Git
history.
