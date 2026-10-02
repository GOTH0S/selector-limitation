# selector-limitation

More search is only useful if selection keeps up.

Suppose a research process generates `N` candidate signals. Candidate `i` has an unknown true quality `q_i`, but the researcher only sees a noisy validation estimate `v_i`. As `N` grows, the best candidate the process *could* have selected can improve while the candidate it *actually* selects becomes increasingly dominated by validation noise.

This repository studies that gap directly.

The experiments are synthetic first so the latent quality of every candidate is known. That lets us separate two quantities that are usually conflated in backtests:

- **reachable frontier** — the true quality of the best candidate generated;
- **selected quality** — the true quality of the candidate chosen using validation data only.

Their difference is **selection regret**.

## Experiment 1 — independent candidates

The baseline sweep varies candidate count and validation noise under a fixed data-generating process. With validation noise fixed at `2.0`, 2,000 seeds give:

| candidates | oracle quality | selected quality | regret | oracle hit rate |
|---:|---:|---:|---:|---:|
| 10 | 1.529 | 0.720 | 0.809 | 27.5% |
| 100 | 2.501 | 1.133 | 1.368 | 9.5% |
| 1,000 | 3.239 | 1.445 | 1.794 | 4.0% |

The selector benefits from more search in absolute terms, but it captures the expanding frontier much more slowly.

## Experiment 2 — correlated families and adaptive search

Independent candidates are too generous. Real research tends to generate related variants of the same idea, and those variants often share the same validation error.

The second experiment gives each candidate family a latent true mean `mu_f` and a persistent validation bias `b_f`:

```text
candidate quality = mu_f + within-family variation
validation score  = candidate quality + b_f + observation noise
```

Every run begins with two candidates from each of eight families. Two search policies then spend the remaining budget:

- `uniform`: choose the next family uniformly;
- `winner_following`: with 95% probability, generate the next candidate from the family containing the current validation winner.

The family bias is deliberately persistent: repeated variants of the same research idea can share the same misspecification rather than providing independent evidence.

At a budget of 1,000 candidates across 2,000 seeds:

| search | selected quality | oracle quality | regret | selected family is truly best | allocation HHI | effective families |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 0.913 | 2.340 | 1.428 | 28.5% | 0.126 | 7.94 |
| winner-following | 0.982 | 2.190 | 1.208 | 29.6% | 0.874 | 1.16 |

Winner-following does **not** make mean selected quality fall with budget in this model. It slightly improves the selected candidate, but it collapses the effective search breadth from almost eight families to roughly one and lowers the reachable frontier relative to uniform search. Persistent validation bias also remains unresolved: the selected family is the truly best family less than one-third of the time.

That negative result is useful. Correlation and adaptive exploitation alone are not enough to establish the stronger claim that additional search worsens realised decisions. The next experiments should distinguish *search allocation failure* from *selector failure* rather than tune the simulator until degradation appears.

## Run

```bash
python -m pip install -e ".[dev]"
pytest -q
python -m selector_limitation.experiment --out results/synthetic_sweep.csv
python -m selector_limitation.family_experiment --out results/family_search_sweep.csv
```

Both sweeps write machine-readable CSV artifacts. Latent quality is used only for evaluation; neither search policy nor the selector can inspect it.
