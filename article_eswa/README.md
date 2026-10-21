# Article materials — profile-ordered portfolio recommendation as a decision support artefact

This folder holds the analyses and figures added for the journal article that evaluates the
recommendation engine as a decision support system. It reuses the versioned outputs of this
repository (`results/`) and adds one new analysis: suitability measured **window by window**
at three layers.

| Layer | Quantity | Column in `results/*_registros_motor.csv` |
|---|---|---|
| 1. Target | target volatility set by design, sigma_k = sigma_def + alpha_k (sigma_agg − sigma_def) | `target_vol` |
| 2. Attained ex ante | volatility of the portfolio actually built, estimated covariance | `expected_vol` |
| 3. Realised ex post | annualised volatility over the 63-day holding period | `vol` |

## Reproduce

From the repository root, after `scripts/run_oe4.py` (or with the versioned `results/`):

```bash
python article_eswa/window_coherence.py   # tables in article_eswa/output/
python article_eswa/make_fig1.py          # architecture diagram
python article_eswa/make_figures.py       # Figs. 2-4
```

Requirements: numpy, pandas, scipy, matplotlib (see `requirements.txt`).

## Outputs

- `output/window_coherence.csv` — Spearman correlation between orness and volatility, and share of
  the 28 profile pairs correctly ordered, for every market × quarterly window and layer.
- `output/window_coherence_summary.csv` — summary per market and layer (mean, median, share of
  perfectly ordered windows, one-sided Wilcoxon and sign tests of rho > 0).
- `output/attained_vs_realised.csv` — realised order conditional on a perfect attained order.
- `output/attainment_gap.csv`, `output/attainment_misses.csv` — gap between attained and target volatility.
- `figures/Fig1`–`Fig4` (.png and .tif at 600 dpi, .eps).

## Main numbers

| | United States (43 windows) | Colombia (40 windows) |
|---|---|---|
| Mean window rho, target | 1.000 | 1.000 |
| Mean window rho, attained ex ante | 0.931 (35% perfect) | 0.962 (57% perfect) |
| Mean window rho, realised | 0.841 (21% perfect) | 0.555 (12% perfect) |
| Profile pairs correctly ordered, realised | 88.5% | 74.7% |
| Windows with negative realised rho | 2.3% | 15.0% |
| Sign test, realised rho > 0 | p = 5.0e-12 | p = 4.2e-06 |
| Profile-window cells with attained volatility > 1% away from target | 30.8% | 17.2% |

Universes are made of issuers listed at the download date (survivorship bias, declared in the article).

## Stage 2 diagnostics (`range_diagnostics.py`)

Re-runs Stage 2 on the versioned snapshot (`data/snapshot_oe4/`) with `owa-recommender`
(numerically identical to the versioned records: maximum absolute difference 0.0) and
records, per profile and window, the volatility range attainable with the profile's own
defensive and aggressive funds.

| | United States | Colombia |
|---|---|---|
| Windows where the common range needed the safety bound | 0 of 43 | 0 of 40 |
| Cells with target below the profile's attainable range | 11.3% (Guardian 65.1%, Sentinel 23.3%) | 10.6% (Guardian 55.0%, Sentinel 25.0%) |
| Cells with target above the profile's attainable range | 21.2% (Innovator 46.5%, Adventurer 41.9%) | 8.1% (Visionary 15.0%) |

## Stress robustness and market descriptors (`stress_robustness.py`)

The stress window of `src/oe4/stability.worst_window` is the 252-day window of largest fall of the
simple average of adjusted price levels (a price-weighted index). The script repeats the stress test
with the window of largest fall of an equally weighted return index and records, per rebalance,
sigma_agg / sigma_def and the mean pairwise correlation of daily returns over the holding period.

| | Price-level average | Equally weighted returns |
|---|---|---|
| United States window | 2022-01-04 to 2023-01-05, rho = 1.000 | 2021-12-28 to 2022-12-28, rho = 1.000 |
| Colombia window | 2019-02-06 to 2020-03-19, rho = 0.310 | 2022-02-25 to 2023-03-23, rho = 0.405 |

Full-sample descriptors: mean pairwise correlation of daily returns 0.359 (US) and 0.189 (CO);
share of exactly zero daily returns 0.4% (US) and 15.8% (CO). The Colombian universe holds 17 shares of
16 issuers (ordinary and preferred shares of Grupo Cibest).

`window_coherence.py` also writes `output/block_bootstrap.csv`: moving-block bootstrap (blocks of four
windows, 10,000 replications) of the mean realised rho and of the share of correctly ordered pairs.
