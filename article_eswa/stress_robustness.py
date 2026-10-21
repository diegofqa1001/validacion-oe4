"""Stress-window robustness and market descriptors for the article.

1. Identifies the 252-day worst window with two rules:
   (a) price-level average (the rule of src/oe4/stability.worst_window), which is a
       price-weighted index;
   (b) equally weighted return index (cumulative product of the cross-sectional
       mean daily return).
   For each window it re-runs the engine at quarterly rebalances (same grid rule as
   stability.stress_grid) and reports the Spearman correlation between orness and
   mean realised volatility, the quarterly correlations, and descriptors of the
   investable universe at each rebalance: sigma_def, sigma_agg, their ratio, the mean
   pairwise correlation of daily returns over the holding period and the
   cross-sectional dispersion of asset volatilities.
2. Market descriptors over the full sample: mean pairwise correlation of daily
   returns, share of exactly zero daily returns.

Requires owa-recommender >= 2.1.0 (numerically identical to the engine used in
results/; see range_diagnostics.py). Run from the repository root:
    python article_eswa/stress_robustness.py
Outputs: article_eswa/output/stress_robustness.csv, stress_robustness_quarters.csv,
         market_descriptors.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from owa_recommender.config import EngineConfig
from owa_recommender.portfolio import PortfolioBuilder
from owa_recommender.profiles import all_profiles

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data" / "snapshot_oe4"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
W, ANN = 252, 252


def worst_price_average(px):
    idx = px.mean(axis=1)
    ret = idx / idx.shift(W) - 1.0
    end = int(np.nanargmin(ret.values))
    return max(0, end - W), end, float(ret.iloc[end])


def worst_equal_weight(px):
    r = px.pct_change().mean(axis=1).fillna(0.0)
    idx = (1 + r).cumprod()
    ret = idx / idx.shift(W) - 1.0
    end = int(np.nanargmin(ret.values))
    return max(0, end - W), end, float(ret.iloc[end])


def mean_pair_corr(rets):
    c = rets.corr().values
    iu = np.triu_indices_from(c, 1)
    return float(np.nanmean(c[iu]))


cfg = EngineConfig()
profiles = all_profiles()
alphas = [p.alpha for p in profiles]
rows, qrows, desc = [], [], []
for mkt in ["us", "co"]:
    px = pd.read_csv(SNAP / f"{mkt}_precios.csv", index_col=0, parse_dates=True)
    rets_all = px.pct_change().dropna(how="all")
    desc.append({"market": mkt.upper(), "assets": px.shape[1], "days": px.shape[0],
                 "mean_pairwise_corr_full": mean_pair_corr(rets_all),
                 "share_zero_daily_returns": float((rets_all == 0).values.mean())})
    b = PortfolioBuilder(px, cfg)
    for rule, fn in (("price_average", worst_price_average), ("equal_weight", worst_equal_weight)):
        a, e, drop = fn(px)
        grid = list(range(max(cfg.lookback, a), e, cfg.horizon))
        vols = {p.name: [] for p in profiles}
        for t in grid:
            t2 = min(t + cfg.horizon, e)
            s_def, s_agg = b.common_vol_range(t)
            hold = px.iloc[t:t2 + 1].pct_change().dropna()
            v_q = []
            for p in profiles:
                port = b.build(p, t)
                seg = px[port.weights.index].iloc[t:t2 + 1].pct_change().dropna()
                v = float((seg @ port.weights.values).std() * np.sqrt(ANN))
                vols[p.name].append(v); v_q.append(v)
            asset_vol = b._window(t).std() * np.sqrt(ANN)
            qrows.append({"market": mkt.upper(), "rule": rule, "start": str(px.index[t].date()),
                          "end": str(px.index[t2].date()), "rho_realised": spearmanr(alphas, v_q)[0],
                          "sigma_def": s_def, "sigma_agg": s_agg, "agg_over_def": s_agg / s_def,
                          "mean_pair_corr_holding": mean_pair_corr(hold),
                          "asset_vol_cv_estimation": float(asset_vol.std() / asset_vol.mean())})
        mean_v = [float(np.mean(vols[p.name])) for p in profiles]
        rows.append({"market": mkt.upper(), "rule": rule, "start": str(px.index[a].date()),
                     "end": str(px.index[e].date()), "index_drop": drop, "rebalances": len(grid),
                     "rho_mean_realised_vol": spearmanr(alphas, mean_v)[0],
                     **{f"vol_{p.name}": v for p, v in zip(profiles, mean_v)}})
pd.DataFrame(rows).to_csv(OUT / "stress_robustness.csv", index=False)
pd.DataFrame(qrows).to_csv(OUT / "stress_robustness_quarters.csv", index=False)
pd.DataFrame(desc).to_csv(OUT / "market_descriptors.csv", index=False)
with pd.option_context("display.width", 250, "display.precision", 3):
    print(pd.DataFrame(rows).iloc[:, :7]); print(pd.DataFrame(qrows)); print(pd.DataFrame(desc))
