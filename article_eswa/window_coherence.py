"""Window-level suitability analysis for the ESWA article.

Reads the versioned per-window records produced by scripts/run_oe4.py
(results/{us,co}_registros_motor.csv) and measures, for every quarterly
evaluation window, whether the eight profile portfolios are ordered by
orness at three levels:

    target   - target volatility sigma_k set by the design (ex ante)
    attained - ex-ante volatility actually attained by the built portfolio
    realised - realised volatility over the 63-day holding period

Outputs (article_eswa/output/):
    window_coherence.csv          one row per market x window
    window_coherence_summary.csv  summary per market and level
    attainment_gap.csv            |attained - target| per profile and market
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, binomtest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

LEVELS = {"target": "target_vol", "attained": "expected_vol", "realised": "vol"}


def pair_share(alpha, x):
    """Share of the 28 profile pairs whose volatility is ordered as orness."""
    ok = tot = 0
    for i, j in combinations(range(len(alpha)), 2):
        tot += 1
        if (alpha[i] - alpha[j]) * (x[i] - x[j]) > 0:
            ok += 1
    return ok / tot


rows = []
for mkt in ["us", "co"]:
    d = pd.read_csv(RES / f"{mkt}_registros_motor.csv")
    for t, g in d.groupby("t"):
        g = g.sort_values("alpha")
        rec = {"market": mkt.upper(), "t": t, "slice": g["slice"].iloc[0]}
        for lev, col in LEVELS.items():
            rec[f"rho_{lev}"] = spearmanr(g.alpha, g[col])[0]
            rec[f"pairs_{lev}"] = pair_share(g.alpha.values, g[col].values)
        rows.append(rec)
win = pd.DataFrame(rows)
win.to_csv(OUT / "window_coherence.csv", index=False)

summ = []
for mkt, g in win.groupby("market"):
    for lev in LEVELS:
        r = g[f"rho_{lev}"]
        p = g[f"pairs_{lev}"]
        # one-sided Wilcoxon signed-rank test of median rho > 0
        w_p = wilcoxon(r, alternative="greater").pvalue if r.nunique() > 1 else np.nan
        summ.append({
            "market": mkt, "level": lev, "windows": len(r),
            "rho_mean": r.mean(), "rho_median": r.median(),
            "rho_q25": r.quantile(.25), "rho_min": r.min(),
            "share_perfect": (r > 0.9999).mean(),
            "share_ge_0.9": (r >= 0.9).mean(),
            "share_negative": (r < 0).mean(),
            "pairs_correct_mean": p.mean(),
            "wilcoxon_p_rho_gt_0": w_p,
            # sign test: windows with rho > 0 vs < 0
            "sign_test_p": binomtest(int((r > 0).sum()), int((r != 0).sum()),
                                     0.5, alternative="greater").pvalue,
        })
summ = pd.DataFrame(summ)
summ.to_csv(OUT / "window_coherence_summary.csv", index=False)

# does realised order fail more when the attained ex-ante order already fails?
cond = []
for mkt, g in win.groupby("market"):
    perfect = g.rho_attained > 0.9999
    cond.append({"market": mkt,
                 "n_attained_perfect": int(perfect.sum()),
                 "realised_rho_if_attained_perfect": g.loc[perfect, "rho_realised"].mean(),
                 "realised_rho_otherwise": g.loc[~perfect, "rho_realised"].mean(),
                 "corr_attained_realised": spearmanr(g.rho_attained, g.rho_realised)[0]})
pd.DataFrame(cond).to_csv(OUT / "attained_vs_realised.csv", index=False)

gap = []
for mkt in ["us", "co"]:
    d = pd.read_csv(RES / f"{mkt}_registros_motor.csv")
    d["abs_gap"] = (d.expected_vol - d.target_vol).abs()
    d["rel_gap"] = d.abs_gap / d.target_vol
    d["realised_over_target"] = d.vol / d.target_vol
    s = d.groupby("profile", sort=False)[["abs_gap", "rel_gap", "realised_over_target"]].median()
    s.insert(0, "market", mkt.upper())
    gap.append(s.reset_index())
pd.DataFrame(pd.concat(gap)).to_csv(OUT / "attainment_gap.csv", index=False)

with pd.option_context("display.width", 200, "display.precision", 3):
    print(summ)
    print(pd.DataFrame(cond))
    print(pd.concat(gap))

# share of profile-window cells where the attained ex-ante volatility misses the target by > 1%
miss = []
for mkt in ["us", "co"]:
    d = pd.read_csv(RES / f"{mkt}_registros_motor.csv")
    rel = (d.expected_vol - d.target_vol) / d.target_vol
    bad = rel.abs() > 0.01
    miss.append({"market": mkt.upper(), "cells": len(d), "miss_gt_1pct": int(bad.sum()),
                 "share": bad.mean(), "share_below_target": (rel[bad] < 0).mean(),
                 "guardian_share_of_misses": (d.loc[bad, "profile"] == "Guardian").mean()})
pd.DataFrame(miss).to_csv(OUT / "attainment_misses.csv", index=False)
print(pd.DataFrame(miss))

# moving-block bootstrap (blocks of 4 consecutive windows = one year) for the realised layer,
# to account for persistence of volatility regimes and overlapping estimation windows
rng = np.random.default_rng(20261021)
boot = []
for mkt, g in win.groupby("market"):
    g = g.sort_values("t")
    for col in ["rho_realised", "pairs_realised"]:
        x = g[col].values; n = len(x); L = 4
        starts = np.arange(n - L + 1)
        means = []
        for _ in range(10000):
            idx = np.concatenate([np.arange(s, s + L) for s in rng.choice(starts, int(np.ceil(n / L)))])[:n]
            means.append(x[idx].mean())
        means = np.array(means)
        ref = 0.0 if col == "rho_realised" else 0.5
        boot.append({"market": mkt, "statistic": col, "mean": x.mean(),
                     "ci95_low": np.quantile(means, 0.025), "ci95_high": np.quantile(means, 0.975),
                     "share_boot_le_chance": float((means <= ref).mean())})
pd.DataFrame(boot).to_csv(OUT / "block_bootstrap.csv", index=False)
print(pd.DataFrame(boot))
