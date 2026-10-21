"""Diagnostics of Stage 2 (risk control) on the evaluation grid.

For every market and quarterly window it records
  * whether the common range needed the safety bound (sigma_agg <= sigma_def),
  * for each profile, the volatility range attainable by mixing its own
    defensive and aggressive funds, and whether the target lies inside it.
This explains why the attained ex-ante volatility can differ from the target.

Requires the engine package (owa-recommender >= 2.1.0, numerically identical to
motor-owa-v2): pip install owa-recommender  or  pip install -e <path to owa-recommender>
Run from the repository root:  python article_eswa/range_diagnostics.py
Output: article_eswa/output/range_diagnostics.csv and range_diagnostics_summary.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from owa_recommender.config import EngineConfig
from owa_recommender.portfolio import PortfolioBuilder, _cap_and_norm, _portfolio_vol
from owa_recommender.profiles import all_profiles

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data" / "snapshot_oe4"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)


def load(mkt):
    px = pd.read_csv(SNAP / f"{mkt}_precios.csv", index_col=0, parse_dates=True)
    return px


rows = []
cfg = EngineConfig()
for mkt in ["us", "co"]:
    px = load(mkt)
    b = PortfolioBuilder(px, cfg)
    t_grid = list(range(cfg.lookback, len(px) - cfg.horizon, cfg.horizon))
    for t in t_grid:
        before = b.vol_range_violations
        s_def, s_agg = b.common_vol_range(t)
        violated = b.vol_range_violations > before
        for p in all_profiles():
            port = b.build(p, t)
            # attainable range of this profile's own fund mix
            scores = b.owa_scores(p, t)
            rets_u = b._window(t)[scores.index.tolist()]
            vols = rets_u.std() * np.sqrt(252)
            med = float(vols.median())
            low_b, high_b = vols[vols <= med].index, vols[vols > med].index
            n_side = max(3, cfg.top_n // 2)
            sel_def = scores[low_b].nlargest(min(n_side, len(low_b))).index.tolist()
            sel_agg = scores[high_b].nlargest(min(n_side, len(high_b))).index.tolist()
            assets = list(dict.fromkeys(sel_def + sel_agg))
            rets = rets_u[assets]; cov = rets.cov().values
            idx = {a: i for i, a in enumerate(assets)}
            w_def = np.zeros(len(assets)); w_agg = np.zeros(len(assets))
            for a in sel_def:
                w_def[idx[a]] = 1.0 / max(float(rets_u[a].std()), 1e-9)
            smin_sc = float(scores[assets].min())
            for a in sel_agg:
                w_agg[idx[a]] = (float(scores[a]) - smin_sc + 1e-9) * float(vols[a])
            w_def = _cap_and_norm(w_def, cfg.max_weight); w_agg = _cap_and_norm(w_agg, cfg.max_weight)
            grid = [_portfolio_vol(_cap_and_norm(th * w_agg + (1 - th) * w_def, cfg.max_weight), cov)
                    for th in np.linspace(0, 1, 201)]
            lo, hi = min(grid), max(grid)
            rows.append({"market": mkt.upper(), "t": t, "profile": p.name, "alpha": p.alpha,
                         "common_range_bound_applied": violated, "sigma_def": s_def, "sigma_agg": s_agg,
                         "target_vol": port.target_vol, "attained_vol": port.expected_vol,
                         "own_min": lo, "own_max": hi,
                         "target_below_own_range": port.target_vol < lo - 1e-9,
                         "target_above_own_range": port.target_vol > hi + 1e-9})
d = pd.DataFrame(rows)
d.to_csv(OUT / "range_diagnostics.csv", index=False)
s = d.groupby("market").agg(
    windows=("t", "nunique"),
    bound_applied_windows=("common_range_bound_applied", lambda x: int(x.groupby(d.loc[x.index, "t"]).first().sum())),
    cells=("t", "size"),
    target_below_range=("target_below_own_range", "mean"),
    target_above_range=("target_above_own_range", "mean"))
s.to_csv(OUT / "range_diagnostics_summary.csv")
print(s)
print(d.groupby(["market", "profile"])[["target_below_own_range", "target_above_own_range"]].mean().round(3))
