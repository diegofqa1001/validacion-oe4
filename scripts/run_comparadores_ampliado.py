"""Comparadores ampliados (verificacion + validacion) con checkpoint por tramo.

Uso: python scripts/run_comparadores_ampliado.py <mercado co|us> <tramo ve|va>
Al final (tras correr ambos tramos): python scripts/run_comparadores_ampliado.py <mercado> resumen
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "motor-owa-v2", "src"))

import numpy as np, pandas as pd
from motor_owa.config import EngineConfig
from motor_owa.engine import RecommendationEngine
from motor_owa.validation import split_70_20_10
from oe4.benchmarks import equal_weight, min_variance, max_sharpe, mlp_portfolio, anfis_portfolio
from oe4.pipeline import _perf
from oe4.inference import diebold_mariano

mkt, tramo = sys.argv[1], sys.argv[2]
RES = os.path.join(HERE, "..", "results")
ck = os.path.join(RES, f"{mkt}_comp_ampliado_registros.csv")

if tramo == "resumen":
    b = pd.read_csv(ck)
    res = b.groupby("modelo").agg(ret_medio=("ret","mean"), vol_media=("vol","mean"),
                                  mdd_medio=("mdd","mean"), n=("ret","size"))
    res["sharpe_aprox"] = (res["ret_medio"]*252/63)/res["vol_media"]
    res.to_csv(os.path.join(RES, f"{mkt}_comparadores_ampliado.csv"))
    print(res.round(4).to_string())
    # DM: perfiles representativos vs benchmarks, sobre retornos emparejados por t
    piv = b.pivot(index="t", columns="modelo", values="ret").dropna()
    pares = [("OWA-Guardian","1/N"), ("OWA-Guardian","MinVar"),
             ("OWA-Pragmatist","1/N"), ("OWA-Visionary","MaxSharpe"),
             ("OWA-Pragmatist","MLP"), ("OWA-Pragmatist","ANFIS"),
             ("OWA-Visionary","MLP"), ("OWA-Visionary","ANFIS")]
    rows = []
    for a_, b_ in pares:
        dm, p = diebold_mariano(piv[a_].values, piv[b_].values)
        rows.append({"A": a_, "B": b_, "DM_HLN": round(dm,3), "p_valor": round(p,4),
                     "n_ventanas": len(piv)})
    dmdf = pd.DataFrame(rows)
    dmdf.to_csv(os.path.join(RES, f"{mkt}_diebold_mariano.csv"), index=False)
    print(dmdf.to_string(index=False))
    sys.exit(0)

px = pd.read_csv(os.path.join(HERE, "..", "data", f"{mkt}_precios.csv"),
                 parse_dates=["Date"], index_col="Date")
cfg = EngineConfig(); eng = RecommendationEngine(px, cfg)
t_grid = list(range(cfg.lookback, len(px) - cfg.horizon, cfg.horizon))
tr, ve, va = split_70_20_10(len(t_grid))
grid = t_grid[ve] if tramo == "ve" else t_grid[va]
rows = []
for t in grid:
    rw = px.iloc[t-cfg.lookback:t].pct_change().dropna()
    carteras = {"1/N": equal_weight(rw), "MinVar": min_variance(rw, cfg.max_weight),
                "MaxSharpe": max_sharpe(rw, cfg.max_weight),
                "MLP": mlp_portfolio(px, t, cfg.lookback, cfg.horizon, cfg.top_n, cfg.max_weight),
                "ANFIS": anfis_portfolio(px, t, cfg.lookback, cfg.horizon, cfg.top_n, cfg.max_weight)}
    for p_ in eng.profiles:
        carteras[f"OWA-{p_.name}"] = eng.builder.build(p_, t).weights
    for name, w in carteras.items():
        rows.append({"t": t, "modelo": name, **_perf(px, w, t, cfg.horizon, cfg.tc_bps)})
df = pd.DataFrame(rows)
if os.path.exists(ck) and tramo != "ve":
    df = pd.concat([pd.read_csv(ck), df]).drop_duplicates(["t","modelo"])
df.to_csv(ck, index=False)
print(f"{mkt} {tramo}: {len(grid)} ventanas, registros acumulados: {len(df)}")
