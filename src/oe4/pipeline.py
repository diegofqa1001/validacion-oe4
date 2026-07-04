"""Pipeline OE4: resultados citables sobre datos reales CO y US.

Para cada mercado:
  1. Backtest del motor v2 (8 perfiles) con particion 70-20-10 y metricas
     del anteproyecto (RMSE, MAE, MAPE, NDCG@k, MRR, consistencia ordinal,
     coherencia orness-vol).
  2. Comparadores en la MISMA rejilla de fechas: 1/N, min-varianza,
     max-Sharpe (media-varianza), red neuronal (MLP) y ANFIS.
  3. Estabilidad: ruido sobre criterios, estres (peor subperiodo),
     sensibilidad al cambio de perfil.
Salidas: CSV en results/ y figuras Okabe-Ito en figures/.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .benchmarks import (anfis_portfolio, equal_weight, max_sharpe,
                         min_variance, mlp_portfolio)
from .stability import (profile_sensitivity_matrix, ranking_noise_stability,
                        stress_coherence)

_ANNUAL = 252


def _perf(prices: pd.DataFrame, w: pd.Series, t: int, h: int,
          tc_bps: float = 10.0) -> Dict[str, float]:
    t2 = min(t + h, len(prices) - 1)
    px = prices[w.index]
    ret = float((px.iloc[t2] / px.iloc[t] - 1.0) @ w.values) - tc_bps / 1e4
    seg = px.iloc[t:t2].pct_change().dropna()
    vol = float((seg @ w.values).std() * np.sqrt(_ANNUAL))
    curve = (1 + seg @ w.values).cumprod()
    mdd = float((curve / curve.cummax() - 1.0).min())
    return {"ret": ret, "vol": vol, "mdd": mdd}


def run_market(prices: pd.DataFrame, market: str, outdir: str,
               config=None, quick: bool = False) -> Dict[str, object]:
    """Ejecuta el protocolo OE4 completo para un mercado."""
    from motor_owa.config import EngineConfig
    from motor_owa.engine import RecommendationEngine
    from motor_owa.validation import split_70_20_10, coherence_spearman

    cfg = config or EngineConfig()
    eng = RecommendationEngine(prices, cfg)
    os.makedirs(outdir, exist_ok=True)

    # ---- 1. motor v2: backtest de panel + metricas anteproyecto ----
    m = eng.panel_backtest()
    pd.DataFrame(m["per_profile"]).T.to_csv(
        os.path.join(outdir, f"{market}_metricas_motor.csv"))
    m["records"].to_csv(os.path.join(outdir, f"{market}_registros_motor.csv"),
                        index=False)

    # ---- 2. comparadores en la rejilla de validacion (ultimo 30%) ----
    step = cfg.horizon
    t_grid = list(range(cfg.lookback, len(prices) - cfg.horizon, step))
    tr, ve, va = split_70_20_10(len(t_grid))
    grid_eval = t_grid[ve][:: (2 if quick else 1)] + t_grid[va]
    bench_rows = []
    for t in grid_eval:
        rets_w = prices.iloc[t - cfg.lookback:t].pct_change().dropna()
        carteras = {
            "1/N": equal_weight(rets_w),
            "MinVar": min_variance(rets_w, cfg.max_weight),
            "MaxSharpe": max_sharpe(rets_w, cfg.max_weight),
            "MLP": mlp_portfolio(prices, t, cfg.lookback, cfg.horizon,
                                 cfg.top_n, cfg.max_weight),
            "ANFIS": anfis_portfolio(prices, t, cfg.lookback, cfg.horizon,
                                     cfg.top_n, cfg.max_weight),
        }
        for p in eng.profiles:
            carteras[f"OWA-{p.name}"] = eng.builder.build(p, t).weights
        for name, w in carteras.items():
            bench_rows.append({"t": t, "modelo": name,
                               **_perf(prices, w, t, cfg.horizon, cfg.tc_bps)})
    bench = pd.DataFrame(bench_rows)
    resumen = bench.groupby("modelo").agg(
        ret_medio=("ret", "mean"), vol_media=("vol", "mean"),
        mdd_medio=("mdd", "mean"), n=("ret", "size"))
    resumen["sharpe_aprox"] = (resumen["ret_medio"] * _ANNUAL / cfg.horizon
                               ) / resumen["vol_media"]
    resumen.to_csv(os.path.join(outdir, f"{market}_comparadores.csv"))

    # ---- 3. estabilidad ----
    t_last = t_grid[-1]
    noise = ranking_noise_stability(prices, t_last, eng.profiles[2],
                                    cfg.lookback,
                                    n_rep=(10 if quick else 50))
    noise.to_csv(os.path.join(outdir, f"{market}_estabilidad_ruido.csv"),
                 index=False)
    stress = stress_coherence(eng)
    sens = profile_sensitivity_matrix(kappa=cfg.kappa,
                                      loss_lambda=cfg.loss_lambda)
    sens.to_csv(os.path.join(outdir, f"{market}_sensibilidad_perfil.csv"))
    pd.DataFrame([{"coherence_vol_total": m["coherence_vol"],
                   "coherence_ret_total": m["coherence_ret"],
                   "stress_coherence_vol": stress["stress_coherence_vol"]}]
                 ).to_csv(os.path.join(outdir, f"{market}_coherencia.csv"),
                          index=False)
    return {"motor": m, "comparadores": resumen, "ruido": noise,
            "estres": stress, "sensibilidad": sens}
