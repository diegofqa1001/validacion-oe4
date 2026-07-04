"""Analisis de estabilidad predictiva (anteproyecto, Obj. 4).

Tres familias declaradas:
1. PERTURBACIONES: ruido gaussiano sobre la matriz de criterios; se mide
   cuanta consistencia ordinal del ranking sobrevive a cada nivel de ruido.
2. ESTRES: se re-ejecuta la evaluacion restringida al peor subperiodo del
   mercado (ventana de maxima caida del indice equiponderado).
3. SENSIBILIDAD AL CAMBIO DE PERFIL: matriz de migraciones del modulo
   adaptativo del motor (cuantos octiles se desplaza el inversor ante
   sorpresas de distinta magnitud).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

__all__ = ["ranking_noise_stability", "worst_window", "stress_coherence",
           "profile_sensitivity_matrix"]


def ranking_noise_stability(prices: pd.DataFrame, t: int, profile,
                            lookback: int = 126,
                            noise_levels=(0.01, 0.05, 0.10),
                            n_rep: int = 50, seed: int = 20260704) -> pd.DataFrame:
    """Consistencia ordinal del ranking OWA bajo ruido en los criterios."""
    from motor_owa.criteria import compute_criteria, CRITERIA
    from motor_owa.owa import owa
    from motor_owa.validation import ordinal_consistency
    rng = np.random.default_rng(seed)
    crit = compute_criteria(prices, t, lookback)
    w = profile.weights(len(CRITERIA))
    base = crit.apply(lambda row: owa(row.values, w), axis=1)
    base_rank = base.rank().values
    rows = []
    for s in noise_levels:
        vals = []
        for _ in range(n_rep):
            noisy = np.clip(crit.values + rng.normal(0, s, crit.shape), 0, 1)
            sc = np.array([owa(r, w) for r in noisy])
            vals.append(ordinal_consistency(base_rank,
                                            pd.Series(sc).rank().values))
        rows.append({"noise_sd": s, "mean_consistency": float(np.mean(vals)),
                     "p05": float(np.quantile(vals, 0.05)),
                     "p95": float(np.quantile(vals, 0.95))})
    return pd.DataFrame(rows)


def worst_window(prices: pd.DataFrame, width: int = 252) -> tuple:
    """(inicio, fin) de la ventana de maxima caida del indice 1/N."""
    idx = prices.mean(axis=1)
    ret = idx / idx.shift(width) - 1.0
    end = int(np.nanargmin(ret.values))
    return max(0, end - width), end


def stress_coherence(engine, width: int = 252) -> Dict[str, object]:
    """Coherencia orness-vol re-computada SOLO en el peor subperiodo."""
    from motor_owa.validation import coherence_spearman
    a, b = worst_window(engine.prices, width)
    lo = max(engine.cfg.lookback, a)
    grid = list(range(lo, max(lo + 1, b - engine.cfg.horizon),
                      engine.cfg.horizon))
    if not grid:
        return {"stress_coherence_vol": float("nan")}
    _ANNUAL = 252
    vols = {p.name: [] for p in engine.profiles}
    for t in grid:
        for p in engine.profiles:
            port = engine.builder.build(p, t)
            seg = engine.prices[port.weights.index].iloc[
                t:t + engine.cfg.horizon].pct_change().dropna()
            vols[p.name].append(float((seg @ port.weights.values).std()
                                      * np.sqrt(_ANNUAL)))
    alphas = [p.alpha for p in engine.profiles]
    mean_v = [float(np.mean(vols[p.name])) for p in engine.profiles]
    return {"stress_coherence_vol": coherence_spearman(alphas, mean_v),
            "stress_start": int(a), "stress_end": int(b),
            "stress_mean_vols": dict(zip([p.name for p in engine.profiles],
                                         mean_v))}


def profile_sensitivity_matrix(surprises=(-3, -2, -1, 0, 1, 2, 3),
                               kappa: float = 0.25,
                               loss_lambda: float = 2.25) -> pd.DataFrame:
    """Migracion esperada (en octiles) por magnitud de sorpresa y perfil."""
    from motor_owa.adaptive import update_latent
    from motor_owa.latent import classify_z, octile_z
    from motor_owa.config import PROFILE_NAMES
    rows = []
    for k in range(1, 9):
        z0 = octile_z(k)
        row = {"perfil": PROFILE_NAMES[k - 1]}
        for s in surprises:
            z1 = update_latent(z0, float(s), kappa, loss_lambda)
            row[f"s={s:+d}"] = classify_z(z1) - k
        rows.append(row)
    return pd.DataFrame(rows).set_index("perfil")
