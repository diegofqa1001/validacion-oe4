"""Diagnostico del MAPE entre particiones (Tabla 7.2 / Tabla 7.5 de la tesis).

El MAPE de motor_owa.validation divide |s_validacion - s_verificacion| por
|s_validacion|, donde s es el puntaje OWA medio de cada activo en la
particion. Este guion reconstruye, con el mismo protocolo de
RecommendationEngine.panel_backtest (rejilla, particion 70-20-10 y
puntajes de port.scores), los puntajes medios de verificacion y validacion
por perfil y mercado, y resume el denominador: minimo, numero de activos con
puntaje de validacion < 0,10 y la contribucion al MAPE del activo con menor
puntaje. Comprueba ademas que el RMSE, el MAE y el MAPE recalculados
coinciden con results/{us,co}_metricas_motor.csv.

Uso: python scripts/diag_mape_denominador.py
Salida: results/diag_mape_denominador.csv
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "motor-owa-v2", "src"))

import numpy as np
import pandas as pd

from motor_owa.config import EngineConfig
from motor_owa.data import load_csv
from motor_owa.engine import RecommendationEngine
from motor_owa.validation import mae, mape, rmse, split_70_20_10

SNAP = os.path.join(HERE, "..", "data", "snapshot_oe4")
RES = os.path.join(HERE, "..", "results")

rows = []
for mkt in ("us", "co"):
    px = load_csv(os.path.join(SNAP, f"{mkt}_precios.csv"))
    cfg = EngineConfig()
    eng = RecommendationEngine(px, cfg)
    t_grid = list(range(cfg.lookback, len(px) - cfg.horizon, cfg.horizon))
    _, ve, va = split_70_20_10(len(t_grid))
    ref = pd.read_csv(os.path.join(RES, f"{mkt}_metricas_motor.csv"), index_col=0)
    for p in eng.profiles:
        sv_l = [eng.builder.build(p, t).scores.reindex(px.columns) for t in t_grid[ve]]
        sw_l = [eng.builder.build(p, t).scores.reindex(px.columns) for t in t_grid[va]]
        s_ve = pd.concat(sv_l, axis=1).mean(axis=1)
        s_va = pd.concat(sw_l, axis=1).mean(axis=1)
        common = s_ve.dropna().index.intersection(s_va.dropna().index)
        sv, sw = s_ve[common].values, s_va[common].values
        ape = np.abs(sw - sv) / np.maximum(np.abs(sw), 1e-9) * 100
        i_min = int(np.argmin(sw))
        rows.append({
            "mercado": mkt, "perfil": p.name, "orness": p.alpha,
            "n_activos": len(common),
            "rmse": rmse(sw, sv), "mae": mae(sw, sv), "mape": mape(sw, sv),
            "rmse_publicado": ref.loc[p.name, "rmse"],
            "mae_publicado": ref.loc[p.name, "mae"],
            "mape_publicado": ref.loc[p.name, "mape"],
            "puntaje_validacion_min": float(sw.min()),
            "puntaje_validacion_mediana": float(np.median(sw)),
            "activos_puntaje_validacion_lt_0_10": int((sw < 0.10).sum()),
            "activo_min": str(common[i_min]),
            "ape_activo_min_pct": float(ape[i_min]),
            "mape_sin_activo_min": float(np.delete(ape, i_min).mean()),
        })
        print(mkt, p.name, round(rows[-1]["mape"], 2),
              round(rows[-1]["puntaje_validacion_min"], 4))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, "diag_mape_denominador.csv"), index=False)
print(out.round(4).to_string())
