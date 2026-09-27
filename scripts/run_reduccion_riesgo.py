"""Reduccion del riesgo maximo del perfil objetivo (criterio del anteproyecto).

Calcula, sobre las MISMAS ventanas de verificacion + validacion de la
Tabla 7.3 (results/{us,co}_comp_ampliado_registros.csv: 13 ventanas en
EE. UU. y 12 en Colombia), la caida maxima media por ventana del perfil
objetivo (OWA-Guardian) frente a cada comparador y al perfil opuesto
(OWA-Visionary), y la reduccion relativa

    reduccion_% = 100 * (1 - MDD_guardian / MDD_comparador)

(positiva: el Guardian cae menos). Reporta ademas la diferencia media
emparejada por ventana y el numero de ventanas en que el Guardian cae menos.

Uso: python scripts/run_reduccion_riesgo.py
Salida: results/reduccion_riesgo_maximo.csv
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
REFS = ["1/N", "MinVar", "MaxSharpe", "MLP", "ANFIS", "OWA-Visionary"]

rows = []
for mkt in ("us", "co"):
    b = pd.read_csv(os.path.join(RES, f"{mkt}_comp_ampliado_registros.csv"))
    mdd = b.pivot(index="t", columns="modelo", values="mdd")
    vol = b.pivot(index="t", columns="modelo", values="vol")
    g_m, g_v = mdd["OWA-Guardian"], vol["OWA-Guardian"]
    for ref in REFS:
        rows.append({
            "mercado": mkt.upper(), "referente": ref, "n_ventanas": len(mdd),
            "mdd_guardian": g_m.mean(), "mdd_referente": mdd[ref].mean(),
            "reduccion_mdd_%": 100 * (1 - g_m.mean() / mdd[ref].mean()),
            "ventanas_guardian_cae_menos": int((g_m > mdd[ref]).sum()),
            "vol_guardian": g_v.mean(), "vol_referente": vol[ref].mean(),
            "reduccion_vol_%": 100 * (1 - g_v.mean() / vol[ref].mean()),
        })
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, "reduccion_riesgo_maximo.csv"), index=False)
print(out.round(4).to_string(index=False))
