"""Corrección de Holm sobre los contrastes de Diebold-Mariano (HLN) de la Tabla 7.4.

Lee results/{us,co}_diebold_mariano.csv y aplica el procedimiento de
Holm-Bonferroni (i) a la familia de los 16 contrastes (8 por mercado) y
(ii) a cada mercado por separado (familia de 8). Escribe
results/dm_holm.csv con los valores-p ajustados y la decisión al 5 %.
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def holm(p):
    p = pd.Series(p).reset_index(drop=True)
    m = len(p)
    order = p.sort_values().index
    adj = pd.Series(index=p.index, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj


frames = []
for mk in ("us", "co"):
    d = pd.read_csv(os.path.join(RES, f"{mk}_diebold_mariano.csv"))
    d.insert(0, "mercado", mk)
    d["p_holm_mercado"] = holm(d["p_valor"]).values
    frames.append(d)
out = pd.concat(frames, ignore_index=True)
out["p_holm_16"] = holm(out["p_valor"]).values
out["sig_5_sin_ajuste"] = out["p_valor"] < 0.05
out["sig_5_holm_16"] = out["p_holm_16"] < 0.05
out["sig_5_holm_mercado"] = out["p_holm_mercado"] < 0.05
out.to_csv(os.path.join(RES, "dm_holm.csv"), index=False)
print(out.to_string(index=False))
