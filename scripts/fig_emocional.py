"""Figura 7.3 -- Experimento OE4-E (componente emocional).

Lee results/emocional_individual.csv y results/emocional_resumen.csv
(run_emocional.py). Panel (a): brecha emocional media |eps| por inversor.
Panel (b): lambda_hat de la poblacion logica en el rango [0, 6]; el titulo
cuenta TODAS las estimaciones fuera de ese rango (negativas y > 6), que
siguen incluidas en la mediana.
Salida: figures/F32_emocional.png (rotulos en espanol, coma decimal).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pandas as pd
import matplotlib.pyplot as plt
from _es import coma, ejes_coma, estilo, guardar
from motor_owa.config import EngineConfig

RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
ind = pd.read_csv(os.path.join(RES, "emocional_individual.csv"))
res = pd.read_csv(os.path.join(RES, "emocional_resumen.csv"), index_col=0)
cfg = EngineConfig()
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax = axes[0]
for g, c, et in [("logico", "#0072B2", "lógicos"), ("emocional", "#E69F00", "emocionales")]:
    v = ind.loc[ind.grupo == g, "mean_abs_gap"]
    ax.hist(v, bins=14, color=c, edgecolor="black", lw=0.5, alpha=0.85,
            label=f"{et} (n = {len(v)})")
ax.set_xlabel("Brecha emocional media |ε| por inversor")
ax.set_ylabel("Inversores")
ax.set_title("(a) Detección del componente emocional", fontsize=10.5)
ax.legend(fontsize=9, frameon=False)
estilo(ax); ejes_coma(ax)
ax = axes[1]
lam = ind.loc[ind.grupo == "logico", "lambda_hat"].dropna()
n_neg, n_alto = int((lam < 0).sum()), int((lam > 6).sum())
ax.hist(lam, bins=12, range=(0, 6), color="#009E73", edgecolor="black", lw=0.5)
ax.axvline(cfg.loss_lambda, color="#D55E00", ls="--", lw=2,
           label=f"λ sembrado = {coma(cfg.loss_lambda, 2)}")
med = float(res.loc["logico", "lambda_estimado"])
ax.axvline(med, color="black", ls=":", lw=2, label=f"mediana estimada = {coma(med, 2)}")
ax.set_xlabel("λ̂ estimado de las declaraciones (población lógica)")
ax.set_ylabel("Inversores")
ax.set_title(f"(b) Recuperación de λ; fuera del rango [0, 6]: {n_neg} negativas "
             f"y {n_alto} mayores que 6", fontsize=10)
ax.legend(fontsize=9, frameon=False)
estilo(ax); ejes_coma(ax)
fig.tight_layout()
out = os.path.join(FIG, "F32_emocional.png")
guardar(fig, out)
print("OK:", out)
