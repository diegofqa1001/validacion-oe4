"""Figura 7.3 -- OE4-E: la re-elicitacion declarada valida el canal emocional.

Genera figures/F32_emocional.png a partir de results/emocional_individual.csv
y results/emocional_resumen.csv (correr antes scripts/run_emocional.py).

Panel izquierdo: distribucion de la brecha emocional media |eps| por
inversor, poblacion logica (control) vs. emocional. Panel derecho:
lambda_hat (aversion a la perdida estimada de las declaraciones)
recuperado SOLO en la poblacion logica, con la mediana recuperada y el
valor sembrado (loss_lambda de EngineConfig) superpuestos. Se excluyen del
histograma (no de la mediana) los atipicos con lambda_hat > 6 -- pendientes
casi nulas en ganancias que inflan el cociente sin aportar señal.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "motor-owa-v2", "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from motor_owa.viz import style
from motor_owa.config import EngineConfig

RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

ind = pd.read_csv(os.path.join(RES, "emocional_individual.csv"))
resumen = pd.read_csv(os.path.join(RES, "emocional_resumen.csv"), index_col=0)
cfg = EngineConfig()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
for grupo, color in [("logico", "#0072B2"), ("emocional", "#E69F00")]:
    vals = ind.loc[ind["grupo"] == grupo, "mean_abs_gap"]
    ax.hist(vals, bins=14, color=color, edgecolor="black", linewidth=0.5,
           alpha=0.85, label=f"{grupo} (n={len(vals)})")
ax.set_xlabel("Mean |emotional gap| |ε| per investor")
ax.set_ylabel("Investors")
ax.set_title("Detecting the emotional component", fontsize=11)
ax.legend(fontsize=9)
style(ax)

ax = axes[1]
lam_logico = ind.loc[ind["grupo"] == "logico", "lambda_hat"].dropna()
# Rango fijo [0,6]: los pocos lambda_hat negativos (pendiente casi nula o de
# signo invertido en ganancias, ruido del decisor sintetico) y los positivos
# > 6 quedan fuera del histograma por construccion del rango, no por un
# filtro adicional; el titulo solo cuantifica los > 6 porque son los que
# distorsionarian la escala si se incluyeran.
n_out = int((lam_logico > 6).sum())
ax.hist(lam_logico, bins=10, range=(0, 6), color="#009E73",
       edgecolor="black", linewidth=0.5)
ax.axvline(cfg.loss_lambda, color="#D55E00", linestyle="--", linewidth=2,
          label=f"seeded λ = {cfg.loss_lambda:.2f}")
ax.axvline(float(resumen.loc["logico", "lambda_estimado"]), color="black",
          linestyle=":", linewidth=2,
          label=f"recovered median = {resumen.loc['logico', 'lambda_estimado']:.2f}")
ax.set_xlabel("λ̂ estimated from declarations (logical group)")
ax.set_ylabel("Investors")
ax.set_title(f"Loss aversion becomes estimable  ({n_out} outlier > 6 not shown)",
            fontsize=11)
ax.legend(fontsize=9)
style(ax)

fig.suptitle("OE4-E experiment: declared re-elicitation validates the emotional channel",
            fontsize=12.5)
fig.tight_layout()
out = os.path.join(FIG, "F32_emocional.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print("OK:", out)
