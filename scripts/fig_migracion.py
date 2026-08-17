"""Figura 7.2 -- Profile migration (octiles) by standardized surprise.

Genera figures/F31_migracion.png a partir de results/sensibilidad_perfil.csv.
La matriz es independiente del mercado (depende solo de kappa y
loss_lambda, no de los precios -- ver profile_sensitivity_matrix en
src/oe4/stability.py), por eso el repositorio conserva un unico archivo
en vez de una copia por mercado.
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

sens = pd.read_csv(os.path.join(RES, "sensibilidad_perfil.csv"), index_col=0)

cfg = EngineConfig()
fig, ax = plt.subplots(figsize=(9, 6))
data = sens.values.astype(float)
im = ax.imshow(data, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f"{int(data[i, j])}", ha="center", va="center", fontsize=10)
ax.set_xticks(range(len(sens.columns)))
ax.set_xticklabels(sens.columns)
ax.set_yticks(range(len(sens.index)))
ax.set_yticklabels(sens.index)
ax.set_xlabel("Standardized surprise s")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("octile shift")
ax.set_title(f"Profile migration (octiles) by surprise — "
             f"loss-aversion asymmetry $\\lambda$ = {cfg.loss_lambda:.2f}",
             fontsize=12.5)
fig.tight_layout()
out = os.path.join(FIG, "F31_migracion.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print("OK:", out)
