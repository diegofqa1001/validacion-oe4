"""Figura 7.2 -- Migracion de perfil (octiles) segun la sorpresa estandarizada.

Lee results/sensibilidad_perfil.csv (independiente del mercado).
Salida: figures/F31_migracion.png (rotulos en espanol, paleta Okabe-Ito).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pandas as pd
import matplotlib.pyplot as plt
from _es import NOMBRES_ES, DIVERGENTE_OI, coma, guardar
from motor_owa.config import EngineConfig

RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
sens = pd.read_csv(os.path.join(RES, "sensibilidad_perfil.csv"), index_col=0)
cfg = EngineConfig()
fig, ax = plt.subplots(figsize=(9, 6))
data = sens.values.astype(float)
im = ax.imshow(data, cmap=DIVERGENTE_OI, vmin=-2, vmax=2, aspect="auto")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v = int(data[i, j])
        ax.text(j, i, f"{v:+d}".replace("-", "−") if v else "0",
                ha="center", va="center", fontsize=10.5)
ax.set_xticks(range(len(sens.columns)))
ax.set_xticklabels([c.replace("s=", "").replace("-", "−") + "σ"
                    if c != "s=+0" else "0" for c in sens.columns])
ax.set_yticks(range(len(sens.index)))
ax.set_yticklabels([NOMBRES_ES[n] for n in sens.index])
ax.set_xlabel("Sorpresa estandarizada del horizonte")
ax.set_ylabel("Perfil de partida")
cb = fig.colorbar(im, ax=ax, ticks=[-2, -1, 0, 1, 2])
cb.ax.set_yticklabels(["−2", "−1", "0", "+1", "+2"])
cb.set_label("Octiles desplazados (− prudencia, + audacia)")
ax.set_title(f"Asimetría de aversión a la pérdida λ = {coma(cfg.loss_lambda, 2)}; "
             f"paso κ = {coma(cfg.kappa, 2)}", fontsize=11)
fig.tight_layout()
out = os.path.join(FIG, "F31_migracion.png")
guardar(fig, out)
print("OK:", out)
