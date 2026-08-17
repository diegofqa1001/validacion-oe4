"""Figura 7.1 -- Behavioral coherence under market stress.

Genera figures/F30_estres.png a partir de results/{us,co}_estres.csv
(correr antes run_oe4.py para ambos mercados, que ahora persiste el
detalle completo del subperiodo de estres -- ver pipeline.py,
correccion de reproducibilidad 2026-08-17).

Reproduce la Figura 7.1 de la tesis: volatilidad realizada media de las
ocho carteras de perfil, recomputada SOLO en el peor subperiodo de cada
mercado (ventana de maxima caida del indice 1/N), con la correlacion de
Spearman(orness, vol) de esa ventana en el titulo de cada panel.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "motor-owa-v2", "src"))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from motor_owa.viz import OKABE_ITO, style
from motor_owa.config import PROFILE_NAMES

RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

TITLES = {"us": "U.S. — 2022 bear market", "co": "Colombia — COVID-19 crash"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, mkt in zip(axes, ["us", "co"]):
    row = pd.read_csv(os.path.join(RES, f"{mkt}_estres.csv")).iloc[0]
    vols = [row[f"vol_{n}"] for n in PROFILE_NAMES]
    ax.bar(PROFILE_NAMES, vols, color=OKABE_ITO, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Realized volatility (ann.)")
    ax.set_title(f"{TITLES[mkt]}\nSpearman(orness, vol) = {row['stress_coherence_vol']:+.3f}",
                fontsize=10.5)
    style(ax)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.suptitle("Behavioral coherence under market stress", fontsize=13)
fig.tight_layout()
out = os.path.join(FIG, "F30_estres.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print("OK:", out)
