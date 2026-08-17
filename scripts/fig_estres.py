"""Figura 7.1 -- Behavioral coherence under market stress (trayectoria).

Genera figures/F30_estres.png a partir de
results/{us,co}_estres_trayectoria.csv (correr antes run_oe4.py para ambos
mercados, que ahora persiste la trayectoria dia a dia -- ver pipeline.py) y
de results/{us,co}_estres.csv (para la cifra de coherencia del titulo).

Curvas de valor acumulado (base 1.0), dia a dia, de las ocho carteras de
perfil Y de los cinco comparadores del anteproyecto (1/N, minima varianza,
maximo Sharpe, MLP, ANFIS), encadenadas SOLO en el peor subperiodo de cada
mercado (2022 en EE. UU., COVID-19 en Colombia). Reemplaza la version en
barras (que solo promediaba la volatilidad realizada por perfil) por la
trayectoria completa, agregando los comparadores para la comparacion
directa perfil-vs-modelo econometrico (a pedido del autor, 2026-08-17).
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

BENCH_STYLE = {"1/N": ("#666666", (4, 1.5)), "MinVar": ("#333333", (1, 1)),
              "MaxSharpe": ("#999999", (6, 2, 1, 2)),
              "MLP": ("#444444", (2, 1)), "ANFIS": ("#777777", (1, 1, 4, 1))}
TITLES = {"us": "U.S. — 2022 bear market", "co": "Colombia — COVID-19 crash"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, mkt in zip(axes, ["us", "co"]):
    traj = pd.read_csv(os.path.join(RES, f"{mkt}_estres_trayectoria.csv"),
                       index_col=0, parse_dates=True)
    coh = pd.read_csv(os.path.join(RES, f"{mkt}_estres.csv")).iloc[0]

    for i, n in enumerate(PROFILE_NAMES):
        s = traj[f"OWA-{n}"]
        ax.plot(s.index, (s.values - 1) * 100, color=OKABE_ITO[i],
               linewidth=1.8, zorder=3, label=f"OWA-{n}" if mkt == "us" else None)
    for name, (color, dashes) in BENCH_STYLE.items():
        s = traj[name]
        ax.plot(s.index, (s.values - 1) * 100, color=color, linewidth=1.3,
               dashes=dashes, zorder=2, label=name if mkt == "us" else None)
    ax.axhline(0, color="black", linewidth=0.6, zorder=1)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title(f"{TITLES[mkt]}\nSpearman(orness, vol) = {coh['stress_coherence_vol']:+.3f}",
                fontsize=10.5)
    style(ax)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=7.8,
          frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Behavioral coherence under market stress: profile ladder vs. "
            "econometric models, day by day", fontsize=13)
fig.tight_layout(rect=[0, 0.05, 1, 1])
out = os.path.join(FIG, "F30_estres.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print("OK:", out)
