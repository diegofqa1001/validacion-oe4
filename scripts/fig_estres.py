"""Figura 7.1 -- Coherencia conductual bajo estres (trayectorias diarias).

Lee results/{us,co}_estres_trayectoria.csv y results/{us,co}_estres.csv
(run_oe4.py). La ventana de estres es la de maxima caida a 252 dias del
indice equiponderado y se evalua completa (stability.stress_grid).
Salida: figures/F30_estres.png (rotulos en espanol, coma decimal).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from _es import (NOMBRES_ES, COMPARADORES_ES, OKABE_ITO, coma, ejes_coma,
                 estilo, guardar, fecha_es, MESES)
from motor_owa.config import PROFILE_NAMES

RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
BENCH = {"1/N": ("#666666", (4, 1.5)), "MinVar": ("#333333", (1, 1)),
         "MaxSharpe": ("#999999", (6, 2, 1, 2)), "MLP": ("#444444", (2, 1)),
         "ANFIS": ("#777777", (1, 1, 4, 1))}
TIT = {"us": "Estados Unidos", "co": "Colombia"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, mkt in zip(axes, ["us", "co"]):
    traj = pd.read_csv(os.path.join(RES, f"{mkt}_estres_trayectoria.csv"),
                       index_col=0, parse_dates=True)
    coh = pd.read_csv(os.path.join(RES, f"{mkt}_estres.csv")).iloc[0]
    for i, n in enumerate(PROFILE_NAMES):
        s = traj[f"OWA-{n}"]
        ax.plot(s.index, (s.values - 1) * 100, color=OKABE_ITO[i], lw=1.8,
                zorder=3, label=NOMBRES_ES[n] if mkt == "us" else None)
    for b, (c, d) in BENCH.items():
        s = traj[b]
        ax.plot(s.index, (s.values - 1) * 100, color=c, lw=1.3, dashes=d,
                zorder=2, label=COMPARADORES_ES[b] if mkt == "us" else None)
    ax.axhline(0, color="black", lw=0.6, zorder=1)
    ax.set_ylabel("Retorno acumulado (%)")
    ax.set_title(f"{TIT[mkt]}: {fecha_es(traj.index[0])} – "
                 f"{fecha_es(traj.index[-1])}\n"
                 f"Spearman(orness, volatilidad) = "
                 f"{coma(coh['stress_coherence_vol'], 3, signo=True)}",
                 fontsize=10.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: (lambda d: f"{MESES[d.month-1]} {d.year}")(mdates.num2date(x))))
    estilo(ax); ejes_coma(ax, x=False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=7, fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, -0.07))
fig.tight_layout(rect=[0, 0.06, 1, 1])
out = os.path.join(FIG, "F30_estres.png")
guardar(fig, out)
print("OK:", out)
