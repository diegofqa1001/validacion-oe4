"""Figura comparativa de todos los modelos (F29): escalera OWA vs benchmarks.

Genera figures/F29_model_comparison.png a partir de
results/{us,co}_comparadores_ampliado.csv (correr antes
run_comparadores_ampliado.py para ambos mercados y tramos).
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

BENCH_STYLE = {"1/N": ("s", "#666666"), "MinVar": ("D", "#333333"),
               "MaxSharpe": ("^", "#999999"), "MLP": ("v", "#444444"),
               "ANFIS": ("P", "#777777")}
BENCH_LABELS = {"1/N": "1/N", "MinVar": "Min-variance", "MaxSharpe": "Max-Sharpe",
                "MLP": "Neural network (MLP)", "ANFIS": "ANFIS"}
OFFSETS = {
 "us": {"1/N": (7, 4), "MinVar": (7, -12), "MaxSharpe": (8, -2), "MLP": (-108, 8), "ANFIS": (8, 2)},
 "co": {"1/N": (7, 4), "MinVar": (8, 4), "MaxSharpe": (-78, 10), "MLP": (8, -3), "ANFIS": (8, 2)},
}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
for ax, mkt, title, nwin in zip(axes, ["us", "co"],
                                ["United States (2015–2026)", "Colombia (2015–2026)"],
                                [13, 14]):
    df = pd.read_csv(os.path.join(RES, f"{mkt}_comparadores_ampliado.csv"), index_col=0)
    owa = df.loc[[f"OWA-{n}" for n in PROFILE_NAMES]]
    ax.plot(owa["vol_media"]*100, owa["ret_medio"]*100, "-", color="#BBBBBB",
            linewidth=1.4, zorder=1)
    for i, n in enumerate(PROFILE_NAMES):
        r = df.loc[f"OWA-{n}"]
        ax.scatter(r["vol_media"]*100, r["ret_medio"]*100, s=120,
                   color=OKABE_ITO[i], edgecolor="black", linewidth=0.7,
                   zorder=3, label=f"OWA-{n}" if mkt == "us" else None)
    for b, (m, c) in BENCH_STYLE.items():
        r = df.loc[b]
        ax.scatter(r["vol_media"]*100, r["ret_medio"]*100, s=130, marker=m,
                   color=c, edgecolor="black", linewidth=0.7, zorder=4,
                   label=BENCH_LABELS[b] if mkt == "us" else None)
        ax.annotate(BENCH_LABELS[b], (r["vol_media"]*100, r["ret_medio"]*100),
                    textcoords="offset points", xytext=OFFSETS[mkt][b],
                    fontsize=7.5, color="#333333")
    for n, dx, dy in [("Guardian", 6, -13), ("Visionary", -20, 10)]:
        r = df.loc[f"OWA-{n}"]
        ax.annotate(n, (r["vol_media"]*100, r["ret_medio"]*100),
                    textcoords="offset points", xytext=(dx, dy), fontsize=8,
                    fontweight="bold", color=OKABE_ITO[PROFILE_NAMES.index(n)])
    ax.set_xlabel("Mean realized volatility (% ann.)")
    ax.set_ylabel("Mean quarterly return (%)")
    ax.set_title(f"{title} — {nwin} out-of-sample windows", fontsize=11)
    style(ax)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=7.6,
           frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("All models compared: the OWA engine spans an ordered risk ladder; "
             "each benchmark offers a single point", fontsize=12.5)
fig.tight_layout(rect=[0, 0.03, 1, 1])
out = os.path.join(FIG, "F29_model_comparison.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print("OK:", out)
