"""Figures 2-4 of the ESWA article (Fig. 1, the architecture diagram, is drawn separately).

Navy-and-grey palette, white background, sans-serif font, no internal titles, width <= 174 mm.
Run from the repository root after article_eswa/window_coherence.py:
    python article_eswa/make_figures.py
Inputs: results/{us,co}_registros_motor.csv, results/{us,co}_estres.csv,
        results/{us,co}_comparadores.csv, article_eswa/output/window_coherence.csv
Outputs: article_eswa/figures/Fig2..Fig4 as .png, .tif (600 dpi) and .eps.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

for f in fm.findSystemFonts():
    if "LiberationSans" in f:
        fm.fontManager.addfont(f)
FONT = "Liberation Sans" if any(x.name == "Liberation Sans" for x in fm.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": FONT, "font.size": 8, "axes.edgecolor": "#9E9E9E",
                     "axes.labelcolor": "#222222", "xtick.color": "#444444", "ytick.color": "#444444",
                     "savefig.facecolor": "white", "figure.facecolor": "white",
                     "ps.fonttype": 42, "pdf.fonttype": 42})
MM = 1 / 25.4
NAVY, GREY_D, GREY_M, GREY_L, GRID = "#1F3A5F", "#3F3F3F", "#8C8C8C", "#BDBDBD", "#D9D9D9"
PROFILES = ["Guardian", "Sentinel", "Pragmatist", "Analyst", "Strategist", "Adventurer", "Innovator", "Visionary"]
MKT = {"US": "United States", "CO": "Colombia"}


def save(fig, name):
    for ext, kw in (("png", {"dpi": 600}), ("tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
                    ("eps", {})):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ---------- Fig. 2: window-level rank correlation at three levels ----------
win = pd.read_csv(HERE / "output" / "window_coherence.csv")
levels = [("rho_target", "Target\n(design)"), ("rho_attained", "Attained\n(ex ante)"),
          ("rho_realised", "Realised\n(ex post)")]
colors = [GREY_L, GREY_M, NAVY]
fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 68 * MM), sharey=True)
rng = np.random.default_rng(7)
for ax, mkt in zip(axes, ["US", "CO"]):
    g = win[win.market == mkt]
    data = [g[c].values for c, _ in levels]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops={"color": "#111111", "lw": 1.0},
                    whiskerprops={"color": GREY_D, "lw": 0.6}, capprops={"color": GREY_D, "lw": 0.6})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(GREY_D); patch.set_linewidth(0.6)
    for i, (d, c) in enumerate(zip(data, colors), start=1):
        x = i + rng.uniform(-0.16, 0.16, size=len(d))
        ax.scatter(x, d, s=6, color=c if c != GREY_L else GREY_M, edgecolor="none", alpha=0.9, zorder=3)
        ax.text(i, -1.12, f"mean {d.mean():.2f}\nperfect {np.mean(d > 0.9999):.0%}",
                ha="center", va="bottom", fontsize=6.3, color=GREY_D)
    ax.axhline(0, color=GREY_M, lw=0.6, ls=":")
    ax.set_xticks([1, 2, 3], [l for _, l in levels])
    ax.set_ylim(-1.15, 1.08)
    ax.set_xlabel(f"{MKT[mkt]} ({len(g)} quarterly windows)")
    style(ax)
axes[0].set_ylabel("Spearman correlation between orness and volatility")
fig.tight_layout()
save(fig, "Fig2")

# ---------- Fig. 3: realised volatility ladders, full grid vs stress window ----------
fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 66 * MM), sharey=False)
x = np.arange(8)
for ax, mkt in zip(axes, ["us", "co"]):
    d = pd.read_csv(RES / f"{mkt}_registros_motor.csv")
    q = d.groupby("profile")["vol"].quantile([.25, .5, .75]).unstack().reindex(PROFILES)
    m = d.groupby("profile")["vol"].mean().reindex(PROFILES)
    s = pd.read_csv(RES / f"{mkt}_estres.csv").iloc[0]
    sv = np.array([s[f"vol_{p}"] for p in PROFILES])
    ax.fill_between(x, q[.25] * 100, q[.75] * 100, color=GREY_L, alpha=0.45, lw=0,
                    label="Full grid, interquartile range across windows")
    ax.plot(x, m * 100, color=GREY_D, lw=1.0, marker="o", ms=3, label="Full grid, mean")
    ax.plot(x, sv * 100, color=NAVY, lw=1.0, ls="--", marker="s", ms=3,
            label=f"Stress window ({s['fecha_inicio'][:7]} to {s['fecha_fin'][:7]})")
    ax.set_xticks(x, PROFILES, rotation=40, ha="right")
    ax.set_ylabel("Annualised realised volatility (%)")
    ax.set_xlabel(MKT[mkt.upper()])
    style(ax)
h, l = axes[0].get_legend_handles_labels()
h2, l2 = axes[1].get_legend_handles_labels()
fig.legend(h, l[:2] + ["Stress window (United States: 2022-01 to 2023-01; "
                         "Colombia: 2019-02 to 2020-03)"], frameon=False, fontsize=6.5,
           loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "Fig3")

# ---------- Fig. 4: risk-return plane, profile portfolios vs comparators ----------
fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 70 * MM))
for ax, mkt in zip(axes, ["us", "co"]):
    c = pd.read_csv(RES / f"{mkt}_comparadores.csv")
    own = c[c.modelo.str.startswith("OWA-")].copy()
    own["p"] = own.modelo.str.replace("OWA-", "")
    own = own.set_index("p").reindex(PROFILES)
    oth = c[~c.modelo.str.startswith("OWA-")]
    ax.plot(own.vol_media * 100, own.ret_medio * 100, color=NAVY, lw=0.8, marker="o", ms=3.5, zorder=3,
            label="Engine, eight profiles")
    for name, r in own.iloc[[0, -1]].iterrows():
        ax.annotate(name, (r.vol_media * 100, r.ret_medio * 100), xytext=(4, -3),
                    textcoords="offset points", fontsize=6.3, color=NAVY)
    ax.scatter(oth.vol_media * 100, oth.ret_medio * 100, marker="^", s=22, color=GREY_D, zorder=3,
               label="Single-portfolio comparators")
    lab = {"1/N": "1/N", "ANFIS": "ANFIS", "MLP": "MLP", "MaxSharpe": "Max. Sharpe", "MinVar": "Min. variance"}
    off = {"us": {"MLP": (-14, 4), "MaxSharpe": (4, -9)}, "co": {"MinVar": (-10, 6), "MLP": (6, -2)}}[mkt]
    for _, r in oth.iterrows():
        ax.annotate(lab[r.modelo], (r.vol_media * 100, r.ret_medio * 100), xytext=off.get(r.modelo, (4, 2)),
                    textcoords="offset points", fontsize=6.3, color=GREY_D)
    ax.set_xlabel(f"Mean annualised volatility (%), {MKT[mkt.upper()]}")
    ax.set_ylabel("Mean return per quarterly horizon (%)")
    style(ax)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=6.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, "Fig4")
print("Figures written to", OUT)
