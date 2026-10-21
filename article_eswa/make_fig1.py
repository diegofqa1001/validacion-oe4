"""Fig. 1 of the ESWA article: architecture of the profile-ordered recommendation engine.

Run: python article_eswa/make_fig1.py  ->  article_eswa/figures/Fig1.{png,tif,eps}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
for f in fm.findSystemFonts():
    if "LiberationSans" in f:
        fm.fontManager.addfont(f)
FONT = "Liberation Sans" if any(x.name == "Liberation Sans" for x in fm.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": FONT, "font.size": 7, "savefig.facecolor": "white",
                     "ps.fonttype": 42, "pdf.fonttype": 42})
MM = 1 / 25.4
NAVY, GREY_D, GREY_L, FILL_N = "#1F3A5F", "#3F3F3F", "#EFEFEF", "#E3E9F1"

fig, ax = plt.subplots(figsize=(174 * MM, 92 * MM))
ax.set_xlim(-2, 176); ax.set_ylim(-1, 92); ax.axis("off")


def box(x, y, w, h, title, body, fc=GREY_L, ec=GREY_D, tc="#111111"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=6.9, weight="bold", color=tc)
    ax.text(x + w / 2, y + h - 7.2, body, ha="center", va="top", fontsize=5.9, color="#222222",
            linespacing=1.35)


def arrow(x1, y1, x2, y2, c=GREY_D, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                 lw=0.8, color=c, linestyle=ls))


# row 1: profile elicitation and the two stages
box(0, 64, 46, 25, "Investor profile",
    "Questionnaire (eight dimensions)\n→ latent score $z$\n→ octile profile $k=1,\\ldots,8$\n"
    "→ orness anchor $\\alpha_k=(2k-1)/16$")
box(51, 64, 61, 25, "Stage 1 · Asset selection",
    "Four criteria per asset (return, low volatility,\nlow drawdown, liquidity), causal window\n"
    "OWA score with RIM weights of orness $\\alpha_k$\nbest-scored assets in each volatility stratum",
    fc=FILL_N, ec=NAVY, tc=NAVY)
box(117, 64, 57, 25, "Stage 2 · Risk control",
    "Common range $[\\sigma_{def},\\sigma_{agg}]$ of the universe\n"
    "target $\\sigma_k=\\sigma_{def}+\\alpha_k(\\sigma_{agg}-\\sigma_{def})$\n"
    "mix $\\theta$ of defensive and aggressive funds\nclosest to $\\sigma_k$; 30% cap per asset",
    fc=FILL_N, ec=NAVY, tc=NAVY)
arrow(46.8, 76.5, 50.2, 76.5)
arrow(112.8, 76.5, 116.2, 76.5)
ax.text(81.5, 61.0, "the profile decides which assets enter", ha="center", fontsize=6.0, style="italic", color=NAVY)
ax.text(145.5, 61.0, "the profile decides how much risk", ha="center", fontsize=6.0, style="italic", color=NAVY)

# row 2
box(0, 32, 46, 22, "Rebalancing",
    "every 63 business days\n126-day estimation window\n10 bp cost per rebalance")
box(51, 32, 61, 22, "Traceability log",
    "questionnaire → $z$ → $k$ → $\\alpha_k$ → scores\n→ selected assets → $\\theta$ → weights\n"
    "stored for audit (suitability)")
box(117, 32, 57, 22, "Recommendation",
    "eight portfolios, one per profile:\nweights, expected return and volatility\n"
    "return projection declared not guaranteed")
arrow(145.5, 59.3, 145.5, 54.8)
arrow(81.5, 59.3, 81.5, 54.8, ls="--")
arrow(116.2, 43, 112.8, 43, ls="--")
arrow(23, 54.8, 23, 63.2, ls="--")
ax.text(24.5, 58.5, "re-run", fontsize=5.8, color=GREY_D, va="center")

# row 3: evaluation layers
ax.add_patch(FancyBboxPatch((0, 1), 174, 22, boxstyle="round,pad=0.4,rounding_size=1.5",
                            fc="white", ec=NAVY, lw=0.9, ls=(0, (4, 2))))
ax.text(87, 21.0, "Suitability evaluated in three layers (this study)", ha="center", va="top",
        fontsize=7.0, weight="bold", color=NAVY)
layers = [("1 · Target", "$\\sigma_k$ strictly increasing in $\\alpha_k$\n(guaranteed by design)"),
          ("2 · Attained ex ante", "volatility of the portfolio built,\nestimated covariance"),
          ("3 · Realised ex post", "volatility over the next\n63-day holding period")]
for i, (t, b) in enumerate(layers):
    x = 6 + i * 58
    ax.text(x + 22, 15.0, t, ha="center", va="top", fontsize=6.6, weight="bold", color="#111111")
    ax.text(x + 22, 10.5, b, ha="center", va="top", fontsize=6.0, color="#222222", linespacing=1.3)
    if i < 2:
        arrow(x + 45, 11.0, x + 54, 11.0, c=NAVY)
arrow(145.5, 31.2, 145.5, 23.8, c=NAVY)

for ext, kw in (("png", {"dpi": 600}), ("tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
                ("eps", {})):
    fig.savefig(OUT / f"Fig1.{ext}", bbox_inches="tight", **kw)
print("Fig1 written")
