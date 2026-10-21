"""Fig. 3 of the article on regime-induced orness and declared profile updating (A4).

Parameter-recovery study (scripts/run_emocional.py): (a) mean absolute emotional gap per
investor in the logical and emotional populations; (b) individual estimates of the
loss-aversion asymmetry in the logical population, range [0, 6], generating value 2.25.
Navy-and-grey palette, white background, no internal titles.

Run from the repository root:  python figures_article_a4/make_fig3.py
Input: results/emocional_individual.csv.  Output: figures_article_a4/Fig3 (.png, .tif, .eps).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'figures_article_a4'
for f in fm.findSystemFonts():
    if 'LiberationSans' in f:
        fm.fontManager.addfont(f)
FONT = 'Liberation Sans' if any(x.name == 'Liberation Sans' for x in fm.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams.update({'font.family': FONT, 'font.size': 8, 'axes.edgecolor': '#9E9E9E',
                     'axes.labelcolor': '#222222', 'xtick.color': '#444444', 'ytick.color': '#444444',
                     'savefig.facecolor': 'white', 'figure.facecolor': 'white',
                     'ps.fonttype': 42, 'pdf.fonttype': 42})
MM = 1 / 25.4
NAVY, GREY_D, GREY_M, GRID = '#1F3A5F', '#3F3F3F', '#8C8C8C', '#D9D9D9'
D = pd.read_csv(ROOT / 'results' / 'emocional_individual.csv')
rng = np.random.default_rng(0)  # jitter only


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.set_axisbelow(True)


fig, (a, b) = plt.subplots(1, 2, figsize=(174 * MM, 66 * MM), gridspec_kw={'wspace': 0.35})
for i, (g, lab, col, mk) in enumerate([('logico', 'Logical', GREY_M, 's'), ('emocional', 'Emotional', NAVY, 'o')]):
    y = D[D.grupo == g].mean_abs_gap.values
    a.boxplot(y, positions=[i], widths=0.45, showfliers=False,
              medianprops={'color': '#111111'}, boxprops={'color': GREY_D}, whiskerprops={'color': GREY_D},
              capprops={'color': GREY_D})
    a.scatter(i + rng.uniform(-0.12, 0.12, len(y)), y, s=10, color=col, marker=mk, zorder=3)
a.set_xticks([0, 1], ['Logical', 'Emotional'])
a.set_ylabel('Mean absolute emotional gap |ε|')
a.text(-0.18, 1.02, '(a)', transform=a.transAxes, fontsize=8)
style(a)
lam = D[D.grupo == 'logico'].lambda_hat
shown = lam[(lam >= 0) & (lam <= 6)]
b.scatter(rng.uniform(-0.15, 0.15, len(shown)), shown, s=12, color=GREY_D, marker='s', zorder=3)
b.axhline(2.25, color=NAVY, lw=1.0, ls='--')
b.axhline(float(lam.median()), color=GREY_M, lw=1.0, ls=':')
b.text(0.93, 2.25, 'generating value 2.25', color=NAVY, fontsize=6.5, va='bottom', ha='right')
b.text(0.93, float(lam.median()), f'median {lam.median():.2f}', color=GREY_D, fontsize=6.5, va='top', ha='right')
b.set_xlim(-0.4, 0.95); b.set_xticks([0], ['Logical population'])
b.set_ylim(0, 6)
b.set_ylabel('Estimated loss-aversion asymmetry')
b.text(-0.18, 1.02, '(b)', transform=b.transAxes, fontsize=8)
style(b)
for ext, kw in (('png', {'dpi': 600}), ('tif', {'dpi': 600, 'pil_kwargs': {'compression': 'tiff_lzw'}}), ('eps', {})):
    fig.savefig(OUT / f'Fig3.{ext}', bbox_inches='tight', **kw)
print('Fig3 written to', OUT, '| shown', len(shown), 'of', len(lam))
