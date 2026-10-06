"""Figura per la tesi: l'incertezza di registrazione contro il carico di amiloide,
sui tre traccianti.

Tre pannelli affiancati, uno per tracciante, stessa scala verticale: cosi' si
vede a colpo d'occhio che il fenomeno non e' una particolarita' dell'FBB ma si
ripete su tre archivi indipendenti. Dentro ogni pannello, i due segmenti
orizzontali sono le mediane dei negativi e dei positivi all'amiloide.

    python -m scripts.figure.figura_stabilita --lingua it    # etichette in italiano
    python -m scripts.figure.figura_stabilita --lingua en    # etichette in inglese

Il capitolo di tesi e' in italiano e i capitoli 1-2 in inglese: la figura si
genera nelle due lingue perche' la scelta resta aperta. Il nome del file segue
la lingua ('stabilita_vs_suvr.pdf' per l'italiano, '..._en.pdf' per l'inglese).
"""
import argparse
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared import percorsi                 # noqa: E402
SRC = percorsi.dati('stabilita_ants')        # prodotti da diag_ants_stabilita.py
# SUVr che separa negativi e positivi all'amiloide in questa figura (linea
# tratteggiata e due mediane per pannello).
SOGLIA = 1.2

_p = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument('--lingua', choices=['it', 'en'], default='it')
_p.add_argument('--out', default=None, help="nome base del file, senza estensione")
ARGS = _p.parse_args()

# Nella figura vanno solo le etichette degli assi e i pannelli. Titolo,
# sottotitolo e commento stanno nella didascalia LaTeX, dove e' il loro posto:
# una figura di tesi non deve raccontarsi da sola.
TESTI = {
 'it': dict(
   y='dev. standard del centiloide (CL)',
   x='SUVr medio del soggetto',
   nomi={'FBB': 'Florbetaben', 'FMM': 'Flutemetamolo', 'FBP': 'Florbetapir'},
   base='stabilita_vs_suvr'),
 'en': dict(
   y='standard deviation of the centiloid (CL)',
   x='mean SUVr of the subject',
   nomi={'FBB': 'Florbetaben', 'FMM': 'Flutemetamol', 'FBP': 'Florbetapir'},
   base='stabilita_vs_suvr_en'),
}[ARGS.lingua]

# palette validata (slot 1-3, light) + forme diverse = codifica secondaria
TRACCIANTI = [('FBB', '#2a78d6', 'o'), ('FMM', '#eb6834', 's'),
              ('FBP', '#1baf7a', '^')]
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#dcdbd7', '#fcfcfb'

for f in ('Arial', 'Helvetica', 'DejaVu Sans'):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = f
        break
plt.rcParams.update({'font.size': 9.5, 'axes.edgecolor': GRID, 'text.color': INK,
                     'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2})

# Un CSV per tracciante con, per soggetto, il SUVr medio e la deviazione
# standard del centiloide fra ripetizioni della sola registrazione ANTs.
dati = {tr: pd.read_csv(SRC / f'stabilita_{tr}_per_soggetto.csv')
        for tr, _, _ in TRACCIANTI}

fig = plt.figure(figsize=(7.4, 3.05), facecolor=SURF)

# Assi posizionati a mano (frazioni della figura) invece di subplots: tre
# pannelli di uguale larghezza e una sola etichetta x centrata sotto.
LARG, PASSO = 0.276, 0.300
for k, (tr, col, mk) in enumerate(TRACCIANTI):
    nome = TESTI['nomi'][tr]
    d = dati[tr]
    ax = fig.add_axes([0.100 + k * PASSO, 0.235, LARG, 0.640])
    ax.set_facecolor(SURF)

    neg = d[d.suvr_medio < SOGLIA]
    pos = d[d.suvr_medio >= SOGLIA]
    # Correlazione di Spearman (sui ranghi) fra SUVr e deviazione standard:
    # si stampa a terminale, nella figura non compare.
    rho = stats.spearmanr(d.suvr_medio, d.cl_devstd).statistic
    print(f'  {tr}: n={len(d)}  rho={rho:.3f}  mediana neg/pos = '
          f'{neg.cl_devstd.median():.3f} / {pos.cl_devstd.median():.3f} CL')

    ax.axvline(SOGLIA, color=GRID, lw=1, ls=(0, (4, 3)), zorder=1)
    for sub, x0, x1 in ((neg, 0.86, SOGLIA), (pos, SOGLIA, 2.06)):
        ax.plot([x0, x1], [sub.cl_devstd.median()] * 2, color=INK2, lw=1.4,
                zorder=2, solid_capstyle='butt')
    ax.scatter(d.suvr_medio, d.cl_devstd, s=34, marker=mk, color=col,
               edgecolor=SURF, linewidth=1.0, zorder=3)

    ax.set_title(nome, fontsize=9.6, color=INK, pad=6, loc='left')
    # etichette all'estremita' destra dei due segmenti: i soggetti si
    # addensano a sinistra di ogni strato, quindi li' non danno fastidio
    ax.text(SOGLIA - 0.02, neg.cl_devstd.median() + 0.030,
            f'{neg.cl_devstd.median():.3f}', fontsize=7.9, color=INK2,
            ha='right', va='bottom')
    ax.text(2.04, pos.cl_devstd.median() + 0.052,
            f'{pos.cl_devstd.median():.3f}', fontsize=7.9, color=INK2,
            ha='right', va='bottom')

    # Stessi limiti sui tre pannelli, per confrontarli a colpo d'occhio;
    # l'asse x parte da 0.86 come il segmento della mediana dei negativi.
    ax.set_ylim(-0.05, 0.92)
    ax.set_xlim(0.86, 2.06)
    ax.set_xticks([1.0, 1.4, 1.8])
    ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if k == 0:
        ax.set_ylabel(TESTI['y'], labelpad=6)
    else:
        ax.set_yticklabels([])

fig.text(0.537, 0.035, TESTI['x'], fontsize=9.5, color=INK2, ha='center')

# PNG a 300 dpi per anteprime, PDF vettoriale per LaTeX.
cartella = percorsi.dati('figure_tesi')
cartella.mkdir(parents=True, exist_ok=True)
base = ARGS.out or str(cartella / TESTI['base'])
fig.savefig(f'{base}.png', dpi=300, facecolor=SURF)
fig.savefig(f'{base}.pdf', facecolor=SURF)
print(f'scritti {base}.pdf e {base}.png')
