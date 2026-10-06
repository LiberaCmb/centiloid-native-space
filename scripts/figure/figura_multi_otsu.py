"""Multi-Otsu su una MRI reale, per il capitolo 2.

Riesegue la segmentazione Otsu a tre classi del modulo 04 sul soggetto 1018
(florbetaben, lo stesso usato come PET negativa in `figure_capitoli.py`), con
la stessa maschera del cervello e la stessa correzione N4 della pipeline, e
disegna:

  a sinistra  l'istogramma delle intensita' nel cervello, colorato per classe,
              con le due soglie t1* e t2*;
  a destra    una fetta assiale con le tre classi sopra la MRI.

La maschera del cervello si ricava dalla segmentazione Otsu gia' salvata
(ogni voxel del cervello ha un'etichetta 1, 2 o 3), quindi non serve
ripetere la registrazione dell'atlante.

Uso:  python -m scripts.figure.figura_multi_otsu
Uscita: data/figure_tesi/fig_multi_otsu.pdf
"""
from __future__ import annotations
import argparse, importlib, sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared import percorsi                 # noqa: E402

_seg = importlib.import_module("04_Segmentation")
USCITA = percorsi.dati("figure_tesi")
SOGGETTO = "1018"

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
SOGLIA = "#c0392b"                           # come la soglia di fig_otsu
CLASSI = [("CSF", "#4c78a8"), ("grey matter", "#e3a65b"),
          ("white matter", "#efe9d6")]

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "pdf.fonttype": 42,
    "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def segmenta(sid=SOGGETTO):
    """Rifa' la segmentazione Otsu del soggetto come nel modulo 04.

    sid       codice del soggetto (predefinito 1018)

    Restituisce (arr, lab, t1, soglie): MRI dopo N4, mappa delle etichette
    0-3, MRI originale (tutte come array z, y, x) e le due soglie di Otsu.
    """
    mri = sitk.ReadImage(str(percorsi.dati(
        "aligned_ants", sid, "Realigned_Anatomical",
        "Realigned_Anatomical.nii.gz")), sitk.sitkFloat32)
    salvata = sitk.ReadImage(str(percorsi.dati(
        "results_fbb_atlas", sid, "otsu", "segmentation", "otsu_seg.nii.gz")))
    maschera = sitk.Cast(salvata > 0, sitk.sitkUInt8)
    # La segmentazione salvata e' sulla griglia della MRI allineata: si copiano
    # i metadati della MRI, cosi' SimpleITK le tratta come la stessa griglia.
    maschera.CopyInformation(mri)

    # stessa correzione N4 del segmentatore, per avere le intensita' su cui
    # sono state calcolate le soglie
    base = importlib.import_module("04_Segmentation.base")
    corr = base.n4_bias_correct(mri, mask=sitk.GetArrayFromImage(maschera), shrink=4)
    res = _seg.get_segmenter("otsu").segment(mri=mri, external_brain_mask=maschera)

    arr = sitk.GetArrayFromImage(corr)                    # (z, y, x)
    lab = sitk.GetArrayFromImage(res.label_map)
    t1 = sitk.GetArrayFromImage(mri)
    soglie = res.extras["thresholds"]
    return arr, lab, t1, np.asarray(soglie, float)


def disegna(arr, lab, t1, soglie):
    """Istogramma per classe con le soglie e fetta assiale (fig_multi_otsu, cap. 2)."""
    dentro = lab > 0
    v = arr[dentro]
    # Il limite destro al 99.8-esimo percentile toglie la coda di pochi voxel
    # molto chiari che schiaccerebbe l'istogramma.
    hi = np.percentile(v, 99.8)
    bins = np.linspace(0, hi, 90)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    for k, (nome, col) in enumerate(CLASSI, start=1):
        # La sostanza bianca e' quasi bianca: le serve un bordo grigio.
        ax.hist(arr[lab == k], bins=bins, color=col, edgecolor="#8c8a83"
                if k == 3 else "white", linewidth=0.4, zorder=2, label=nome)
    for i, t in enumerate(soglie, start=1):
        ax.axvline(t, color=SOGLIA, linewidth=1.4, zorder=3)
        ax.text(t, 1.0, f" $t_{i}^\\star$", transform=ax.get_xaxis_transform(),
                color=SOGLIA, ha="left", va="top", fontsize=10)
    ax.set_xlim(0, hi)
    ax.set_xlabel("T1 intensity after N4 correction")
    ax.set_ylabel("voxels in the brain (thousands)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y/1000:.0f}"))
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # fetta assiale all'altezza dei ventricoli laterali (circa i due terzi
    # dell'estensione verticale della maschera), dove le tre classi si vedono
    zz = np.where(dentro.any(axis=(1, 2)))[0]
    z = int(zz.min() + 0.66 * (zz.max() - zz.min()))
    # flipud: l'asse y dell'array cresce verso il basso, cosi' la parte
    # anteriore finisce in alto. Poi si ritaglia sul cervello con 4 voxel di
    # margine e si normalizza la MRI fra l'1 e il 99.5 percentile.
    sl_t1, sl_lab = np.flipud(t1[z]), np.flipud(lab[z])
    r, c = np.where(sl_lab > 0)
    r0, r1, c0, c1 = r.min() - 4, r.max() + 4, c.min() - 4, c.max() + 4
    sl_t1, sl_lab = sl_t1[r0:r1, c0:c1], sl_lab[r0:r1, c0:c1]
    a, b = np.percentile(sl_t1[sl_t1 > 0], [1, 99.5])
    ax = axes[1]
    ax.imshow(np.clip((sl_t1 - a) / (b - a), 0, 1), cmap="gray",
              interpolation="nearest")
    # Etichette 1-3 su tre colori discreti (vmin/vmax a mezzo passo); lo
    # sfondo 0 e' mascherato e resta trasparente.
    ax.imshow(np.ma.masked_where(sl_lab == 0, sl_lab),
              cmap=ListedColormap([c for _, c in CLASSI]), vmin=0.5, vmax=3.5,
              alpha=0.85, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # L'8% in basso resta libero per la legenda comune ai due pannelli.
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.5)
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#8c8a83", linewidth=0.5,
                              label=n) for n, c in CLASSI],
               loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0), labelcolor=INK)
    fig.savefig(USCITA / "fig_multi_otsu.pdf")
    plt.close(fig)


def main() -> int:
    """Segmenta, stampa soglie e numero di voxel per classe, scrive il PDF."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--soggetto", default=SOGGETTO)
    a = ap.parse_args()
    USCITA.mkdir(parents=True, exist_ok=True)
    arr, lab, t1, soglie = segmenta(a.soggetto)
    print("soglie:", np.round(soglie, 1))
    for k, (nome, _) in enumerate(CLASSI, start=1):
        print(f"  {nome}: {int((lab == k).sum())} voxel")
    disegna(arr, lab, t1, soglie)
    print(f"Scritta {USCITA / 'fig_multi_otsu.pdf'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
