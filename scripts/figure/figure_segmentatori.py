"""Figure del capitolo 2 per i metodi di segmentazione, sulla MRI reale del
soggetto 1018 (florbetaben), nello stile di `figura_multi_otsu.py`.

  fig_kmeans.pdf        k-means: istogramma per gruppo, centri e confini a
                        meta' strada fra i centri, soglie di Otsu a confronto
  fig_fast.pdf          FSL FAST: istogramma di ciascuna classe, che si
                        sovrappongono perche' conta anche il vicinato (campo
                        di Markov)
  fig_samseg.pdf        SAMSEG: intensita' di corteccia, sostanza bianca,
                        nuclei grigi profondi e ventricoli, e la fetta con le
                        strutture
  fig_dl_parcellation.pdf  SynthSeg e FastSurfer: parcellazione completa sulla
                        stessa fetta, stesso colore per la stessa etichetta

Le intensita' sono quelle della T1 corretta con N4 dentro la maschera del
cervello, come in `figura_multi_otsu.py`; le segmentazioni sono quelle salvate
dalla pipeline, non ricalcolate.

Uso:  python -m scripts.figure.figure_segmentatori [--soggetto 1018]
"""
from __future__ import annotations
import argparse, importlib, json, sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared import percorsi                 # noqa: E402
from scripts.figure.figura_multi_otsu import (   # noqa: E402  stesso stile
    INK, INK2, GRID, SOGLIA, CLASSI)

USCITA = percorsi.dati("figure_tesi")
SOGGETTO = "1018"
COLORI_CLASSI = [c for _, c in CLASSI]
BORDO = "#8c8a83"


# ── dati ──────────────────────────────────────────────────────────────────────
def _leggi(*parti):
    """Array (z, y, x) di un NIfTI sotto data/."""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(percorsi.dati(*parti))))


def carica(sid):
    """Carica MRI, MRI dopo N4, maschera del cervello e segmentazioni salvate.

    sid       codice del soggetto

    Restituisce (t1, n4, cervello, seg, info_km, info_ot): seg e' un dizionario
    metodo -> etichette; info_km/info_ot sono gli `seg_extras` di k-means e
    Otsu (centri e soglie), info_ot e' None se Otsu non ha info.json.
    """
    # I metodi classici scrivono in results_fbb_atlas, quelli che producono
    # etichette anatomiche in results_fbb_ants. La maschera del cervello e'
    # quella di Otsu, come in figura_multi_otsu.
    mri = sitk.ReadImage(str(percorsi.dati(
        "aligned_ants", sid, "Realigned_Anatomical",
        "Realigned_Anatomical.nii.gz")), sitk.sitkFloat32)
    cervello = _leggi("results_fbb_atlas", sid, "otsu", "segmentation",
                      "otsu_seg.nii.gz") > 0
    base = importlib.import_module("04_Segmentation.base")
    n4 = sitk.GetArrayFromImage(base.n4_bias_correct(
        mri, mask=cervello.astype(np.uint8), shrink=4))
    seg = {
        "kmeans": _leggi("results_fbb_atlas", sid, "kmeans", "segmentation",
                         "kmeans_seg.nii.gz"),
        "fsl_fast": _leggi("results_fbb_atlas", sid, "fsl_fast", "segmentation",
                           "fsl_fast_seg.nii.gz"),
        "samseg": _leggi("results_fbb_ants", sid, "samseg", "segmentation",
                         "samseg_seg.nii.gz"),
        "synthseg": _leggi("results_fbb_ants", sid, "synthseg", "segmentation",
                           "synthseg_seg.nii.gz"),
        "fastsurfer": _leggi("results_fbb_ants", sid, "fastsurfer", "segmentation",
                             "fastsurfer_seg.nii.gz"),
    }
    info_km = json.loads(percorsi.dati("results_fbb_atlas", sid, "kmeans",
                                       "info.json").read_text())["seg_extras"]
    info_ot = json.loads(percorsi.dati("results_fbb_atlas", sid, "otsu",
                                       "info.json").read_text())["seg_extras"] \
        if percorsi.dati("results_fbb_atlas", sid, "otsu", "info.json").exists() \
        else None
    return sitk.GetArrayFromImage(mri), n4, cervello, seg, info_km, info_ot


# ── utilita' grafiche ─────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})


def _fetta(cervello):
    """Fetta assiale all'altezza dei ventricoli, come in fig_multi_otsu."""
    zz = np.where(cervello.any(axis=(1, 2)))[0]
    return int(zz.min() + 0.66 * (zz.max() - zz.min()))


def _ritaglio(cervello, z, m=4):
    """Riquadro attorno al cervello nella fetta capovolta, con m voxel di margine."""
    r, c = np.where(np.flipud(cervello[z]))
    return slice(r.min() - m, r.max() + m), slice(c.min() - m, c.max() + m)


def _immagine(ax, t1, z, rit, etichette=None, cmap=None, alpha=0.85):
    """Fetta della T1 (1-99.5 percentile) con le etichette sopra, se date.

    Le etichette devono essere 1..cmap.N: vmin/vmax a mezzo passo fanno
    corrispondere ogni intero a un colore della tavolozza; lo 0 e' trasparente.
    """
    sl = np.flipud(t1[z])[rit]
    a, b = np.percentile(sl[sl > 0], [1, 99.5])
    ax.imshow(np.clip((sl - a) / (b - a), 0, 1), cmap="gray",
              interpolation="nearest")
    if etichette is not None:
        e = np.flipud(etichette[z])[rit]
        ax.imshow(np.ma.masked_where(e == 0, e), cmap=cmap, alpha=alpha,
                  interpolation="nearest", vmin=0.5, vmax=cmap.N + 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _asse_istogramma(ax, hi):
    """Assi comuni agli istogrammi di intensita', conteggi in migliaia."""
    ax.set_xlim(0, hi)
    ax.set_xlabel("T1 intensity after N4 correction")
    ax.set_ylabel("voxels (thousands)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y/1000:.0f}"))
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def _legenda_classi(fig, voci, y=0.0, ncol=None):
    """Legenda sotto la figura, di norma tutte le voci su una riga."""
    fig.legend(handles=voci, loc="lower center", ncol=ncol or len(voci),
               frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, y), labelcolor=INK)


def _patch(nome, col):
    """Voce di legenda: quadratino col bordo grigio (serve alla classe chiara)."""
    return Patch(facecolor=col, edgecolor=BORDO, linewidth=0.5, label=nome)


# ── 1. k-means ────────────────────────────────────────────────────────────────
def fig_kmeans(t1, n4, cervello, seg, info_km, info_ot):
    """fig_kmeans (capitolo 2): istogramma per gruppo, centri e confini.

    Restituisce (centri, confini) per la stampa a terminale.
    """
    lab = seg["kmeans"]
    hi = np.percentile(n4[lab > 0], 99.8)
    bins = np.linspace(0, hi, 90)
    # In una dimensione k-means assegna ogni voxel al centro piu' vicino: i
    # confini fra le classi sono i punti medi fra centri consecutivi.
    centri = np.sort(np.asarray(info_km["centroids"], float))
    confini = (centri[:-1] + centri[1:]) / 2

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3),
                             gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    for k, col in enumerate(COLORI_CLASSI, start=1):
        ax.hist(n4[lab == k], bins=bins, color=col, zorder=2,
                edgecolor=BORDO if k == 3 else "white", linewidth=0.4)
    for i, c in enumerate(centri, start=1):
        # Triangolo sopra il bordo superiore (coordinate miste: x in dati,
        # y in frazione dell'asse), fuori dall'area dei dati.
        ax.plot([c], [1.0], marker="v", color=INK, markersize=6, clip_on=False,
                transform=ax.get_xaxis_transform(), zorder=4)
        ax.text(c, 1.02, f"$\\mu_{i}$", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9.5, color=INK)
    for b in confini:
        ax.axvline(b, color=SOGLIA, linewidth=1.3, linestyle=(0, (4, 2.5)),
                   zorder=3)
    # Soglie di Otsu in punteggiato, per confronto con i confini di k-means.
    if info_ot:
        for t in info_ot["thresholds"]:
            ax.axvline(t, color=INK2, linewidth=0.9, linestyle=":", zorder=3)
    _asse_istogramma(ax, hi)

    z = _fetta(cervello)
    rit = _ritaglio(cervello, z)
    _immagine(axes[1], t1, z, rit, lab, ListedColormap(COLORI_CLASSI))
    fig.tight_layout(rect=(0, 0.13, 1, 0.95), w_pad=1.5)
    voci = [_patch(n, c) for (n, _), c in zip(CLASSI, COLORI_CLASSI)]
    voci += [Line2D([], [], color=SOGLIA, linestyle=(0, (4, 2.5)),
                    label="halfway between centres")]
    if info_ot:
        voci += [Line2D([], [], color=INK2, linestyle=":", label="Otsu thresholds")]
    _legenda_classi(fig, voci, ncol=3)
    fig.savefig(USCITA / "fig_kmeans.pdf")
    plt.close(fig)
    return centri, confini


# ── 2. FSL FAST ───────────────────────────────────────────────────────────────
def fig_fast(t1, n4, cervello, seg):
    """fig_fast (capitolo 2): istogrammi sovrapposti delle classi di FAST."""
    lab = seg["fsl_fast"]
    hi = np.percentile(n4[lab > 0], 99.8)
    bins = np.linspace(0, hi, 90)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    # Riempimento trasparente piu' contorno, perche' le classi si sovrappongono;
    # il contorno della sostanza bianca e' scuro perche' il suo colore e' chiaro.
    linee = {1: COLORI_CLASSI[0], 2: COLORI_CLASSI[1], 3: "#6f6b5e"}
    for k in (1, 2, 3):
        v = n4[lab == k]
        ax.hist(v, bins=bins, histtype="stepfilled", color=COLORI_CLASSI[k - 1],
                alpha=0.45, zorder=2)
        ax.hist(v, bins=bins, histtype="step", color=linee[k], linewidth=1.3,
                zorder=3)
    _asse_istogramma(ax, hi)
    z = _fetta(cervello)
    rit = _ritaglio(cervello, z)
    _immagine(axes[1], t1, z, rit, lab, ListedColormap(COLORI_CLASSI))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.5)
    _legenda_classi(fig, [_patch(n, c) for (n, _), c in zip(CLASSI, COLORI_CLASSI)])
    fig.savefig(USCITA / "fig_fast.pdf")
    plt.close(fig)


# ── 3. SAMSEG ─────────────────────────────────────────────────────────────────
# Etichette FreeSurfer raggruppate in quattro tessuti (sinistra e destra
# insieme): ventricoli laterali, inferiori, terzo e quarto; corteccia;
# talamo, caudato, putamen e pallido; sostanza bianca cerebrale.
GRUPPI_SAMSEG = [
    ("ventricles", (4, 43, 5, 44, 14, 15), "#4c78a8"),
    ("cerebral cortex", (3, 42), "#e3a65b"),
    ("deep grey nuclei", (10, 49, 11, 50, 12, 51, 13, 52), "#8e6bb0"),
    ("cerebral white matter", (2, 41), "#efe9d6"),
]


def fig_samseg(t1, n4, cervello, seg):
    """fig_samseg (capitolo 2): intensita' di quattro gruppi di strutture."""
    lab = seg["samseg"]
    gruppi = np.zeros_like(lab, dtype=np.int16)
    for g, (_, ids, _) in enumerate(GRUPPI_SAMSEG, start=1):
        gruppi[np.isin(lab, ids)] = g
    # Limite destro preso sulla sostanza bianca, il tessuto piu' chiaro.
    hi = np.percentile(n4[np.isin(lab, (2, 41))], 99.8)
    bins = np.linspace(0, hi, 90)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    for g, (nome, ids, col) in enumerate(GRUPPI_SAMSEG, start=1):
        v = n4[gruppi == g]
        # Frazione invece del conteggio: le strutture hanno volumi molto
        # diversi e i ventricoli sparirebbero accanto alla sostanza bianca.
        dens = np.histogram(v, bins=bins)[0] / max(len(v), 1)
        centri = (bins[:-1] + bins[1:]) / 2
        ax.fill_between(centri, dens, color=col, alpha=0.45, zorder=2, step="mid")
        ax.step(centri, dens, where="mid", zorder=3, linewidth=1.3,
                color=BORDO if col == "#efe9d6" else col)
    ax.set_xlim(0, hi)
    ax.set_xlabel("T1 intensity after N4 correction")
    ax.set_ylabel("fraction of the structure's voxels")
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    z = _fetta(cervello)
    rit = _ritaglio(cervello, z)
    _immagine(axes[1], t1, z, rit, gruppi,
              ListedColormap([c for _, _, c in GRUPPI_SAMSEG]))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.5)
    _legenda_classi(fig, [_patch(n, c) for n, _, c in GRUPPI_SAMSEG])
    fig.savefig(USCITA / "fig_samseg.pdf")
    plt.close(fig)


# ── 4. SynthSeg e FastSurfer ──────────────────────────────────────────────────
# emisfero destro -> sinistro, cosi' le strutture omologhe hanno lo stesso colore
_DESTRA = {41: 2, 42: 3, 43: 4, 44: 5, 46: 7, 47: 8, 49: 10, 50: 11, 51: 12,
           52: 13, 53: 17, 54: 18, 58: 26, 60: 28, 63: 31}


# Colori fissi per sostanza bianca (2), ventricoli (4, 5, 14, 15) e liquor (24),
# uguali a quelli delle figure precedenti.
FISSI = {2: "#efe9d6", 4: "#4c78a8", 5: "#4c78a8", 14: "#4c78a8",
         15: "#4c78a8", 24: "#9fb8d0"}


def _chiave(l):
    """Etichetta FreeSurfer -> chiave di colore, unendo i due emisferi.

    Le parcelle corticali destre (2000-2999) diventano le sinistre (1000-1999),
    le strutture destre sottocorticali passano per _DESTRA.
    """
    l = int(l)
    if 2000 <= l < 3000:
        return l - 1000
    return _DESTRA.get(l, l)


def _tavolozza(chiavi):
    """Un colore per chiave, mescolando tab20, tab20b e Set3."""
    base = (list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
            + list(plt.get_cmap("Set3").colors))
    rng = np.random.default_rng(7)           # ordine fisso, colori vicini diversi
    ordine = rng.permutation(len(base))
    col = {k: base[ordine[i % len(base)]] for i, k in enumerate(sorted(chiavi))}
    # tessuti di fondo con i colori delle altre figure, cosi' risaltano le
    # regioni: sostanza bianca, ventricoli, liquor
    col.update({k: c for k, c in FISSI.items() if k in col})
    return col


def fig_dl(t1, cervello, seg):
    """fig_dl_parcellation (capitolo 2): SynthSeg e FastSurfer sulla stessa fetta."""
    # Tavolozza costruita sull'unione delle etichette dei due metodi, cosi' la
    # stessa struttura ha lo stesso colore nei due pannelli.
    metodi = [("synthseg", "SynthSeg"), ("fastsurfer", "FastSurfer")]
    chiavi = {_chiave(l) for m, _ in metodi for l in np.unique(seg[m]) if l > 0}
    col = _tavolozza(chiavi)
    ordinate = sorted(chiavi)
    cmap = ListedColormap([col[k] for k in ordinate])
    indice = {k: i + 1 for i, k in enumerate(ordinate)}

    z = _fetta(cervello)
    rit = _ritaglio(cervello, z)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1))
    _immagine(axes[0], t1, z, rit)
    axes[0].set_title("T1 MRI", color=INK)
    for ax, (m, nome) in zip(axes[1:], metodi):
        l = seg[m]
        # Si rimappa solo la fetta disegnata, nell'indice della tavolozza.
        mappa = np.zeros_like(l, dtype=np.int16)
        for v in np.unique(l[z]):
            if v > 0:
                mappa[z][l[z] == v] = indice[_chiave(v)]
        _immagine(ax, t1, z, rit, mappa, cmap, alpha=0.9)
        n = len([v for v in np.unique(l) if v > 0])
        ax.set_title(f"{nome} ({n} labels)", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(USCITA / "fig_dl_parcellation.pdf")
    plt.close(fig)


FIGURE = ["kmeans", "fast", "samseg", "dl"]


def main() -> int:
    """Disegna le figure scelte con --solo (predefinito: tutte)."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--soggetto", default=SOGGETTO)
    ap.add_argument("--solo", nargs="*", choices=FIGURE)
    a = ap.parse_args()
    USCITA.mkdir(parents=True, exist_ok=True)
    t1, n4, cervello, seg, info_km, info_ot = carica(a.soggetto)
    for nome in (a.solo or FIGURE):
        if nome == "kmeans":
            c, b = fig_kmeans(t1, n4, cervello, seg, info_km, info_ot)
            print("k-means: centri", np.round(c, 1), "confini", np.round(b, 1))
        elif nome == "fast":
            fig_fast(t1, n4, cervello, seg)
        elif nome == "samseg":
            fig_samseg(t1, n4, cervello, seg)
        else:
            fig_dl(t1, cervello, seg)
        print(f"{nome} fatto")
    print(f"Scritte in {USCITA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
