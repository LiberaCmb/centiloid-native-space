"""Figure anatomiche per i capitoli 1, 3 e 4.

Produce quattro PDF in data/figure_tesi. Le etichette sono in inglese.

  fig_pet_pos_neg.pdf     capitolo 1: una PET amiloide-positiva e una negativa
                          affiancate, entrambe divise per la media nel
                          cervelletto, cosi' la scala di colore E' il SUVr
  fig_alignment_qc.pdf    capitolo 3: la PET riallineata sopra la MR con i
                          contorni, cioe' quello che il modulo 03 fa guardare
  fig_voi_two_methods.pdf capitolo 3: la stessa PET con la corteccia bersaglio
                          di due metodi diversi, e i due SUVr che ne escono
  fig_dlbs_excluded.pdf   capitolo 4: uno dei sette soggetti DLBS esclusi, in
                          cui la corteccia non e' piu' brillante del
                          cervelletto eppure il riferimento lo dice positivo

Uso:  python -m scripts.figure.figure_capitoli
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared import percorsi                 # noqa: E402
from shared.percorsi import risolvi         # noqa: E402
OUT = percorsi.dati("figure_tesi")

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 9.5, "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
})

# soggetti scelti a mano, uno per ciascun ruolo
POSITIVO = ("1009", "data/results_fbb_ants", "data/aligned_ants")
NEGATIVO = ("1018", "data/results_fbb_ants", "data/aligned_ants")
ESCLUSO  = ("sub-4288", "data/results_dlbs", "data/aligned_dlbs_ants")


def carica(p: Path):
    """Volume riorientato in RAS, con la sua matrice e i suoi passi."""
    img = nib.as_closest_canonical(nib.load(str(p)))
    return np.asarray(img.dataobj, dtype=float), img.header.get_zooms()[:3]


def risultato(base: str, sub: str, metodo: str) -> pd.Series:
    """Legge il result.csv del metodo, cercandolo anche nel ramo _atlas.

    I metodi che leggono etichette scrivono in results_<tracciante>_ants, i
    classici in results_<tracciante>_atlas: il nome della cartella dipende da
    come la VOI e' stata ricavata, non dal metodo.
    """
    for b in (base, base.replace("_ants", "_atlas")):
        f = risolvi(b) / sub / metodo / "result.csv"
        if f.exists():
            return pd.read_csv(f).iloc[0]
    raise FileNotFoundError(f"nessun result.csv per {sub}/{metodo}")


def pet_normalizzata(base_al: str, base_ris: str, sub: str, metodo="synthseg"):
    """PET riallineata divisa per la media nel cervelletto di quel soggetto.

    Dividere per il denominatore del SUVr toglie di mezzo dose, peso e
    calibrazione dello scanner, e rende la scala di colore confrontabile fra
    soggetti: il valore letto in un voxel e' il suo rapporto rispetto al
    cervelletto, cioe' esattamente cio' che il SUVr media sulla corteccia.
    """
    d = risultato(base_ris, sub, metodo)
    pet, zoom = carica(risolvi(base_al) / sub / "Realigned_Functional" /
                       "Realigned_Functional.nii.gz")
    return pet / float(d["wc_mean"]), zoom, d


def assiale(vol: np.ndarray, frazione=0.60):
    """Fetta assiale a una frazione dell'altezza del volume, anteriore in alto.

    Il 60% dell'asse z (RAS) cade di solito sopra i ventricoli, dove si vede
    bene la corteccia; rot90 porta l'asse y del volume in verticale.
    """
    k = int(round(vol.shape[2] * frazione))
    return np.rot90(vol[:, :, k])


def coronale(vol: np.ndarray, frazione=0.38):
    """Fetta coronale a una frazione della profondita' (asse y), testa in alto."""
    j = int(round(vol.shape[1] * frazione))
    return np.rot90(vol[:, j, :])


def _spegni(ax):
    """Toglie tacche e cornice: per le immagini non servono."""
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ── 1. PET positiva e negativa, capitolo 1 ───────────────────────────────────
def fig_pet_pos_neg():
    """fig_pet_pos_neg (capitolo 1): PET positiva e negativa, stessa scala."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.4))
    im = None
    for ax, (sub, bris, bal), etichetta in zip(
            axes, (POSITIVO, NEGATIVO), ("amyloid-positive", "amyloid-negative")):
        pet, zoom, d = pet_normalizzata(bal, bris, sub)
        # Scala fissa 0-2 su entrambe: con la PET divisa per il cervelletto il
        # colore si legge direttamente come rapporto. aspect compensa voxel
        # non isotropi nel piano della fetta.
        im = ax.imshow(assiale(pet), cmap="inferno", vmin=0, vmax=2.0,
                       aspect=zoom[1] / zoom[0])
        ax.set_title(f"{etichetta}\nSUVr {float(d['suvr']):.2f}", color=INK)
        _spegni(ax)
    cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cb.set_label("uptake relative to the cerebellum", color=INK2, fontsize=8.5)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    cb.outline.set_visible(False)
    fig.savefig(OUT / "fig_pet_pos_neg.pdf", bbox_inches="tight")
    plt.close(fig)


# ── 2. Controllo di qualita' dell'allineamento, capitolo 3 ───────────────────
def fig_alignment_qc():
    """fig_alignment_qc (capitolo 3): PET sopra la MR, come nel modulo 03."""
    sub, bris, bal = POSITIVO
    d = risultato(bris, sub, "synthseg")
    mr, zoom = carica(risolvi(bal) / sub / "Realigned_Anatomical" /
                      "Realigned_Anatomical.nii.gz")
    pet, _, _ = pet_normalizzata(bal, bris, sub)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    viste = (("axial", assiale), ("coronal", coronale))
    for ax, (nome, taglia) in zip(axes[:2], viste):
        a, p = taglia(mr), taglia(pet)
        ax.imshow(a, cmap="gray", aspect=zoom[1] / zoom[0])
        # Sotto 0.35 volte il cervelletto (fondo, fuori dalla testa) la PET
        # resta trasparente; il contorno giallo a 0.9 segue il bordo del
        # cervello che la PET vede, da confrontare con quello della MR.
        ax.imshow(np.ma.masked_where(p < 0.35, p), cmap="hot", alpha=0.45,
                  vmin=0, vmax=2.0, aspect=zoom[1] / zoom[0])
        ax.contour(p, levels=[0.9], colors="#ffd400", linewidths=0.7)
        ax.set_title(nome, color=INK)
        _spegni(ax)
    # Terzo pannello: solo il contorno sulla MR, senza la PET che copre.
    a, p = assiale(mr), assiale(pet)
    axes[2].imshow(a, cmap="gray", aspect=zoom[1] / zoom[0])
    axes[2].contour(p, levels=[0.9], colors="#ffd400", linewidths=0.8)
    axes[2].set_title("contours alone", color=INK)
    _spegni(axes[2])
    fig.suptitle("Realigned PET over the subject's own MR", color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_alignment_qc.pdf", bbox_inches="tight")
    plt.close(fig)


# ── 3. Due metodi, due corteccie, due SUVr, capitolo 3 ───────────────────────
def fig_voi_two_methods(metodi=("synthseg", "fsl_fast"),
                        colori=("#2a78d6", "#1baf7a")):
    """fig_voi_two_methods (capitolo 3): stessa PET, target di due metodi.

    metodi    i due segmentatori da confrontare (cartelle dei risultati)
    colori    colore del contorno di ciascun metodo
    """
    sub, bris, bal = POSITIVO
    pet, zoom, _ = pet_normalizzata(bal, bris, sub)
    sfondo = assiale(pet)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    axes[0].imshow(sfondo, cmap="inferno", vmin=0, vmax=2.0,
                   aspect=zoom[1] / zoom[0])
    axes[0].set_title("PET alone\n", color=INK)
    _spegni(axes[0])

    for ax, met, col in zip(axes[1:], metodi, colori):
        d = risultato(bris, sub, met)
        # La VOI e' sulla griglia della MR, la stessa della PET riallineata:
        # la fetta assiale coincide con quella dello sfondo.
        voi, _ = carica(risolvi(d["voi_ctx_path"]))
        ax.imshow(sfondo, cmap="inferno", vmin=0, vmax=2.0,
                  aspect=zoom[1] / zoom[0])
        m = assiale(voi)
        ax.contour(m, levels=[0.5], colors=col, linewidths=0.8)
        nome = {"synthseg": "SynthSeg", "fsl_fast": "FSL FAST"}.get(met, met)
        ax.set_title(f"{nome}\nSUVr {float(d['suvr']):.2f}", color=INK)
        _spegni(ax)
    fig.suptitle("The same PET, two target cortices", color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_voi_two_methods.pdf", bbox_inches="tight")
    plt.close(fig)


# ── 4. Un soggetto DLBS escluso, capitolo 4 ──────────────────────────────────
def fig_dlbs_excluded():
    """fig_dlbs_excluded (capitolo 4): soggetto DLBS escluso, con le due VOI."""
    sub, bris, bal = ESCLUSO
    d = risultato(bris, sub, "synthseg")
    pet, zoom, _ = pet_normalizzata(bal, bris, sub)
    ctx, _ = carica(risolvi(d["voi_ctx_path"]))
    wc, _ = carica(risolvi(d["voi_wc_path"]))

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3))
    im = None
    for ax, (nome, taglia) in zip(axes, (("axial", assiale), ("coronal", coronale))):
        im = ax.imshow(taglia(pet), cmap="inferno", vmin=0, vmax=2.0,
                       aspect=zoom[1] / zoom[0])
        ax.contour(taglia(ctx), levels=[0.5], colors="#2a78d6", linewidths=0.7)
        ax.contour(taglia(wc), levels=[0.5], colors="#1baf7a", linewidths=0.7)
        ax.set_title(nome, color=INK)
        _spegni(ax)
    cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cb.set_label("uptake relative to the cerebellum", color=INK2, fontsize=8.5)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(f"{sub}: target cortex in blue, cerebellum in green, "
                 f"SUVr {float(d['suvr']):.2f}", color=INK, y=1.06, fontsize=9)
    fig.savefig(OUT / "fig_dlbs_excluded.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    """Scrive i quattro PDF in data/figure_tesi."""
    # Nessun argomento: il parser serve solo per --help.
    argparse.ArgumentParser().parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_pet_pos_neg()
    print("fig_pet_pos_neg.pdf")
    fig_alignment_qc()
    print("fig_alignment_qc.pdf")
    fig_voi_two_methods()
    print("fig_voi_two_methods.pdf")
    fig_dlbs_excluded()
    print("fig_dlbs_excluded.pdf")
    print(f"\nScritte in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
