"""Figure di controllo qualita': allineamento PET/MRI e regioni di interesse.

Il valore finale dell'informazione mutua non basta a dire se un allineamento
e' corretto: e' l'obiettivo che l'ottimizzatore ha massimizzato, quindi e' alto
anche quando la ricerca si ferma in un massimo locale sbagliato. Gli errori
grossolani (PET ruotata, serie sbagliata, campo di vista che taglia il
cervelletto) si riconoscono invece subito a occhio. Queste funzioni producono
le immagini su cui fare quel controllo.

  figura_allineamento   PET (scala "hot" + contorno giallo) sopra la MRI,
                        cinque fette per ciascuna delle tre orientazioni.
  figura_voi            target corticale (rosso) e cervelletto (azzurro)
                        sopra la MRI o la PET, in tre sezioni a coordinate
                        mondo fisse, nello stile dei controlli di FreeSurfer.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

# Posizioni (frazione dell'estensione del cervello) delle fette mostrate.
SLICE_FRACTIONS = (0.35, 0.45, 0.55, 0.65, 0.75)
# Righe della figura: nome della sezione e asse dell'array lungo cui si taglia.
AXES = (("Assiale", 2), ("Coronale", 1), ("Sagittale", 0))

CTX_RGB = (0.88, 0.20, 0.18)     # target corticale = rosso
WC_RGB = (0.24, 0.60, 0.73)      # cervelletto = azzurro
ALPHA = 0.65                     # opacita' delle VOI sopra l'immagine


def _pyplot():
    """Importa pyplot solo quando serve, con un backend senza finestra."""
    import matplotlib
    matplotlib.use("Agg")            # nessuno schermo necessario (WSL, SSH)
    import matplotlib.pyplot as plt
    return plt


def _norm(img, lo=1.0, hi=99.0):
    """Normalizza in [0, 1] tagliando ai percentili, per la visualizzazione."""
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros_like(img)
    a, b = np.percentile(finite, [lo, hi])
    if b <= a:
        b = a + 1e-6
    return np.clip((img - a) / (b - a), 0, 1)


# ════════════════════════════════════════════════════════════════════════════
# Allineamento PET su MRI
# ════════════════════════════════════════════════════════════════════════════
def _take(vol, axis, idx):
    """Fetta `idx` lungo `axis`, ruotata di 90 gradi per la visualizzazione."""
    return np.rot90(np.take(vol, idx, axis=axis))


def _brain_bounds(mr, axis, frac):
    """Indice lungo `axis` a una frazione `frac` dell'estensione del cervello."""
    # Il cervello e' approssimato dai voxel MRI sopra il 30o percentile dei
    # valori positivi: basta a escludere lo sfondo per scegliere le fette.
    mask = mr > np.percentile(mr[mr > 0], 30) if (mr > 0).any() else mr > 0
    coords = np.where(mask.any(axis=tuple(a for a in range(3) if a != axis)))[0]
    if coords.size == 0:
        return mr.shape[axis] // 2
    lo, hi = coords.min(), coords.max()
    return int(lo + frac * (hi - lo))


def figura_allineamento(mr_path, pet_path, out_png, titolo: str = "",
                        pet_threshold_pct: float = 55.0) -> str:
    """PET allineata sopra la MRI: se il contorno giallo segue i bordi del
    cervello l'allineamento e' buono, se ne esce e' spostato o ruotato.

    `pet_threshold_pct`: percentile della PET non nulla sotto cui la
    sovrapposizione e' trasparente, per non colorare lo sfondo.
    """
    import nibabel as nib
    plt = _pyplot()

    mr = np.asarray(nib.load(str(mr_path)).get_fdata(), dtype=np.float32)
    pet = np.asarray(nib.load(str(pet_path)).get_fdata(), dtype=np.float32)
    if mr.ndim == 4:
        mr = mr[..., 0]
    if pet.ndim == 4:
        pet = pet[..., 0]
    if mr.shape != pet.shape:
        raise ValueError(f"MRI {mr.shape} e PET {pet.shape} hanno griglie "
                         "diverse: la PET va prima ricampionata (modulo 03).")

    # Sovrapposizione: la PET sotto la soglia viene mascherata (trasparente).
    mr_n, pet_n = _norm(mr), _norm(pet)
    nz = pet[pet > 0]
    thr = np.percentile(nz, pet_threshold_pct) if nz.size else 0
    pet_disp = np.ma.masked_where(pet <= thr, pet_n)

    # Griglia di pannelli: una riga per orientazione, una colonna per fetta.
    nrows, ncols = len(AXES), len(SLICE_FRACTIONS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows))
    for r, (nome, ax_id) in enumerate(AXES):
        for c, frac in enumerate(SLICE_FRACTIONS):
            ax = axes[r, c]
            idx = _brain_bounds(mr, ax_id, frac)
            ax.imshow(_take(mr_n, ax_id, idx), cmap="gray", interpolation="nearest")
            ax.imshow(_take(pet_disp, ax_id, idx), cmap="hot", alpha=0.45,
                      interpolation="nearest")
            try:      # contorno della PET a meta' scala: il segnale da guardare
                ax.contour(_take(pet_n, ax_id, idx), levels=[0.5],
                           colors="yellow", linewidths=0.6, alpha=0.9)
            except Exception:
                pass
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(nome, fontsize=11)
    fig.suptitle((titolo + " — " if titolo else "")
                 + "PET (hot + contorno giallo) sopra la MRI (grigio)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=110)
    plt.close(fig)
    return str(out_png)


# ════════════════════════════════════════════════════════════════════════════
# Regioni di interesse sopra MRI o PET
# ════════════════════════════════════════════════════════════════════════════
def _load_canon(p):
    """Legge un NIfTI in orientazione RAS: (dati, affine, dimensione voxel)."""
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(str(p)))
    return img.get_fdata(), img.affine, img.header.get_zooms()[:3]


def _clampi(v, n):
    """Arrotonda `v` a un indice valido in [0, n-1]."""
    return int(min(max(round(v), 0), n - 1))


def _panel(ax, base2d, ctx2d, wc2d, vline, hline, aspect):
    """Un pannello: sfondo in grigio, VOI colorate, mirino sulle coordinate.

    Il target corticale e' disegnato per ultimo, quindi dove le due VOI si
    sovrappongono prevale il rosso.
    """
    rgb = np.stack([base2d] * 3, -1)
    for m, c in ((wc2d, WC_RGB), (ctx2d, CTX_RGB)):
        mm = m > 0
        for k in range(3):
            rgb[..., k][mm] = (1 - ALPHA) * rgb[..., k][mm] + ALPHA * c[k]
    ax.imshow(rgb, origin="lower", aspect=aspect)
    ax.axvline(vline, color="white", lw=0.8)
    ax.axhline(hline, color="white", lw=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _ricampiona_su(path, riferimento, cartella) -> str:
    """Ricampiona `path` sulla griglia di `riferimento` (interpolazione lineare)."""
    import SimpleITK as sitk
    ref = sitk.ReadImage(str(riferimento))
    img = sitk.ReadImage(str(path), sitk.sitkFloat32)
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(ref)
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetDefaultPixelValue(0)
    out = os.path.join(cartella, "base_ricampionata.nii.gz")
    sitk.WriteImage(rs.Execute(img), out)
    return out


def figura_voi(base_path, voi_ctx_path, voi_wc_path, out_png, titolo: str = "",
               coords=(15.0, 0.0, 0.0), ricampiona_su=None) -> str:
    """Sezioni coronale, sagittale e assiale con le due VOI sovrapposte.

    Le sezioni passano per coordinate mondo fisse (mm, RAS), cosi' le figure
    di soggetti e metodi diversi sono confrontabili. Il rapporto d'aspetto di
    ogni pannello segue la dimensione dei voxel, quindi i volumi anisotropi
    non risultano schiacciati.

    `ricampiona_su`: se indicato (tipicamente la MRI), l'immagine di sfondo
    viene prima portata sulla griglia delle VOI. Serve per mostrarle sopra la
    PET.

    base_path     immagine di sfondo (MRI o PET)
    voi_ctx_path  maschera del target corticale
    voi_wc_path   maschera del cervelletto intero
    out_png       file della figura
    coords        punto (x, y, z) in mm per cui passano le tre sezioni

    Restituisce il percorso del PNG scritto.
    """
    plt = _pyplot()
    with tempfile.TemporaryDirectory() as tmp:
        if ricampiona_su is not None:
            base_path = _ricampiona_su(base_path, ricampiona_su, tmp)
        A, aff, zooms = _load_canon(base_path)
    C, _, _ = _load_canon(voi_ctx_path)
    W, _, _ = _load_canon(voi_wc_path)

    # Dalle coordinate mondo all'indice del voxel, tramite l'affine inversa.
    sx, sy, sz = float(zooms[0]), float(zooms[1]), float(zooms[2])
    base = _norm(A)
    wx, wy, wz = coords
    vx = np.linalg.inv(aff) @ np.array([wx, wy, wz, 1.0])
    cx = _clampi(vx[0], A.shape[0])
    cy = _clampi(vx[1], A.shape[1])
    cz = _clampi(vx[2], A.shape[2])

    fig, axs = plt.subplots(1, 3, figsize=(11, 4.2), facecolor="black")
    # coronale (righe = Z, colonne = X)
    _panel(axs[0], base[:, cy, :].T, C[:, cy, :].T, W[:, cy, :].T, cx, cz, sz / sx)
    axs[0].text(0.03, 0.95, "L", color="w", transform=axs[0].transAxes, fontsize=14, va="top")
    axs[0].text(0.93, 0.95, "R", color="w", transform=axs[0].transAxes, fontsize=14, va="top")
    axs[0].text(0.03, 0.04, f"y={wy:.0f}", color="w", transform=axs[0].transAxes, fontsize=11)
    # sagittale (righe = Z, colonne = Y)
    _panel(axs[1], base[cx, :, :].T, C[cx, :, :].T, W[cx, :, :].T, cy, cz, sz / sy)
    axs[1].text(0.03, 0.04, f"x={wx:.0f}", color="w", transform=axs[1].transAxes, fontsize=11)
    # assiale (righe = Y, colonne = X)
    _panel(axs[2], base[:, :, cz].T, C[:, :, cz].T, W[:, :, cz].T, cx, cy, sy / sx)
    axs[2].text(0.03, 0.95, "L", color="w", transform=axs[2].transAxes, fontsize=14, va="top")
    axs[2].text(0.93, 0.95, "R", color="w", transform=axs[2].transAxes, fontsize=14, va="top")
    axs[2].text(0.03, 0.04, f"z={wz:.0f}", color="w", transform=axs[2].transAxes, fontsize=11)

    fig.suptitle(titolo, color="white", x=0.12, ha="left", fontsize=14, weight="bold")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.04)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=120, facecolor="black")
    plt.close(fig)
    return str(out_png)
