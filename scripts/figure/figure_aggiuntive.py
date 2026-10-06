"""Dieci figure aggiuntive per i capitoli 3 e 4, nello stesso stile di
`figure_risultati.py` (PDF vettoriali, etichette in inglese, colori per
famiglia di metodi).

  fig_seg_seven_methods.pdf       capitolo 3, VOI: lo stesso cervello visto
                                  dai sette metodi, target in rosso e
                                  cervelletto in azzurro
  fig_seg_label_methods.pdf       capitolo 3, VOI: le regioni lette dalle
                                  etichette di SynthSeg, FastSurfer e SAMSEG
                                  (soggetto 1030, florbetaben), target in
                                  rosso e cervelletto in azzurro
  fig_samseg_vs_atlas.pdf         capitolo 3, VOI: SAMSEG e samseg_atlas, la
                                  stessa segmentazione con due target diversi
  fig_atlas_transfer.pdf          capitolo 3, VOI: l'altra via, per i quattro
                                  metodi senza parcellazione: VOI standard sul
                                  template ICBM152, la stessa deformata sulla
                                  MRI del soggetto, intersezione con la grigia
                                  di Otsu
  fig_registration_seeds.pdf      capitolo 3, registrazione: variabilita' della
                                  mutua informazione di SimpleITK al cambiare
                                  del seme, 79 soggetti per 10 tentativi
  fig_res_fbp_exclusion.pdf       capitolo 4, apertura: SUVr di SynthSeg sui
                                  soggetti florbetapir, soglia a 0.85, e media
                                  della corteccia in scala logaritmica
  fig_res_calibration_scatter.pdf capitolo 4, calibrazioni: centiloide stimato
                                  contro riferimento per SynthSeg, formula
                                  pubblicata e retta ricalibrata
  fig_dilation_mask.pdf           capitolo 4, dilatazione: ingrandimento del
                                  target e dell'anello aggiunto, quattro metodi
  fig_freesurfer_comparison.pdf   capitolo 4, Dice: target e cervelletto di
                                  recon-all, SynthSeg e FastSurfer su un
                                  soggetto DLBS, in tre viste
  fig_res_dilation.pdf            capitolo 4, dilatazione: MAE senza e con la
                                  dilatazione di un voxel, quattro archivi

La figura dell'atlante richiede una registrazione SyN del template sulla MRI
(qualche minuto): il risultato viene salvato in
data/figure_tesi/cache_atlante_<soggetto>/ e riusato alle esecuzioni
successive.

Uso:  python -m scripts.figure.figure_aggiuntive [--solo NOME ...]
"""
from __future__ import annotations
import argparse, importlib, sys
from pathlib import Path

import numpy as np
import pandas as pd
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
from scripts.figure.figure_risultati import (   # noqa: E402  stesso stile
    COL, FAM, ETICHETTA, TRACCIANTE, INK, INK2, GRID, _pulisci, _legenda_famiglie)

_cmp = importlib.import_module("07_Comparison")
D = percorsi.CONFRONTO
USCITA = percorsi.dati("figure_tesi")

METODI = ["synthseg", "fastsurfer", "samseg", "samseg_atlas",
          "otsu", "kmeans", "fsl_fast"]
# Rosso e azzurro come nei controlli visivi delle VOI (qc_voi_overlay).
ROSSO, AZZURRO = "#d62728", "#1f9bd6"
# Florbetaben, soggetto tipico: errore <= 4 CL con tutti i metodi.
SOGGETTO = "1030"


# ── utilita' per le immagini ──────────────────────────────────────────────────
def _carica(path):
    """Volume in orientamento RAS, come array float (x, y, z)."""
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(str(path)))
    return np.asarray(img.dataobj, dtype=np.float32)


def _coronale(vol, y):
    """Fetta coronale pronta per imshow, alto in alto e sinistra del soggetto a
    sinistra, come nelle figure di `figure_capitoli.py`."""
    return np.flipud(vol[:, y, :].T)


def _finestra(sl, lo=1, hi=99.5):
    """Normalizza la fetta fra i percentili lo e hi dei voxel non nulli."""
    v = sl[sl > 0]
    if v.size == 0:
        return sl
    a, b = np.percentile(v, [lo, hi])
    return np.clip((sl - a) / (b - a + 1e-9), 0, 1)


def _ritaglia(*fette, sopra=28, margine=10):
    """Ritaglio comune attorno alle VOI (fette successive alla prima, che e'
    la MRI): si tiene il cervello e si lasciano fuori collo e spalle."""
    m = np.zeros(fette[0].shape, bool)
    for f in fette[1:]:
        m |= f > 0.5
    if not m.any():
        m = fette[0] > np.percentile(fette[0][fette[0] > 0], 20)
    r, c = np.where(m)
    # sopra le VOI c'e' ancora corteccia (non tutta e' target) e il cranio
    r0, r1 = max(r.min() - sopra, 0), r.max() + margine
    c0, c1 = max(c.min() - margine, 0), c.max() + margine
    return [f[r0:r1, c0:c1] for f in fette]


def _sovrapponi(ax, mri, ctx, wc, alpha=0.75):
    """MRI in grigio con target (rosso) e cervelletto (azzurro) sopra."""
    ax.imshow(mri, cmap="gray", interpolation="nearest")
    for m, col in ((ctx, ROSSO), (wc, AZZURRO)):
        ax.imshow(np.ma.masked_where(m < 0.5, m), cmap=ListedColormap([col]),
                  alpha=alpha, interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _legenda_voi(fig, y=0.0, extra=()):
    """Legenda target/cervelletto sotto la figura, con eventuali voci in piu'."""
    voci = [Patch(facecolor=ROSSO, label="cortical target"),
            Patch(facecolor=AZZURRO, label="whole cerebellum"), *extra]
    fig.legend(handles=voci, loc="lower center", ncol=len(voci), frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, y), labelcolor=INK)


def _voi_metodo(metodo, sid=SOGGETTO):
    """(target, cervelletto) del metodo per un soggetto florbetaben.

    I tre metodi classici scrivono nel ramo _atlas, gli altri nel ramo _ants.
    """
    cart = "results_fbb_atlas" if metodo in ("otsu", "kmeans", "fsl_fast") \
        else "results_fbb_ants"
    base = percorsi.dati(cart, sid, metodo, "voi")
    return (_carica(base / f"{metodo}_voi_ctx.nii.gz"),
            _carica(base / f"{metodo}_voi_wc.nii.gz"))


# ── 1. Lo stesso cervello visto dai sette metodi ──────────────────────────────
def fig_sette_metodi():
    """Lo stesso cervello visto dai sette metodi: in alto i tre che leggono il
    target dalle proprie etichette, in basso i quattro che lo ricavano
    dall'atlante. Figura fig_seg_seven_methods (capitolo 3)."""
    mri = _carica(percorsi.dati("aligned_ants", SOGGETTO, "Realigned_Anatomical",
                                "Realigned_Anatomical.nii.gz"))
    voi = {m: _voi_metodo(m) for m in METODI}
    # fetta coronale al centro del cervelletto di SynthSeg
    y = int(np.argmax((voi["synthseg"][1] > 0).sum(axis=(0, 2))))

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 5.2))
    # Lista piatta: MRI, poi (target, cervelletto) per ogni metodo, cosi'
    # _ritaglia usa lo stesso riquadro per tutti; il metodo i sta in
    # posizione 1 + 2i e 2 + 2i.
    fette = [_coronale(mri, y)] + [_coronale(v, y) for m in METODI for v in voi[m]]
    fette = _ritaglia(*fette)
    mri_s = _finestra(fette[0])
    ax0 = axes.flat[0]
    ax0.imshow(mri_s, cmap="gray", interpolation="nearest")
    ax0.set_xticks([]); ax0.set_yticks([])
    for sp in ax0.spines.values():
        sp.set_visible(False)
    ax0.set_title("T1 MRI", color=INK)
    for i, m in enumerate(METODI):
        ax = axes.flat[i + 1]
        _sovrapponi(ax, mri_s, fette[1 + 2 * i], fette[2 + 2 * i])
        ax.set_title(ETICHETTA[m], color=COL[FAM[m]], fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=0.4, h_pad=1.6)
    _legenda_voi(fig)
    fig.savefig(USCITA / "fig_seg_seven_methods.pdf")
    plt.close(fig)
    return y


def fig_metodi_etichette():
    """I tre metodi che leggono il target dalle proprie etichette. I quattro
    che lo ricavano dall'atlante hanno una figura a parte (fig_atlante), che
    mostra il passaggio dal template al soggetto. Figura fig_seg_label_methods
    (capitolo 3)."""
    metodi = ["synthseg", "fastsurfer", "samseg"]
    sottotitoli = {"synthseg": "38 DKT cortical labels",
                   "fastsurfer": "38 DKT cortical labels",
                   "samseg": "whole-hemisphere cortex"}
    mri = _carica(percorsi.dati("aligned_ants", SOGGETTO, "Realigned_Anatomical",
                                "Realigned_Anatomical.nii.gz"))
    voi = {m: _voi_metodo(m) for m in metodi}
    # fetta coronale al centro del cervelletto di SynthSeg
    y = int(np.argmax((voi["synthseg"][1] > 0).sum(axis=(0, 2))))

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.55))
    fette = [_coronale(mri, y)] + [_coronale(v, y) for m in metodi for v in voi[m]]
    fette = _ritaglia(*fette)
    mri_s = _finestra(fette[0])
    ax0 = axes[0]
    ax0.imshow(mri_s, cmap="gray", interpolation="nearest")
    ax0.set_xticks([]); ax0.set_yticks([])
    for sp in ax0.spines.values():
        sp.set_visible(False)
    ax0.set_title("T1 MRI", color=INK, pad=12)
    for i, m in enumerate(metodi):
        ax = axes[i + 1]
        _sovrapponi(ax, mri_s, fette[1 + 2 * i], fette[2 + 2 * i])
        ax.set_title(ETICHETTA[m], color=COL[FAM[m]], fontweight="bold", pad=12)
        ax.text(0.5, 1.015, sottotitoli[m], transform=ax.transAxes, ha="center",
                va="bottom", fontsize=7.8, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97), w_pad=0.5)
    _legenda_voi(fig)
    fig.savefig(USCITA / "fig_seg_label_methods.pdf", bbox_inches="tight")
    plt.close(fig)
    return y


# ── 1b. SAMSEG e samseg_atlas: stessa segmentazione, due target ──────────────
def _assiale(vol, z):
    """Fetta assiale pronta per imshow, anteriore in alto, sinistra del
    soggetto a sinistra come nelle fette coronali."""
    return np.flipud(vol[:, :, z].T)


def fig_samseg_atlas():
    """fig_samseg_vs_atlas (capitolo 3): stessa segmentazione SAMSEG, due target.

    Due righe (assiale, coronale) per due colonne (samseg, samseg_atlas), con
    SUVr e numero di voxel del target nel sottotitolo.
    """
    mri = _carica(percorsi.dati("aligned_ants", SOGGETTO, "Realigned_Anatomical",
                                "Realigned_Anatomical.nii.gz"))
    voi = {m: _voi_metodo(m) for m in ("samseg", "samseg_atlas")}
    ris = {m: pd.read_csv(percorsi.dati("results_fbb_ants", SOGGETTO, m,
                                        "result.csv")).iloc[0]
           for m in voi}
    ctx = voi["samseg"][0] > 0
    zz = np.where(ctx.any(axis=(0, 1)))[0]
    z = int(zz.min() + 0.70 * (zz.max() - zz.min()))    # sopra i ventricoli
    y = int(np.argmax((voi["samseg"][1] > 0).sum(axis=(0, 2))))

    fig, axes = plt.subplots(2, 2, figsize=(5.6, 5.9))
    viste = (("axial", lambda v: _assiale(v, z)), ("coronal", lambda v: _coronale(v, y)))
    for r, (nome_vista, taglia) in enumerate(viste):
        # Nell'assiale non c'e' collo da escludere: basta poco margine sopra.
        fette = _ritaglia(taglia(mri), *[taglia(v) for m in voi for v in voi[m]],
                          sopra=12 if nome_vista == "axial" else 28)
        mri_s = _finestra(fette[0])
        for c, m in enumerate(voi):
            ax = axes[r, c]
            _sovrapponi(ax, mri_s, fette[1 + 2 * c], fette[2 + 2 * c])
            if r == 0:
                d = ris[m]
                ax.set_title(ETICHETTA[m], color=COL[FAM[m]], fontweight="bold",
                             pad=14)
                ax.text(0.5, 1.012, f"SUVr {d.suvr:.2f}, target "
                        f"{int(d.ctx_voxels)/1000:.0f}k voxels", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=7.8, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.6, w_pad=0.6)
    _legenda_voi(fig)
    fig.savefig(USCITA / "fig_samseg_vs_atlas.pdf", bbox_inches="tight")
    plt.close(fig)


# ── 2. Dall'atlante al soggetto ───────────────────────────────────────────────
def _atlante_nativo(sid=SOGGETTO):
    """VOI standard portate sulla MRI del soggetto, con cache su disco."""
    import SimpleITK as sitk
    cache = USCITA / f"cache_atlante_{sid}"
    f_ctx, f_wc = cache / "voi_ctx_nativa.nii.gz", cache / "voi_wc_nativa.nii.gz"
    if not (f_ctx.exists() and f_wc.exists()):
        _atl = importlib.import_module("05_VOIExtraction.atlas_voi")
        mri = sitk.ReadImage(str(percorsi.dati(
            "aligned_ants", sid, "Realigned_Anatomical",
            "Realigned_Anatomical.nii.gz")), sitk.sitkFloat32)
        print("  registrazione SyN del template sulla MRI (qualche minuto) ...")
        _, ctx, wc, _ = _atl.warp_atlas_to_native(mri, _atl.config_predefinita())
        cache.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(ctx, str(f_ctx))
        sitk.WriteImage(wc, str(f_wc))
    return _carica(f_ctx), _carica(f_wc)


def fig_atlante():
    """fig_atlas_transfer (capitolo 3): VOI standard, deformate, intersecate.

    Tre pannelli: VOI sul template ICBM152, le stesse portate sul soggetto con
    SyN, e le VOI finali di Otsu (deformate e intersecate con il suo tessuto).
    """
    tpl = _carica(percorsi.TEMPLATE_ICBM)
    t_ctx, t_wc = _carica(percorsi.VOI_CTX_STD), _carica(percorsi.VOI_WC_STD)
    mri = _carica(percorsi.dati("aligned_ants", SOGGETTO, "Realigned_Anatomical",
                                "Realigned_Anatomical.nii.gz"))
    n_ctx, n_wc = _atlante_nativo()
    o_ctx, o_wc = _voi_metodo("otsu")
    # stesso livello anatomico nelle due immagini: la fetta in cui la VOI del
    # cervelletto ha l'area massima, sul template e sul soggetto (dove la VOI
    # e' quella del template deformata)
    y_tpl = int(np.argmax((t_wc > 0).sum(axis=(0, 2))))
    y_sogg = int(np.argmax((n_wc > 0).sum(axis=(0, 2))))

    a = _ritaglia(_coronale(tpl, y_tpl), _coronale(t_ctx, y_tpl),
                  _coronale(t_wc, y_tpl))
    # le VOI di SynthSeg entrano solo nel ritaglio, cosi' l'inquadratura e'
    # la stessa della figura dei metodi con etichette
    s_ctx, s_wc = _voi_metodo("synthseg")
    b = _ritaglia(_coronale(mri, y_sogg), _coronale(n_ctx, y_sogg),
                  _coronale(n_wc, y_sogg), _coronale(o_ctx, y_sogg),
                  _coronale(o_wc, y_sogg), _coronale(s_ctx, y_sogg),
                  _coronale(s_wc, y_sogg))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    _sovrapponi(axes[0], _finestra(a[0]), a[1], a[2])
    axes[0].set_title("standard VOIs on ICBM152", color=INK)
    _sovrapponi(axes[1], _finestra(b[0]), b[1], b[2])
    axes[1].set_title("warped onto the subject (SyN)", color=INK)
    _sovrapponi(axes[2], _finestra(b[0]), b[3], b[4])
    axes[2].set_title("intersected with Otsu tissue", color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97), w_pad=0.6)
    _legenda_voi(fig)
    fig.savefig(USCITA / "fig_atlas_transfer.pdf")
    plt.close(fig)


# ── 3. SimpleITK: variabilita' con il seme ────────────────────────────────────
def fig_semi():
    """fig_registration_seeds (capitolo 3): mutua informazione per seme.

    Una colonna per soggetto, ordinate per ampiezza della dispersione: punti
    grigi i dieci semi, punto blu il migliore, quello che il multi-restart
    tiene.
    """
    s = pd.read_csv(percorsi.dati("registration_stability.csv"))
    prove = [c for c in s.columns if c.startswith("trial_")]
    mi = -s[prove].to_numpy()          # metrica di Mattes: piu' negativa = meglio
    ordine = np.argsort(mi.max(axis=1) - mi.min(axis=1))
    mi = mi[ordine]
    x = np.arange(len(mi))
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    for xi, r in zip(x, mi):
        ax.plot([xi, xi], [r.min(), r.max()], color=GRID, linewidth=1.6, zorder=1,
                solid_capstyle="round")
    ax.scatter(np.repeat(x, mi.shape[1]), mi.ravel(), s=7, color=INK2,
               alpha=0.55, linewidth=0, zorder=2)
    ax.scatter(x, mi.max(axis=1), s=14, color=COL["apprendimento"], zorder=3,
               linewidth=0)
    rng = mi.max(axis=1) - mi.min(axis=1)
    ax.text(0.015, 0.97, f"range over the 10 seeds: median {np.median(rng):.2f}, "
            f"largest {rng.max():.2f}", transform=ax.transAxes, ha="left",
            va="top", fontsize=8.2, color=INK2)
    ax.set_xlim(-1, len(mi))
    ax.set_xticks([])
    ax.set_xlabel(f"{len(mi)} subjects, ordered by the spread across seeds")
    ax.set_ylabel("mutual information\n(− Mattes metric)")
    _pulisci(ax, "y")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=INK2, alpha=0.6,
                               markersize=3.5, label="one seed"),
                        Line2D([], [], marker="o", ls="", markersize=4.5,
                               color=COL["apprendimento"],
                               label="best of ten (kept by the multi-restart)")],
               loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0), labelcolor=INK)
    fig.savefig(USCITA / "fig_registration_seeds.pdf")
    plt.close(fig)


# ── 4. Florbetapir: perche' 14 soggetti sono esclusi ──────────────────────────
def fig_esclusione_fbp():
    """fig_res_fbp_exclusion (capitolo 4): soggetti florbetapir esclusi.

    A sinistra il SUVr di SynthSeg ordinato, con la soglia SUVR_MIN_FBP del
    modulo 07; a destra lo stesso SUVr contro la media corticale grezza.
    """
    r = _cmp.raccogli_risultati([percorsi.dati("results_fbp_ants")])
    r = r[r.segmenter == "synthseg"].copy()
    r["subject_id"] = r.subject_id.astype(str)
    vuoti = r[r.suvr.isna()].subject_id.tolist()
    r = r.dropna(subset=["suvr"]).sort_values("suvr").reset_index(drop=True)
    soglia = _cmp.SUVR_MIN_FBP
    ok = r.suvr >= soglia
    col = np.where(ok, COL["apprendimento"], "#9a9893")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    # Banda grigia fra il SUVr piu' alto degli esclusi e il piu' basso degli
    # ammessi: mostra che la soglia cade in un intervallo vuoto.
    lo, hi = r.suvr[~ok].max(), r.suvr[ok].min()
    ax.axhspan(lo, hi, color="#f1f0ec", zorder=0)
    ax.axhline(soglia, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.scatter(np.arange(len(r)), r.suvr, s=18, color=col, zorder=3,
               edgecolor="white", linewidth=0.5)
    ax.text(0.03, 0.96, f"dashed line: threshold {soglia:.2f}\n"
            f"shaded band: no subject between {lo:.3f} and {hi:.3f}",
            transform=ax.transAxes, fontsize=7.8, color=INK2,
            va="top", ha="left", linespacing=1.4)
    ax.set_xticks([])
    ax.set_xlabel(f"{len(r)} subjects with a result, by increasing SUVr")
    ax.set_ylabel("SUVr (SynthSeg)")
    _pulisci(ax, "y")

    ax = axes[1]
    ax.scatter(r.ctx_mean, r.suvr, s=18, color=col, zorder=3,
               edgecolor="white", linewidth=0.5)
    ax.axhline(soglia, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    # Scala log: le medie grezze (non normalizzate) variano molto da un
    # soggetto all'altro.
    ax.set_xscale("log")
    ax.set_xlabel("mean cortical uptake (raw units, log scale)")
    _pulisci(ax, "both")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles=[Patch(facecolor=COL["apprendimento"],
                              label=f"admitted ({int(ok.sum())})"),
                        Patch(facecolor="#9a9893",
                              label=f"excluded ({int((~ok).sum())}"
                                    + (f" + {len(vuoti)} without result)"
                                       if vuoti else ")"))],
               loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0), labelcolor=INK)
    fig.savefig(USCITA / "fig_res_fbp_exclusion.pdf")
    plt.close(fig)


# ── 5. Centiloide stimato contro riferimento, SynthSeg ────────────────────────
def fig_calibrazione_dispersione():
    """fig_res_calibration_scatter (capitolo 4): CL di SynthSeg contro riferimento.

    Cerchi vuoti: formula pubblicata; pieni: retta ricalibrata. Nell'angolo
    il MAE prima e dopo, dal foglio Riepilogo dell'Excel finale.
    """
    finale = D / "confronto_metodi_centiloide_ricalibrato_ANTs_finale.xlsx"
    rie = pd.read_excel(finale, sheet_name="Riepilogo").dropna(subset=["Metodo"])
    s = rie[rie.Metodo == "synthseg"].set_index(["Dataset", "Calibrazione"])["MAE (CL)"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    for ax, ds in zip(axes, ["FBB", "FMM", "FBP"]):
        x = pd.read_excel(finale, sheet_name=ds)
        v = x[["ground_truth", "synthseg\nraw", "synthseg\nreg"]].dropna()
        rif = v.ground_truth.values
        tutti = np.r_[rif, v["synthseg\nraw"], v["synthseg\nreg"]]
        # Assi uguali e quadrati attorno alla diagonale; il limite basso arriva
        # almeno a -25 CL perche' si vedano i negativi sotto lo zero.
        lim = (min(tutti.min(), -20) - 5, tutti.max() + 8)
        ax.plot(lim, lim, color=GRID, linewidth=1.0, zorder=1)
        ax.scatter(rif, v["synthseg\nraw"], s=15, facecolor="white",
                   edgecolor="#8a8984", linewidth=0.8, zorder=2,
                   label="published formula" if ds == "FBB" else None)
        ax.scatter(rif, v["synthseg\nreg"], s=15, color=COL["apprendimento"],
                   edgecolor="white", linewidth=0.4, zorder=3,
                   label="recalibrated (regression)" if ds == "FBB" else None)
        mae_r, mae_c = s[(ds, "raw")], s[(ds, "regressione")]
        ax.text(0.04, 0.96, f"MAE {mae_r:.2f} → {mae_c:.2f} CL",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.8,
                color=INK2)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_title(f"{TRACCIANTE[ds]} (n = {len(v)})", color=INK)
        ax.set_xlabel("reference CL" if ds == "FMM" else "")
        _pulisci(ax, "both")
    axes[0].set_ylabel("SynthSeg CL")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0), labelcolor=INK)
    fig.savefig(USCITA / "fig_res_calibration_scatter.pdf")
    plt.close(fig)


# ── 6. Dilatazione di un voxel ────────────────────────────────────────────────
def _mae_dilatazione() -> pd.DataFrame:
    """MAE senza e con dilatazione, con la stessa logica di
    scripts/analisi/analisi_dilatazione.py (una retta per ciascuna versione;
    sul DLBS rette del florbetapir GAAIN, 219 soggetti)."""
    _ad = importlib.import_module("scripts.analisi.analisi_dilatazione")
    dati = {ds: _ad.leggi(ds, percorsi.QAMYLOID_DLBS, _cmp.SUVR_MIN_FBP)
            for ds in ("FBB", "FMM", "FBP", "DLBS")}
    righe = []
    for ds, d in dati.items():
        for m in METODI:
            dm = d[d.method == m].dropna(subset=["suvr_dil0", "suvr_dil1"]).copy()
            # Sul DLBS non si stima una retta propria: si usano quelle del
            # florbetapir GAAIN (senza e con dilatazione) e si tolgono i
            # soggetti esclusi al controllo qualita'.
            if ds == "DLBS":
                fbp = dati["FBP"][dati["FBP"].method == m].dropna(
                    subset=["suvr_dil0", "suvr_dil1"])
                c0 = _cmp.calibrazione_regressione(fbp.suvr_dil0, fbp.truth_cl)
                c1 = _cmp.calibrazione_regressione(fbp.suvr_dil1, fbp.truth_cl)
                dm = dm[~dm.subject_id.isin(_cmp.DLBS_QC_ESCLUSI)]
                cl0 = _ad.ricalibra(dm, "suvr_dil0", c0)
                cl1 = _ad.ricalibra(dm, "suvr_dil1", c1)
            else:
                cl0 = _ad.ricalibra(dm, "suvr_dil0")
                cl1 = _ad.ricalibra(dm, "suvr_dil1")
            righe.append({"ds": ds, "metodo": m, "n": len(dm),
                          "mae0": np.nanmean(np.abs(cl0 - dm.truth_cl)),
                          "mae1": np.nanmean(np.abs(cl1 - dm.truth_cl))})
    return pd.DataFrame(righe)


def fig_dilatazione():
    """fig_res_dilation (capitolo 4): MAE senza e con la dilatazione di un voxel.

    Una freccia per metodo, dal MAE originale (cerchio vuoto) a quello con
    target dilatato; la differenza e' scritta accanto. Restituisce la tabella.
    """
    t = _mae_dilatazione()
    titoli = {**TRACCIANTE, "DLBS": "DLBS"}
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.0), sharey=True)
    y = np.arange(len(METODI))[::-1]
    for ax, ds in zip(axes, ["FBB", "FMM", "FBP", "DLBS"]):
        d = t[t.ds == ds].set_index("metodo").loc[METODI]
        for yi, a, b, m in zip(y, d.mae0, d.mae1, METODI):
            ax.annotate("", xy=(b, yi), xytext=(a, yi),
                        arrowprops=dict(arrowstyle="-|>", color=COL[FAM[m]],
                                        lw=1.1, mutation_scale=7,
                                        shrinkA=2.5, shrinkB=2.5), zorder=2)
            ax.scatter([a], [yi], s=20, facecolor="white", edgecolor=INK2,
                       linewidth=0.9, zorder=3)
            ax.scatter([b], [yi], s=22, color=COL[FAM[m]], zorder=4,
                       edgecolor="white", linewidth=0.5)
            ax.text(max(a, b) + 0.35, yi, f"{b - a:+.2f}", va="center",
                    fontsize=7.4, color=INK)
        # Spazio a destra (+2.4 CL) per il numero della differenza.
        hi = float(np.r_[d.mae0, d.mae1].max())
        ax.set_xlim(max(0, float(np.r_[d.mae0, d.mae1].min()) - 1), hi + 2.4)
        n = int(d.n.max())
        ax.set_title(f"{titoli[ds]} ({n})", color=INK)
        ax.set_xlabel("MAE (CL)")
        _pulisci(ax)
    axes[0].set_yticks(y); axes[0].set_yticklabels([ETICHETTA[m] for m in METODI])
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.6)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor="white",
                               markeredgecolor=INK2, label="target as segmented"),
                        Line2D([], [], marker="o", ls="", color=INK2,
                               label="dilated by one voxel (colour by family)")],
               loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0), labelcolor=INK)
    fig.savefig(USCITA / "fig_res_dilation.pdf")
    plt.close(fig)
    return t


# ── 7. Che cosa aggiunge la dilatazione ───────────────────────────────────────
def fig_dilatazione_maschera():
    """Ingrandimento di un tratto di corteccia: target originale e anello
    aggiunto dalla dilatazione di un voxel, con la stessa funzione usata
    nell'esperimento (05_VOIExtraction.dilata_target, dentro il cervello del
    metodo). Figura fig_dilation_mask (capitolo 4); restituisce la crescita
    percentuale del target per metodo."""
    import SimpleITK as sitk
    _voi = importlib.import_module("05_VOIExtraction")
    metodi = ["synthseg", "fastsurfer", "samseg", "otsu"]
    seg_file = {"synthseg": ("results_fbb_ants", "synthseg"),
                "fastsurfer": ("results_fbb_ants", "fastsurfer"),
                "samseg": ("results_fbb_ants", "samseg"),
                "otsu": ("results_fbb_atlas", "otsu")}
    leggi = lambda p: sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
    t1 = leggi(percorsi.dati("aligned_ants", SOGGETTO, "Realigned_Anatomical",
                             "Realigned_Anatomical.nii.gz")).astype(np.float32)
    dati = {}
    for m in metodi:
        cart, nome = seg_file[m]
        base = percorsi.dati(cart, SOGGETTO, m)
        ctx = leggi(base / "voi" / f"{m}_voi_ctx.nii.gz") > 0
        seg = leggi(base / "segmentation" / f"{nome}_seg.nii.gz")
        dil = _voi.dilata_target(ctx, seg, 1) > 0
        # (target, anello aggiunto, crescita in % dei voxel)
        dati[m] = (ctx, dil & ~ctx, 100 * (dil.sum() / ctx.sum() - 1))

    # fetta assiale all'altezza dei ventricoli; riquadro di 64 mm dove il target di
    # Otsu (il piu' rado, perche' viene dalla VOI standard) e quello di
    # SynthSeg sono insieme piu' densi, cosi' tutti e quattro hanno qualcosa
    # da mostrare
    from scipy import ndimage
    zz = np.where(dati["samseg"][0].any(axis=(1, 2)))[0]
    z = int(zz.min() + 0.60 * (zz.max() - zz.min()))
    sl = lambda v: np.flipud(v[z])
    lato = 64
    # Media mobile su finestre di lato x lato: il massimo indica il centro del
    # riquadro con piu' voxel di target.
    densita = ndimage.uniform_filter(
        (sl(dati["otsu"][0]).astype(float) + sl(dati["synthseg"][0])), size=lato,
        mode="constant")
    rc, cc = np.unravel_index(np.argmax(densita), densita.shape)
    r0, c0 = max(rc - lato // 2, 0), max(cc - lato // 2, 0)
    box = (slice(r0, r0 + lato), slice(c0, c0 + lato))

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35))
    base_t1 = sl(t1)[box]
    a, b = np.percentile(base_t1[base_t1 > 0], [1, 99.5])
    base_t1 = np.clip((base_t1 - a) / (b - a), 0, 1)
    GIALLO = "#f2c12e"
    for ax, m in zip(axes, metodi):
        ctx, anello, cresc = dati[m]
        ax.imshow(base_t1, cmap="gray", interpolation="nearest")
        for msk, col in ((sl(ctx)[box], ROSSO), (sl(anello)[box], GIALLO)):
            ax.imshow(np.ma.masked_where(~msk, msk), cmap=ListedColormap([col]),
                      alpha=0.8, interpolation="nearest", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(ETICHETTA[m], color=COL[FAM[m]], fontweight="bold", pad=12)
        ax.text(0.5, 1.015, f"target +{cresc:.0f}% voxels", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7.8, color=INK2)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97), w_pad=0.5)
    fig.legend(handles=[Patch(facecolor=ROSSO, label="target as segmented"),
                        Patch(facecolor=GIALLO, label="added by the one-voxel dilation")],
               loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0), labelcolor=INK)
    fig.savefig(USCITA / "fig_dilation_mask.pdf", bbox_inches="tight")
    plt.close(fig)
    return {m: round(v[2], 1) for m, v in dati.items()}


# ── 8. FreeSurfer, SynthSeg e FastSurfer sullo stesso soggetto DLBS ───────────
SOGGETTO_DLBS = "sub-1023"      # Dice vicino alla mediana per entrambe le reti


def fig_confronto_freesurfer(sid=SOGGETTO_DLBS):
    """Tre viste con il mirino, nello stile del controllo visivo di recon-all:
    target e cervelletto letti dalle etichette di FreeSurfer, SynthSeg e
    FastSurfer sullo stesso soggetto DLBS. Figura fig_freesurfer_comparison
    (capitolo 4), una riga per metodo e una colonna per vista."""
    metodi = [("recon_all", "FreeSurfer recon-all"), ("synthseg", "SynthSeg"),
              ("fastsurfer", "FastSurfer")]
    mri = _carica(percorsi.dati("aligned_dlbs_ants", sid, "Realigned_Anatomical",
                                "Realigned_Anatomical.nii.gz"))
    voi = {m: tuple(_carica(percorsi.dati("results_dlbs", sid, m, "voi",
                                          f"{m}_voi_{k}.nii.gz")) for k in ("ctx", "wc"))
           for m, _ in metodi}
    dice = pd.read_csv(D / "dice_per_soggetto.csv")
    dice = dice[dice.soggetto == sid].set_index("metodo")

    # mirino: centro del cervelletto di FreeSurfer in x e y, ventricoli in z
    rif_ctx, rif_wc = voi["recon_all"]
    cx = int(round(np.where(rif_wc > 0)[0].mean()))
    cy = int(round(np.where(rif_wc > 0)[1].mean()))
    zz = np.where(rif_ctx.any(axis=(0, 1)))[0]
    cz = int(zz.min() + 0.45 * (zz.max() - zz.min()))
    viste = [("coronal", lambda v: np.flipud(v[:, cy, :].T)),
             # sagittale spostata di 18 voxel dal centro del cervelletto, che
             # sta sulla linea mediana, per tagliare un emisfero
             ("sagittal", lambda v: np.flipud(v[cx + 18, :, :].T)),
             ("axial", lambda v: np.flipud(v[:, :, cz].T))]

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.4))
    for c, (vista, taglia) in enumerate(viste):
        fette = _ritaglia(taglia(mri), *[taglia(v) for m, _ in metodi for v in voi[m]],
                          sopra=30)
        mri_s = _finestra(fette[0])
        for r, (m, nome) in enumerate(metodi):
            ax = axes[r, c]
            _sovrapponi(ax, mri_s, fette[1 + 2 * r], fette[2 + 2 * r], alpha=0.7)
            if r == 0:
                ax.set_title(vista, color=INK)
            if c == 0:
                etichetta = nome
                if m in dice.index:
                    d = dice.loc[m]
                    etichetta += (f"\nDice: cortex {d.dice_ctx:.2f}, "
                                  f"cerebellum {d.dice_wc:.2f}")
                else:
                    # recon-all non ha righe in dice_per_soggetto: e' il riferimento.
                    etichetta += "\n(reference)"
                ax.set_ylabel(etichetta, color=INK, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.5, w_pad=0.3)
    _legenda_voi(fig)
    fig.savefig(USCITA / "fig_freesurfer_comparison.pdf")
    plt.close(fig)


# Nome usato da --solo -> funzione che disegna la figura.
FIGURE = {
    "sette_metodi": fig_sette_metodi,
    "metodi_etichette": fig_metodi_etichette,
    "samseg_atlas": fig_samseg_atlas,
    "atlante": fig_atlante,
    "semi": fig_semi,
    "esclusione_fbp": fig_esclusione_fbp,
    "calibrazione": fig_calibrazione_dispersione,
    "dilatazione": fig_dilatazione,
    "dilatazione_maschera": fig_dilatazione_maschera,
    "confronto_freesurfer": fig_confronto_freesurfer,
}


def main() -> int:
    """Disegna le figure scelte con --solo (predefinito: tutte).

    Le funzioni che restituiscono una tabella (fig_dilatazione) la stampano.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo", nargs="*", choices=list(FIGURE),
                    help="genera solo le figure indicate")
    a = ap.parse_args()
    USCITA.mkdir(parents=True, exist_ok=True)
    for nome in (a.solo or FIGURE):
        print(f"{nome} ...")
        out = FIGURE[nome]()
        if isinstance(out, pd.DataFrame):
            print(out.round(2).to_string(index=False))
    print(f"\nScritte in {USCITA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
