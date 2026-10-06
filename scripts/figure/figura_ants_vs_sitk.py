"""ANTs contro SimpleITK sullo stesso soggetto, per il capitolo 3.

Prende la PET allineata da ANTs (quella usata per tutti i risultati, in
data/aligned_ants) e quella allineata da SimpleITK nel confronto fra i due
motori (data/results_<tracciante>/_realigned_cache). Se quest'ultima manca,
rifa' l'allineamento SimpleITK (multi-restart con i dieci semi del codice) dai
NIfTI del manifest. Le due PET, divise ciascuna per la propria media nel
cervelletto, sono mostrate sopra la MRI con il contorno del target di
SynthSeg.

Il centiloide riportato e' quello finale di ciascun motore: SUVr sulle VOI di
SynthSeg e retta di regressione stimata sui soggetti del tracciante allineati
con quel motore (data/comparison_all_methods/comparison_ants_vs_sitk/
comparison_persubject.csv), cioe' la stessa calibrazione adottata nella tesi,
ottimizzata separatamente per ANTs e per SimpleITK.

Il soggetto predefinito, AD_105 (flutemetamol), e' quello su cui la
differenza fra i due motori pesa di piu' sul centiloide: e' un caso estremo,
non tipico (vedi la tabella ANTs/SimpleITK del capitolo 3).

Un allineamento SimpleITK rifatto viene salvato in
data/figure_tesi/cache_sitk_<id>/ e riusato alle esecuzioni successive.

Uso:  python -m scripts.figure.figura_ants_vs_sitk [--soggetto AD_105 --dataset FMM]
Uscita: data/figure_tesi/fig_ants_vs_sitk.pdf
"""
from __future__ import annotations
import argparse, importlib, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared import percorsi, ProjectContext          # noqa: E402
from shared.percorsi import risolvi                  # noqa: E402
from scripts.figure.figure_risultati import INK, INK2, TRACCIANTE   # noqa: E402

_cmp = importlib.import_module("07_Comparison")
USCITA = percorsi.dati("figure_tesi")
MANIFEST = {"FBB": "manifest_fbb_nifti.csv", "FMM": "manifest_fmm_nifti.csv"}
RISULTATI = {"FBB": "results_fbb_ants", "FMM": "results_fmm_ants"}
CONTORNO = "#2fd3e0"      # ciano: si distingue dalla scala "hot" della PET
SOGLIA_PET = 1.0          # sotto la media del cervelletto la PET non si colora


def _carica(p):
    """NIfTI come array float32, riorientato in RAS (assi x, y, z)."""
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(str(p)))
    return np.asarray(img.dataobj, dtype=np.float32)


def allinea_sitk(sid: str, ds: str) -> Path:
    """Cartella con la PET allineata da SimpleITK per il soggetto.

    sid       codice del soggetto
    ds        tracciante ('FBB' o 'FMM'), sceglie manifest e cache

    Cerca prima la cache del confronto fra motori, poi quella della figura;
    solo se mancano entrambe rifa' l'allineamento (moduli 01-03).
    """
    originale = percorsi.dati(f"results_{ds.lower()}", "_realigned_cache", sid)
    if (originale / "Realigned_Functional" / "Realigned_Functional.nii.gz").exists():
        return originale
    cache = USCITA / f"cache_sitk_{sid}"
    pet = cache / "Realigned_Functional" / "Realigned_Functional.nii.gz"
    if pet.exists():
        return cache
    # Contesto minimo come nella pipeline: soggetto, tracciante e i due NIfTI
    # del manifest, poi registrazione e scrittura delle immagini allineate.
    man = pd.read_csv(percorsi.dati(MANIFEST[ds]), dtype={"subject_id": str})
    riga = man[man.subject_id.str.strip() == sid].iloc[0]
    load = importlib.import_module("01_LoadImages")
    reg = importlib.import_module("02_Realignment")
    vis = importlib.import_module("03_VisualCheck")
    ctx = ProjectContext()
    ctx.subject_id, ctx.dataset_name = sid, str(riga.dataset)
    ctx.selected_tracer = ds
    ctx.is_co_acquired = int(riga.get("co_acquired", 0)) == 1
    for chiave, col in (("anat_1", "mri_path"), ("func_1", "pet_path")):
        ok, msg = load.update_context_with_file(ctx, chiave, str(risolvi(riga[col])))
        if not ok:
            raise RuntimeError(msg)
    print("  allineamento SimpleITK (dieci tentativi) ...")
    transform, esito = reg.get_registration("sitk")(ctx)
    if transform is None:
        raise RuntimeError(f"SimpleITK non riuscito: {esito}")
    print("  ", esito)
    cache.mkdir(parents=True, exist_ok=True)
    vis.ImageFusionLogic.save_realigned_images_with_transform(ctx, str(cache), transform)
    return cache


def rette_finali(ds: str) -> dict:
    """Retta di regressione di SynthSeg per ciascun motore, stimata su tutti i
    soggetti del tracciante come nel confronto ANTs/SimpleITK."""
    f = percorsi.CONFRONTO / "comparison_ants_vs_sitk" / "comparison_persubject.csv"
    d = pd.read_csv(f)
    d = d[(d.dataset == ds) & (d.method == "synthseg")]
    a0, b0 = _cmp.formule_pubblicate()[ds]
    rette = {}
    for motore, g in d.groupby("alignment"):
        suvr = (g.cl_raw - b0) / a0          # dal centiloide con la formula pubblicata
        rette[{"ants": "ANTs", "sitk": "SimpleITK"}[motore]] = \
            _cmp.calibrazione_regressione(suvr, g.truth_cl)
    return rette


def misura(pet, ctx, wc, a, b):
    """SUVr (target / cervelletto) e centiloide con la retta (a, b)."""
    suvr = float(pet[ctx].mean() / pet[wc].mean())
    return suvr, a * suvr + b


def disegna(sid: str, ds: str):
    """Figura fig_ants_vs_sitk (capitolo 3): una riga per motore, tre viste.

    sid       codice del soggetto
    ds        tracciante

    Restituisce ({motore: (suvr, cl)}, cl_riferimento).
    """
    # 1. Immagini: MRI e le due PET sulla stessa griglia, VOI di SynthSeg.
    cache = allinea_sitk(sid, ds)
    al = percorsi.dati("aligned_ants", sid)
    mri = _carica(al / "Realigned_Anatomical" / "Realigned_Anatomical.nii.gz")
    pets = {"ANTs": _carica(al / "Realigned_Functional" / "Realigned_Functional.nii.gz"),
            "SimpleITK": _carica(cache / "Realigned_Functional" /
                                 "Realigned_Functional.nii.gz")}
    voi = percorsi.dati(RISULTATI[ds], sid, "synthseg", "voi")
    ctx = _carica(voi / "synthseg_voi_ctx.nii.gz") > 0
    wc = _carica(voi / "synthseg_voi_wc.nii.gz") > 0
    for n, p in pets.items():
        if p.shape != mri.shape:
            raise RuntimeError(f"PET {n} su una griglia diversa dalla MRI: {p.shape}")

    # 2. Rette di calibrazione dei due motori e valori di riferimento
    #    (centiloide e SUVr) del Centiloid Project per il soggetto.
    rette = rette_finali(ds)
    gt = _cmp.leggi_ground_truth(percorsi.GROUND_TRUTH / _cmp.GROUND_TRUTH[ds])
    rif = float(gt.set_index("subject_id").loc[sid, "truth_cl"])
    _g = pd.read_csv(percorsi.GROUND_TRUTH / _cmp.GROUND_TRUTH[ds], dtype={"subject_id": str})
    rif_suvr = float(_g.set_index("subject_id").loc[sid, "truth_suvr"])

    # 3. Tre fette attorno al baricentro del target. L'assiale sale di 12
    #    voxel; la sagittale si sposta di 8 a lato, perche' il baricentro di
    #    un target bilaterale cade sulla linea mediana, fra i due emisferi.
    cx, cy, cz = [int(round(v)) for v in np.array(np.where(ctx)).mean(axis=1)]
    viste = [("axial", lambda v: np.flipud(v[:, :, cz + 12].T)),
             ("coronal", lambda v: np.flipud(v[:, cy, :].T)),
             ("sagittal", lambda v: np.flipud(v[cx + 8, :, :].T))]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.3))
    for r, (nome, pet) in enumerate(pets.items()):
        norm = pet / pet[wc].mean()
        suvr, cl = misura(pet, ctx, wc, *rette[nome])
        for k, (vista, taglia) in enumerate(viste):
            ax = axes[r, k]
            m, p, t = taglia(mri), taglia(norm), taglia(ctx.astype(float))
            # Ritaglio sulla testa: voxel della MRI sopra il 30-esimo
            # percentile (fuori resta aria e fondo), con 3 voxel di margine.
            rr, cc = np.where(taglia((mri > np.percentile(mri[mri > 0], 30))
                                     .astype(float)) > 0)
            sl = (slice(max(rr.min() - 3, 0), rr.max() + 3),
                  slice(max(cc.min() - 3, 0), cc.max() + 3))
            m, p, t = m[sl], p[sl], t[sl]
            a, b = np.percentile(m[m > 0], [1, 99.5])
            # MRI un po' scurita, cosi' la PET sopra si legge bene
            ax.imshow(0.9 * np.clip((m - a) / (b - a), 0, 1), cmap="gray",
                      vmin=0, vmax=1, interpolation="nearest")
            # PET mostrata solo dove supera la media del cervelletto; la
            # scala si ferma a 2.4 volte quella media, uguale per i due motori.
            im = ax.imshow(np.ma.masked_where(p < SOGLIA_PET, p), cmap="hot",
                           vmin=SOGLIA_PET, vmax=2.4, alpha=0.62,
                           interpolation="bilinear")
            ax.contour(t, levels=[0.5], colors=CONTORNO, linewidths=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(vista, color=INK)
        axes[r, 0].set_ylabel(f"{nome}\nSUVr {suvr:.2f}, {cl:.0f} CL",
                              color=INK, fontsize=9)
    fig.suptitle(f"subject {sid} ({TRACCIANTE[ds]}); Centiloid Project reference: "
                 f"SUVr {rif_suvr:.2f}, {rif:.0f} CL",
                 color=INK2, fontsize=9, y=0.995)
    # 4. Il 10% a destra resta alla barra dei colori, il 5% in basso alla
    #    legenda del contorno.
    fig.tight_layout(rect=(0, 0.05, 0.9, 0.97), h_pad=0.6, w_pad=0.4)
    cax = fig.add_axes([0.915, 0.22, 0.015, 0.55])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("PET / cerebellar mean", color=INK2, fontsize=8.5)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    cb.outline.set_visible(False)
    fig.legend(handles=[Line2D([], [], color=CONTORNO, label="SynthSeg cortical target")],
               loc="lower center", ncol=1, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.45, 0.0), labelcolor=INK)
    fig.savefig(USCITA / "fig_ants_vs_sitk.pdf")
    plt.close(fig)
    return {n: misura(p, ctx, wc, *rette[n]) for n, p in pets.items()}, rif


def main() -> int:
    """Disegna la figura e stampa SUVr e centiloide dei due motori."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--soggetto", default="AD_105")
    ap.add_argument("--dataset", default="FMM", choices=sorted(MANIFEST))
    a = ap.parse_args()
    USCITA.mkdir(parents=True, exist_ok=True)
    valori, rif = disegna(a.soggetto, a.dataset)
    for n, (s, c) in valori.items():
        print(f"{n}: SUVr {s:.3f}, CL {c:.1f}")
    print(f"riferimento {rif:.1f} CL\nScritta {USCITA / 'fig_ants_vs_sitk.pdf'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
