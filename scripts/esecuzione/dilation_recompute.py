"""Esperimento di dilatazione: il target corticale allargato di un voxel.

Per ogni metodo e ogni soggetto si prende la VOI corticale gia' salvata, la si
dilata di un voxel (`scipy.ndimage.binary_dilation`, restando dentro il
cervello, cioe' dentro i voxel non nulli della segmentazione), si lascia
invariato il cervelletto e si ricalcolano SUVr e Centiloide. E' la stessa
operazione di `--ctx-dilation 1` della pipeline, ma senza rifare
registrazione e segmentazione.

Uscita: data/comparison_all_methods/dilation_<DATASET>.csv, una riga per
(soggetto, metodo), con le colonne *_dil0 (valori originali) e *_dil1
(target dilatato). L'analisi e' in `scripts/analisi/analisi_dilatazione.py`.

    python -m scripts.esecuzione.dilation_recompute --dataset FBB
    (poi FMM, FBP, DLBS)
"""
from __future__ import annotations
import argparse, csv, glob, importlib, os, sys
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared import load_image                        # noqa: E402
from shared import percorsi                          # noqa: E402
_voi = importlib.import_module("05_VOIExtraction")
_cl = importlib.import_module("06_CentiloidCalculation")
_cmp = importlib.import_module("07_Comparison")


def configurazione(ds: str) -> dict:
    """{metodo: cartella dei risultati} per un archivio."""
    # I metodi classici hanno le VOI dall'atlante e stanno in una cartella a
    # parte; il DLBS ha invece un'unica cartella per tutti i metodi.
    cartelle = [percorsi.dati(c) for c in _cmp.CARTELLE_RISULTATI[ds]]
    if len(cartelle) == 1:                                     # DLBS
        return {m: cartelle[0] for m in _cmp.METODI}
    ants, atlante = cartelle
    return {m: (atlante if m in ("otsu", "kmeans", "fsl_fast") else ants)
            for m in _cmp.METODI}


def parse_args():
    """Opzioni da riga di comando (vedi la docstring del modulo)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(_cmp.CARTELLE_RISULTATI))
    p.add_argument("--iterazioni", type=int, default=1)
    p.add_argument("--out", default=None,
                   help="CSV di uscita (predefinito: data/comparison_all_methods/dilation_<DATASET>.csv).")
    return p.parse_args()


def dilata(voi_ctx_img, seg_img, iterazioni=1):
    """Target dilatato di `iterazioni` voxel, dentro il cervello."""
    ctx = sitk.GetArrayFromImage(voi_ctx_img) > 0
    # La segmentazione serve come limite del cervello: va portata sulla griglia
    # della VOI (vicino piu' prossimo, per non inventare etichette).
    if seg_img.GetSize() != voi_ctx_img.GetSize():
        rs = sitk.ResampleImageFilter()
        rs.SetReferenceImage(voi_ctx_img)
        rs.SetInterpolator(sitk.sitkNearestNeighbor)
        seg_img = rs.Execute(seg_img)
    grown = _voi.dilata_target(ctx, sitk.GetArrayFromImage(seg_img), iterazioni)
    out = sitk.GetImageFromArray(grown)
    out.CopyInformation(voi_ctx_img)
    return out, int(grown.sum())


def main():
    """Ricalcola SUVr e CL con il target dilatato per tutti i risultati."""
    a = parse_args()
    ds = a.dataset
    out = Path(a.out) if a.out else percorsi.CONFRONTO / f"dilation_{ds}.csv"
    righe = []
    # 1. Raccolta dei result.csv esistenti, raggruppati per soggetto.
    per_soggetto = {}                    # la PET si carica una volta per soggetto
    for m, d in configurazione(ds).items():
        for f in glob.glob(os.path.join(str(d), "*", m, "result.csv")):
            sub = Path(f).parts[-3]
            per_soggetto.setdefault(sub, []).append((m, f))
    # 2. Per ogni (soggetto, metodo): immagini salvate, dilatazione del target,
    #    nuovo SUVr/CL accanto ai valori originali letti dal CSV.
    for si, (sub, items) in enumerate(sorted(per_soggetto.items()), 1):
        pet = None
        for m, f in items:
            r = list(csv.DictReader(open(f)))[0]
            try:
                if pet is None:
                    pet = load_image(percorsi.risolvi(r["pet_path"]))
                seg = load_image(percorsi.risolvi(r["seg_path"]))
                vc = load_image(percorsi.risolvi(r["voi_ctx_path"]))
                vw = load_image(percorsi.risolvi(r["voi_wc_path"]))
            except Exception as e:
                print(f"[salto] {sub}/{m}: {e}")
                continue
            suvr0 = float(r.get("suvr") or "nan")
            cl0 = float(r.get("centiloid") or "nan")
            v0 = int(float(r.get("ctx_voxels") or 0))
            vc1, v1 = dilata(vc, seg, a.iterazioni)
            # Il cervelletto (vw) resta quello originale: cambia solo il target.
            _, _, suvr1, cl1, _ = _cl.compute_centiloid(pet, vc1, vw, str(r["tracer"]))
            righe.append(dict(subject_id=sub, method=m, tracer=str(r["tracer"]),
                              suvr_dil0=suvr0, cl_dil0=cl0, ctxvox_dil0=v0,
                              suvr_dil1=suvr1, cl_dil1=cl1, ctxvox_dil1=v1))
        print(f"[{si}/{len(per_soggetto)}] {sub}: {len(items)} metodi")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(righe).to_csv(out, index=False)
    print(f"\nScritto {out} ({len(righe)} righe soggetto-metodo)")


if __name__ == "__main__":
    main()
