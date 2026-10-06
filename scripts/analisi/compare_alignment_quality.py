"""Qualita' dell'allineamento: ANTs contro SimpleITK, soggetto per soggetto.

Non esiste una trasformazione "vera" con cui confrontarsi, quindi ogni
allineamento si valuta con due indici calcolati allo stesso modo per i due
motori (stessa MRI, stessa PET, cambia solo la trasformazione):

  NMI  informazione mutua normalizzata (H(MR) + H(PET)) / H(MR, PET) sui voxel
       del cervello. Piu' alta = migliore.
  GC   correlazione dei gradienti: correlazione di Pearson fra i moduli dei
       gradienti di MRI e PET nel cervello. Premia i bordi allineati
       indipendentemente dal rapporto fra le intensita'. Piu' alta = migliore.

Ingressi (gia' su disco):
  ANTs        data/aligned_ants/<soggetto>/Realigned_{Anatomical,Functional}/
  SimpleITK   data/results_fbb/_realigned_cache/<soggetto>/...   (FBB)
              data/results_fmm/_realigned_cache/<soggetto>/...   (FMM)

Uscita: data/comparison_all_methods/alignment_quality.csv, scritta un
soggetto alla volta (rilanciando si riprende da dove ci si era fermati).
La sintesi con il confronto sul centiloide e' in `confronto_motori.py`.

    python -m scripts.analisi.compare_alignment_quality
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared import percorsi                       # noqa: E402

ANTS_ROOT = percorsi.dati("aligned_ants")
SITK_ROOTS = {
    "FBB": percorsi.dati("results_fbb", "_realigned_cache"),
    "FMM": percorsi.dati("results_fmm", "_realigned_cache"),
}
FIELDS = ["subject_id", "tracer", "nmi_ants", "nmi_sitk", "gc_ants", "gc_sitk",
          "winner_nmi", "winner_gc"]


def _load(p):
    """Volume NIfTI in float32; di un 4D si tiene il primo frame."""
    a = np.asarray(nib.load(str(p)).get_fdata(), dtype=np.float32)
    return a[..., 0] if a.ndim == 4 else a


def _nmi(mr, pet, bins=64):
    """(H(MR) + H(PET)) / H(MR, PET) sui voxel sopra il 30° percentile della MRI."""
    # Il 30° percentile dei voxel non nulli della MRI fa da maschera grezza del
    # cervello: esclude sfondo e gran parte del liquor senza una segmentazione.
    mask = mr > np.percentile(mr[mr > 0], 30)
    h, _, _ = np.histogram2d(mr[mask].ravel(), pet[mask].ravel(), bins=bins)
    # Istogramma congiunto -> probabilita' congiunte e marginali.
    pxy = h / h.sum()
    px, py = pxy.sum(1), pxy.sum(0)

    def H(p):
        p = p[p > 0]
        return -(p * np.log(p)).sum()
    hxy = H(pxy.ravel())
    return float((H(px) + H(py)) / hxy) if hxy > 0 else float("nan")


def _grad_corr(mr, pet):
    """Correlazione fra |grad MRI| e |grad PET| nel cervello."""
    mask = mr > np.percentile(mr[mr > 0], 30)
    gm = np.sqrt(sum(g ** 2 for g in np.gradient(mr)))
    gp = np.sqrt(sum(g ** 2 for g in np.gradient(pet)))
    x, y = gm[mask].ravel(), gp[mask].ravel()
    # Gradiente costante (immagine vuota o piatta): correlazione non definita.
    if x.std() < 1e-8 or y.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _coppia(root: Path, sid: str):
    """Percorsi (MRI, PET) allineate di un soggetto sotto `root`."""
    return (root / sid / "Realigned_Anatomical" / "Realigned_Anatomical.nii.gz",
            root / sid / "Realigned_Functional" / "Realigned_Functional.nii.gz")


def valuta_soggetto(sid):
    """NMI e GC dei due motori per un soggetto; None se mancano dati.

    Il tracciante si deduce dalla cartella della cache SimpleITK in cui il
    soggetto compare. Restituisce una riga per il CSV (campi di FIELDS).
    """
    tracer = next((tr for tr, r in SITK_ROOTS.items() if (r / sid).is_dir()), None)
    if tracer is None:
        print(f"  [salto] {sid}: manca la cache SimpleITK")
        return None
    a_mr, a_pet = _coppia(ANTS_ROOT, sid)
    s_mr, s_pet = _coppia(SITK_ROOTS[tracer], sid)
    if not all(p.exists() for p in (a_mr, a_pet, s_mr, s_pet)):
        print(f"  [salto] {sid}: manca un file")
        return None

    mr = _load(a_mr)                      # la MRI e' la stessa per i due motori
    pet_a, pet_s = _load(a_pet), _load(s_pet)
    # Gli indici si calcolano voxel per voxel: le tre immagini devono
    # condividere la griglia (quella della MRI).
    if not (mr.shape == pet_a.shape == pet_s.shape):
        print(f"  [salto] {sid}: griglie diverse")
        return None

    nmi_a, nmi_s = _nmi(mr, pet_a), _nmi(mr, pet_s)
    gc_a, gc_s = _grad_corr(mr, pet_a), _grad_corr(mr, pet_s)
    print(f"  {sid} {tracer}: NMI ants={nmi_a:.4f} sitk={nmi_s:.4f}  "
          f"GC ants={gc_a:.3f} sitk={gc_s:.3f}", flush=True)
    return {"subject_id": sid, "tracer": tracer,
            "nmi_ants": round(nmi_a, 5), "nmi_sitk": round(nmi_s, 5),
            "gc_ants": round(gc_a, 5), "gc_sitk": round(gc_s, 5),
            "winner_nmi": "ants" if nmi_a > nmi_s else "sitk",
            "winner_gc": "ants" if gc_a > gc_s else "sitk"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(percorsi.CONFRONTO / "alignment_quality.csv"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)

    # 1. Ripresa: i soggetti gia' presenti nel CSV di uscita non si rifanno.
    fatti = set()
    if out.exists():
        with open(out) as f:
            fatti = {r["subject_id"] for r in csv.DictReader(f)}
    soggetti = sorted(p.name for p in ANTS_ROOT.iterdir()
                      if p.is_dir() and not p.name.startswith("_"))
    da_fare = [s for s in soggetti if s not in fatti]
    if args.limit:
        da_fare = da_fare[:args.limit]
    print(f"{len(fatti)} gia' fatti, {len(da_fare)} da fare.")

    # 2. Scrittura in coda, una riga per soggetto con flush immediato, cosi' un
    #    arresto a meta' non perde il lavoro fatto.
    out.parent.mkdir(parents=True, exist_ok=True)
    nuovo = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if nuovo:
            w.writeheader()
        for sid in da_fare:
            row = valuta_soggetto(sid)
            if row:
                w.writerow(row)
                f.flush()
    print(f"Aggiornato {out}")


if __name__ == "__main__":
    main()
