"""Manifest dell'archivio florbetaben (FBB) del Centiloid Project.

Archivio: "set di dati PIB da 50-70 minuti e FBB da 90-110 minuti (dati DICOM)".
Per ogni soggetto (25 anziani E-25, 10 controlli giovani YC-10) ci sono la
cartella DICOM della MRI (<id>_MR) e quella della PET FBB 90-110 minuti
(<id>_PET_FBB). Le cartelle DICOM delle PET contengono una sottocartella con
lo stesso nome (effetto dell'estrazione degli zip).

Uscita: data/manifest_fbb.csv con le colonne
    subject_id, dataset, tracer, mri_path, pet_path, co_acquired
Il passo successivo e' la conversione in NIfTI (convert_dicom_to_nifti.py),
che produce data/manifest_fbb_nifti.csv, il manifest usato nei risultati.

    python -m scripts.preparazione.build_manifest_fbb
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared import percorsi                        # noqa: E402

ARCHIVIO = "set di dati PIB da 50-70 minuti e FBB da 90-110 minuti (dati DICOM)"
# (cartella MRI, cartella PET FBB) per ciascun gruppo
GRUPPI = [
    ("FBBproject_E-25_MR", "FBBproject_E-25_FBB_90110"),
    ("FBBproject_YC-10_MR", "FBBproject_YC-10_FBB_90110"),
]
DATASET_TAG = "fbb_project"
CAMPI = ["subject_id", "dataset", "tracer", "mri_path", "pet_path", "co_acquired"]


def main():
    """Accoppia le cartelle MRI e PET di ogni soggetto e scrive il manifest."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-root", default=str(percorsi.DATASETS / ARCHIVIO))
    ap.add_argument("--out", default=str(percorsi.dati("manifest_fbb.csv")))
    a = ap.parse_args()

    src = Path(a.src_root)
    if not src.exists():
        sys.exit(f"Archivio non trovato: {src}")

    righe, mancanti = [], []
    for mr_dir, fbb_dir in GRUPPI:
        mr_root = src / mr_dir
        # Cartella ripetuta due volte: effetto dell'estrazione degli zip.
        fbb_root = src / fbb_dir / fbb_dir
        if not mr_root.exists() or not fbb_root.exists():
            print(f"  attenzione: manca {mr_root} o {fbb_root}")
            continue
        for mr_sub in sorted(p for p in mr_root.iterdir() if p.is_dir()):
            # '<id>_MR' -> '<id>', poi si cerca la PET '<id>_PET_FBB'.
            sid = mr_sub.name.replace("_MR", "")
            fbb_sub = fbb_root / f"{sid}_PET_FBB"
            if not fbb_sub.exists():
                mancanti.append((sid, str(fbb_sub)))
                continue
            righe.append({"subject_id": sid, "dataset": DATASET_TAG, "tracer": "FBB",
                          "mri_path": str(mr_sub), "pet_path": str(fbb_sub),
                          "co_acquired": 0})

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPI)
        w.writeheader()
        w.writerows(righe)
    print(f"Scritte {len(righe)} righe in {out}")
    for sid, p in mancanti:
        print(f"  manca la PET di {sid}: {p}")


if __name__ == "__main__":
    main()
