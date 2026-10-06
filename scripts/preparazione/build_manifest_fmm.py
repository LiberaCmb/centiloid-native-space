"""Manifest dell'archivio flutemetamol (FMM) del Centiloid Project (GE Healthcare).

Archivio: "Set di dati PIBflutemetamolo da 50-70 minuti (dati NIFTI e DICOM)".
74 soggetti, 50 pazienti (AD) e 24 volontari giovani sani (YHV), con MRI T1 e
PET [18F]flutemetamol 90-110 minuti gia' in NIfTI:
    GE_AD_<id>_MRI_NIFTI.nii      GE_F18_FMT_AD_<id>_NIFTI.nii
    GE_YHV_<id>_NIFTI.nii         GE_F18_FMT_YHV_<id>_NIFTI.nii
Identificativi: 'AD_100', 'YHV_001', ...

Uscita: data/manifest_fmm.csv. La conversione (convert_dicom_to_nifti.py) li
riorienta in RAS e produce data/manifest_fmm_nifti.csv, usato nei risultati.

    python -m scripts.preparazione.build_manifest_fmm
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

ARCHIVIO = "Set di dati PIBflutemetamolo da 50-70 minuti (dati NIFTI e DICOM)"
# (gruppo, cartella MRI, prefisso MRI, suffisso MRI, cartella FMM, prefisso FMM)
GRUPPI = [
    ("AD", "GE_AD_MRI_NIFTI/GE_AD_MRI_NIFTI", "GE_AD_", "_MRI_NIFTI.nii",
     "GE_AD_F18_NIFTI/GE_AD_F18_NIFTI", "GE_F18_FMT_AD_"),
    ("YHV", "GE_YHV_MRI_NIFTI/GE_YHV_MRI_NIFTI", "GE_YHV_", "_NIFTI.nii",
     "GE_YHV_F18_NIFTI/GE_YHV_F18_NIFTI", "GE_F18_FMT_YHV_"),
]
CAMPI = ["subject_id", "dataset", "tracer", "mri_path", "pet_path", "co_acquired"]


def main():
    """Accoppia i NIfTI di MRI e PET per numero di soggetto e scrive il manifest."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-root", default=str(percorsi.DATASETS / ARCHIVIO))
    ap.add_argument("--out", default=str(percorsi.dati("manifest_fmm.csv")))
    a = ap.parse_args()

    src = Path(a.src_root)
    if not src.exists():
        sys.exit(f"Archivio non trovato: {src}")

    righe, mancanti = [], []
    for gruppo, mri_dir, mri_pref, mri_suf, fmm_dir, fmm_pref in GRUPPI:
        mri_root, fmm_root = src / mri_dir, src / fmm_dir
        if not mri_root.exists() or not fmm_root.exists():
            print(f"  attenzione: manca {mri_root} o {fmm_root}")
            continue
        # Si parte dalle MRI: il numero del soggetto, estratto dal nome,
        # individua il file PET corrispondente.
        for mri_file in sorted(mri_root.iterdir()):
            nome = mri_file.name
            if not (mri_file.is_file() and nome.startswith(mri_pref)
                    and nome.endswith(mri_suf)):
                continue
            num = nome[len(mri_pref):-len(mri_suf)]     # '100', '001'
            if not num.isdigit():
                continue
            fmm_file = fmm_root / f"{fmm_pref}{num}_NIFTI.nii"
            sid = f"{gruppo}_{num}"
            if not fmm_file.exists():
                mancanti.append((sid, str(fmm_file)))
                continue
            righe.append({"subject_id": sid, "dataset": "fmm_project",
                          "tracer": "FMM", "mri_path": str(mri_file),
                          "pet_path": str(fmm_file), "co_acquired": 0})

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
