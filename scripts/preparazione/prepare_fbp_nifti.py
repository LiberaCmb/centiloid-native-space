"""Archivio florbetapir (FBP) del Centiloid Project: da DICOM a NIfTI e manifest.

L'archivio ha la struttura di XNAT e non si carica come gli altri:
    <soggetto>_mr/scans/<id>/DICOM/*.dcm        la MRI T1 (una serie)
    <soggetto>/scans/<frameA>/DICOM/*.dcm       PET 50-55 minuti
    <soggetto>/scans/<frameB>/DICOM/*.dcm       PET 55-60 minuti
I due frame di cinque minuti vanno MEDIATI per ottenere l'immagine statica
50-60 minuti prevista dal protocollo.

Per ogni soggetto lo script
  * prende la serie DICOM della MRI con piu' fette,
  * legge tutte le serie PET e ne fa la media (ricampionando sulla griglia del
    primo frame se serve),
  * riorienta in RAS e scrive <out>/<soggetto>/MR.nii.gz e PET.nii.gz,
  * scrive il manifest data/manifest_fbp_nifti.csv, con la colonna `group`
    (YC / elderly): gli identificativi FBP sono numerici e solo da qui si sa
    chi e' controllo giovane, cosa che serve alla calibrazione two-point.

    python -m scripts.preparazione.prepare_fbp_nifti
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared import percorsi                        # noqa: E402

ARCHIVIO = "50-60 minute florbetapir and 50-70 minute PiB dataset DICOM data"
# (gruppo, cartella delle MRI, cartella delle PET florbetapir)
GRUPPI = [
    ("elderly", "Elder_subject_MRI/Elder_subject_MRI",
     "Florbetapir_Elder_subject_florbetapir/Elder_subject_florbetapir"),
    ("YC", "Young_Control_MRI/Young_Control_MRI",
     "Florbetapir_Young_Control_florbetapir/Young_Control_florbetapir"),
]


def _orient_ras(img):
    """Riorienta l'immagine in RAS, come fa il modulo 00 per gli altri archivi."""
    f = sitk.DICOMOrientImageFilter()
    f.SetDesiredCoordinateOrientation("RAS")
    return f.Execute(img)


def _read_series(dicom_dir):
    """Legge la serie DICOM con piu' file presente nella cartella."""
    r = sitk.ImageSeriesReader()
    ids = r.GetGDCMSeriesIDs(dicom_dir)
    if not ids:
        raise RuntimeError(f"nessuna serie DICOM in {dicom_dir}")
    best = max(ids, key=lambda s: len(r.GetGDCMSeriesFileNames(dicom_dir, s)))
    r.SetFileNames(r.GetGDCMSeriesFileNames(dicom_dir, best))
    # Avvisi di ITK silenziati solo durante la lettura, poi riattivati.
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
    img = r.Execute()
    sitk.ProcessObject.SetGlobalWarningDisplay(True)
    return img


def _cartelle_dicom(radice):
    """Le cartelle scans/*/DICOM che contengono davvero dei .dcm."""
    out = []
    for d in sorted(glob.glob(os.path.join(radice, "scans", "*", "DICOM"))):
        if glob.glob(os.path.join(d, "*.dcm")) or glob.glob(os.path.join(d, "*.DCM")):
            out.append(d)
    return out


def _carica_mri(radice):
    """MRI in RAS dalla cartella DICOM con piu' file, e numero di file letti."""
    dirs = _cartelle_dicom(radice)
    if not dirs:
        raise RuntimeError(f"nessuna MRI DICOM in {radice}")
    conta = {d: len([f for f in os.listdir(d) if f.lower().endswith(".dcm")]) for d in dirs}
    best = max(dirs, key=lambda d: conta[d])           # la T1 ha piu' fette
    return _orient_ras(_read_series(best)), conta[best]


def _carica_pet_media(radice):
    """Media voxel per voxel dei frame PET, in RAS, e numero di frame mediati."""
    dirs = _cartelle_dicom(radice)
    if not dirs:
        raise RuntimeError(f"nessuna PET DICOM in {radice}")
    imgs = [sitk.Cast(_read_series(d), sitk.sitkFloat32) for d in dirs]
    ref = imgs[0]
    arrs = [sitk.GetArrayFromImage(ref)]
    for im in imgs[1:]:
        if im.GetSize() != ref.GetSize():
            rs = sitk.ResampleImageFilter()
            rs.SetReferenceImage(ref)
            # Interpolazione lineare: e' un'immagine di intensita', non una
            # maschera.
            rs.SetInterpolator(sitk.sitkLinear)
            im = rs.Execute(im)
        arrs.append(sitk.GetArrayFromImage(im))
    media = sitk.GetImageFromArray(np.mean(arrs, axis=0).astype(np.float32))
    media.CopyInformation(ref)
    return _orient_ras(media), len(dirs)


def main():
    """Converte tutti i soggetti FBP e scrive il manifest con la colonna group."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-root", default=str(percorsi.DATASETS / ARCHIVIO))
    ap.add_argument("--out-root", default="data/nifti_fbp")
    ap.add_argument("--out-manifest", default="data/manifest_fbp_nifti.csv")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.src_root)
    out_root = percorsi.risolvi(args.out_root)
    # 1. Elenco dei soggetti: MRI '<id>_mr' accoppiata alla cartella PET il cui
    #    nome inizia con '<id>_'.
    da_fare = []
    for gruppo, mri_sub, pet_sub in GRUPPI:
        if not (src / mri_sub).exists():
            continue
        pet_per_id = {}
        for p in (src / pet_sub).iterdir():
            if p.is_dir() and not p.name.startswith("."):
                pet_per_id.setdefault(p.name.split("_")[0], p.name)
        for mr in sorted(p for p in (src / mri_sub).iterdir() if p.is_dir()):
            sid = mr.name[:-3] if mr.name.endswith("_mr") else mr.name
            if sid in pet_per_id:
                da_fare.append((sid, gruppo, str(mr), str(src / pet_sub / pet_per_id[sid])))
    if args.limit:
        da_fare = da_fare[:args.limit]
    print(f"{len(da_fare)} soggetti FBP -> {out_root}", flush=True)

    # 2. Conversione; i NIfTI gia' presenti si riusano. Un soggetto che fallisce
    #    non ferma gli altri.
    righe, falliti = [], []
    for i, (sid, gruppo, mr_root, pet_root) in enumerate(da_fare, 1):
        sd = out_root / sid
        sd.mkdir(parents=True, exist_ok=True)
        mr_out, pet_out = sd / "MR.nii.gz", sd / "PET.nii.gz"
        try:
            if not mr_out.exists():
                mr, n_mr = _carica_mri(mr_root)
                sitk.WriteImage(mr, str(mr_out))
            else:
                n_mr = "in cache"
            if not pet_out.exists():
                pet, n_fr = _carica_pet_media(pet_root)
                sitk.WriteImage(pet, str(pet_out))
            else:
                n_fr = "in cache"
            righe.append(dict(subject_id=sid, dataset="fbp_project", tracer="FBP",
                              mri_path=f"{args.out_root}/{sid}/MR.nii.gz",
                              pet_path=f"{args.out_root}/{sid}/PET.nii.gz",
                              co_acquired=0, group=gruppo))
            print(f"[{i}/{len(da_fare)}] {sid} ok (fette MRI={n_mr}, frame PET mediati={n_fr})",
                  flush=True)
        except Exception as e:
            falliti.append((sid, str(e)))
            print(f"[{i}/{len(da_fare)}] {sid} NON RIUSCITO: {e}", flush=True)

    # 3. Manifest: i percorsi delle immagini usano --out-root cosi' come e'
    #    stato scritto (relativo al progetto nel caso predefinito).
    man = percorsi.risolvi(args.out_manifest)
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "dataset", "tracer", "mri_path",
                                          "pet_path", "co_acquired", "group"])
        w.writeheader()
        w.writerows(righe)
    print(f"\nScritto {man}: {len(righe)} soggetti, {len(falliti)} non riusciti")
    for s, e in falliti:
        print("  NON RIUSCITO", s, e)


if __name__ == "__main__":
    main()
