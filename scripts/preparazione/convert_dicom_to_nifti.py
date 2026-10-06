"""Modulo 00 su tutto un manifest: ogni MRI e PET convertita in NIfTI uniforme.

Legge un manifest (quelli di build_manifest_*.py), converte MRI e PET di ogni
soggetto in .nii.gz riorientati in RAS e scrive un NUOVO manifest che punta ai
file convertiti; quello originale non viene toccato.

    data/nifti/<dataset>/<soggetto>/MR.nii.gz
    data/nifti/<dataset>/<soggetto>/PET.nii.gz
    data/nifti/<dataset>/<soggetto>/PET.json        (metadati di dcm2niix, se c'e')
    data/<nome_manifest>_nifti.csv                  (percorsi relativi al progetto)

    python -m scripts.preparazione.convert_dicom_to_nifti --manifest data/manifest_fbb.csv
    python -m scripts.preparazione.convert_dicom_to_nifti --manifest data/manifest_fmm.csv
    # forzando il lettore di SimpleITK invece di dcm2niix:
    python -m scripts.preparazione.convert_dicom_to_nifti --manifest data/manifest_fbb.csv --engine sitk

Si puo' rilanciare: i soggetti gia' convertiti si saltano (salvo --force).
"""
from __future__ import annotations
import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import setup_logger                 # noqa: E402
from shared.percorsi import risolvi             # noqa: E402

_conv = importlib.import_module("00_DicomToNifti")


def parse_args():
    """Opzioni da riga di comando (--limit N: solo i primi N soggetti, per prova)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-root", default="data/nifti")
    p.add_argument("--out-manifest", default=None,
                   help="Predefinito: data/<nome_manifest>_nifti.csv")
    p.add_argument("--engine", choices=["auto", "dcm2niix", "sitk"], default="auto",
                   help="'auto' = dcm2niix se installato, altrimenti SimpleITK.")
    p.add_argument("--dcm2niix-path", default=None)
    p.add_argument("--force", action="store_true",
                   help="Riconverte anche i file gia' esistenti.")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def _converti(converter, src, dst: Path, force, log):
    """Restituisce (percorso, motore) oppure (None, 'NON RIUSCITO: ...')."""
    if dst.exists() and not force:
        return dst, "in cache"
    try:
        return dst, converter.convert(str(src), str(dst))["engine"]
    except Exception as e:                                 # noqa: BLE001
        log.error("    conversione di %s non riuscita: %s", src, e)
        return None, f"NON RIUSCITO: {e}"


def main():
    """Converte ogni soggetto del manifest e scrive il nuovo manifest NIfTI.

    Oltre al manifest scrive <out_root>/conversion_log_<manifest>.csv con il
    motore usato e l'esito per ciascun soggetto.
    """
    log = setup_logger("conversione")
    args = parse_args()

    manifest = pd.read_csv(risolvi(args.manifest))
    if args.limit:
        manifest = manifest.head(args.limit)
    out_root = risolvi(args.out_root)

    # 1. Convertitore: dcm2niix se disponibile (salvo --engine sitk).
    prefer = args.engine != "sitk"
    converter = _conv.DicomToNiftiConverter(dcm2niix_path=args.dcm2niix_path,
                                            prefer_dcm2niix=prefer)
    motore = (("dcm2niix" if converter.dcm2niix_bin else "simpleitk-fallback")
              if prefer else "simpleitk (forzato)")
    log.info("%d soggetti da %s | motore=%s | -> %s",
             len(manifest), args.manifest, motore, out_root)

    # 2. Conversione soggetto per soggetto: nel nuovo manifest entrano solo i
    #    soggetti con MRI e PET entrambe riuscite; il registro li tiene tutti.
    righe, registro = [], []
    for i, (_, r) in enumerate(manifest.iterrows(), 1):
        sid = str(r["subject_id"])
        dataset = str(r.get("dataset", "dataset"))
        log.info("[%d/%d] %s (%s)", i, len(manifest), sid, dataset)
        cartella = out_root / dataset / sid
        mr, eng_mr = _converti(converter, r["mri_path"], cartella / "MR.nii.gz", args.force, log)
        pet, eng_pet = _converti(converter, r["pet_path"], cartella / "PET.nii.gz", args.force, log)
        stato = "ok" if (mr and pet) else "NON RIUSCITO"
        registro.append({"subject_id": sid, "dataset": dataset,
                         "engine_mr": eng_mr, "engine_pet": eng_pet, "status": stato})
        if stato != "ok":
            continue

        def _rel(p):                    # percorsi relativi al progetto, portabili
            try:
                return str(Path(p).resolve().relative_to(ROOT))
            except Exception:
                return str(p)
        righe.append({"subject_id": sid, "dataset": dataset, "tracer": r["tracer"],
                      "mri_path": _rel(mr), "pet_path": _rel(pet),
                      "co_acquired": int(r.get("co_acquired", 0))})

    # 3. Scrittura del nuovo manifest e del registro della conversione.
    stem = Path(args.manifest).stem
    out_manifest = (risolvi(args.out_manifest) if args.out_manifest
                    else risolvi(f"data/{stem}_nifti.csv"))
    pd.DataFrame(righe).to_csv(out_manifest, index=False)
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(registro).to_csv(out_root / f"conversion_log_{stem}.csv", index=False)

    n_ok = sum(c["status"] == "ok" for c in registro)
    log.info("Fatto: %d riusciti, %d non riusciti. Nuovo manifest: %s",
             n_ok, len(registro) - n_ok, out_manifest)


if __name__ == "__main__":
    main()
