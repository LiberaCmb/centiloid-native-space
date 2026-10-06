"""Manifest del Dallas Lifespan Brain Study (DLBS), l'archivio di validazione.

Formato BIDS, 226 soggetti, tracciante florbetapir (FBP). Ogni soggetto ha una
sola sessione (ses-1 o ses-2) con
    sub-XXXX/ses-Y/anat/*T1w.nii.gz              MRI MPRAGE
    sub-XXXX/ses-Y/pet/*trc-fbp*_pet.nii.gz      PET statica gia' sommata
(le acquisizioni funzionali `func/` non servono).

Uscita: data/manifest_dlbs.csv con le colonne
    subject_id, dataset, tracer, mri_path, pet_path, co_acquired, session

    python -m scripts.preparazione.build_manifest_dlbs
    python -m scripts.preparazione.build_manifest_dlbs --scan-root /altro/DLBS
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


def _trova(cartella: Path, schema: str) -> Path | None:
    """Primo file (in ordine alfabetico) che corrisponde allo schema, o None."""
    trovati = sorted(cartella.glob(schema))
    return trovati[0] if trovati else None


def main():
    """Scorre le cartelle sub-* e scrive il manifest; elenca i soggetti saltati."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan-root", default=str(percorsi.DLBS),
                   help="Cartella del DLBS (predefinita: la cartella DLBS accanto "
                        "al progetto, o la variabile CENTILOID_DLBS).")
    ap.add_argument("--output", default=str(percorsi.dati("manifest_dlbs.csv")))
    a = ap.parse_args()

    root = Path(a.scan_root)
    if not root.exists():
        sys.exit(f"Cartella non trovata: {root}")

    righe, mancanti = [], []
    for subj_dir in sorted(p for p in root.glob("sub-*") if p.is_dir()):
        sid = subj_dir.name
        sessioni = sorted(subj_dir.glob("ses-*"))
        if not sessioni:
            mancanti.append((sid, "nessuna sessione"))
            continue
        # Ogni soggetto ha una sola sessione: si prende la prima trovata.
        ses = sessioni[0]
        t1 = _trova(ses / "anat", "*T1w.nii.gz")
        pet = _trova(ses / "pet", "*trc-fbp*_pet.nii.gz")
        if t1 is None or pet is None:
            mancanti.append((sid, "manca la T1" if t1 is None else "manca la PET FBP"))
            continue
        righe.append({"subject_id": sid, "dataset": "dlbs", "tracer": "FBP",
                      "mri_path": t1.as_posix(), "pet_path": pet.as_posix(),
                      "co_acquired": 0, "session": ses.name})

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "dataset", "tracer",
                                          "mri_path", "pet_path", "co_acquired",
                                          "session"])
        w.writeheader()
        w.writerows(righe)
    print(f"Scritti {len(righe)} soggetti in {out}")
    for sid, motivo in mancanti:
        print(f"  saltato {sid}: {motivo}")


if __name__ == "__main__":
    main()
