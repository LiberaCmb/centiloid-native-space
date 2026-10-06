"""Pipeline su tutti i soggetti di un manifest, con uno o piu' metodi.

Fase 1  ogni soggetto viene allineato una sola volta (moduli 01-03) e le
        immagini allineate vanno in una cache;
Fase 2  per ogni coppia (soggetto, metodo) si eseguono i moduli 04-06
        riusando la cache, cosi' tutti i metodi leggono la stessa PET
        allineata e le differenze sono dovute solo alla segmentazione.

Comandi usati per i risultati della tesi (FBB; per FMM, FBP e DLBS cambiano
manifest e cartelle):

    # metodi a etichette FreeSurfer + samseg_atlas
    python -m scripts.batch_process --manifest data/manifest_fbb_nifti.csv \\
        --segmenters synthseg fastsurfer samseg samseg_atlas \\
        --cache-dir data/aligned_ants --output data/results_fbb_ants --skip-done

    # metodi classici (VOI dall'atlante)
    python -m scripts.batch_process --manifest data/manifest_fbb_nifti.csv \\
        --segmenters otsu kmeans fsl_fast \\
        --cache-dir data/aligned_ants --output data/results_fbb_atlas --skip-done

Colonne del manifest: subject_id, dataset, tracer, mri_path, pet_path
(facoltativa co_acquired, 0/1). Un soggetto che fallisce non ferma il lotto:
finisce in realign_failures.csv o segmentation_failures.csv. Con --skip-done
un lotto interrotto riparte da dove si era fermato.

Nota: `samseg_atlas` qui rilancia SAMSEG. Se la segmentazione SAMSEG esiste
gia', e' molto piu' rapido `scripts/esecuzione/recompute_samseg_atlas.py`, che
la riusa (ed e' come sono stati prodotti i risultati della tesi).
"""

from __future__ import annotations
import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path

import pandas as pd

# Radice del progetto nel path, per poter lanciare lo script anche senza -m.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import setup_logger, ProjectContext                      # noqa: E402
from shared.percorsi import risolvi                                  # noqa: E402
from shared.pipeline import METODI, esegui_soggetto, scrivi_risultato  # noqa: E402

_load    = importlib.import_module("01_LoadImages")
_realign = importlib.import_module("02_Realignment")
_visual  = importlib.import_module("03_VisualCheck")


def parse_args():
    """Opzioni da riga di comando (vedi la docstring del modulo)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--segmenters", required=True, nargs="+", choices=METODI)
    p.add_argument("--output", required=True,
                   help="Cartella dei risultati: <output>/<soggetto>/<metodo>/.")
    p.add_argument("--cache-dir", default=None,
                   help="Cache delle immagini allineate (predefinita: "
                        "<output>/_realigned_cache).")
    p.add_argument("--registration", choices=["ants", "sitk"], default="ants",
                   help="Motore di allineamento della fase 1.")
    p.add_argument("--force-realign", action="store_true",
                   help="Rifa' l'allineamento anche se e' gia' in cache.")
    p.add_argument("--qc", action="store_true",
                   help="Salva anche la figura di controllo dell'allineamento "
                        "nella cache di ogni soggetto.")
    p.add_argument("--skip-done", action="store_true",
                   help="Salta le coppie (soggetto, metodo) che hanno gia' "
                        "un result.csv.")
    p.add_argument("--ctx-dilation", type=int, default=0)
    p.add_argument("--atlas-transform", default="SyN")
    p.add_argument("--seg-kwargs", default="{}",
                   help="Opzioni JSON passate a tutti i segmentatori; ognuno "
                        "usa solo quelle che riconosce.")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# Fase 1 — allineamento, una volta per soggetto
# ════════════════════════════════════════════════════════════════════════════
def allinea_soggetto(row, cache_root: Path, log, force: bool,
                     registration: str, qc: bool):
    """Moduli 01-03 su un soggetto. Restituisce (mri, pet) allineate o None."""
    subj = str(row["subject_id"])
    out = cache_root / subj
    anat_path = out / "Realigned_Anatomical" / "Realigned_Anatomical.nii.gz"
    pet_path = out / "Realigned_Functional" / "Realigned_Functional.nii.gz"

    # Riuso della cache: gli stessi due file che la fase 2 leggera'.
    if not force and anat_path.exists() and pet_path.exists():
        log.info("    [in cache] %s", subj)
        return str(anat_path), str(pet_path)
    out.mkdir(parents=True, exist_ok=True)

    # Contesto del soggetto, come lo riempirebbe la pipeline interattiva.
    ctx = ProjectContext()
    ctx.subject_id = subj
    ctx.dataset_name = str(row.get("dataset", ""))
    ctx.is_co_acquired = int(row.get("co_acquired", 0)) == 1
    ctx.output_dir = str(out)
    ctx.selected_tracer = str(row["tracer"])

    for slot, col in (("anat_1", "mri_path"), ("func_1", "pet_path")):
        ok, msg = _load.update_context_with_file(ctx, slot, str(risolvi(row[col])))
        if not ok:
            log.error("    caricamento di %s non riuscito: %s", col, msg)
            return None

    transform, msg = _realign.get_registration(registration)(ctx)
    if transform is None:
        log.error("    allineamento non riuscito: %s", msg)
        return None
    log.info("    %s", msg)
    _visual.ImageFusionLogic.save_realigned_images_with_transform(
        ctx, str(out), transform)
    if qc:
        try:
            _visual.figura_allineamento(anat_path, pet_path,
                                        out / "qc_allineamento.png", titolo=subj)
        except Exception as e:
            log.warning("    figura di controllo non generata: %s", e)
    return str(anat_path), str(pet_path)


# ════════════════════════════════════════════════════════════════════════════
# Fase 2 — moduli 04-06 per ogni metodo
# ════════════════════════════════════════════════════════════════════════════
def segmenta(row, metodo, anat, pet, out_root: Path, args, log) -> bool:
    """Moduli 04-06 su una coppia (soggetto, metodo), partendo dalla cache.

    Scrive <out_root>/<soggetto>/<metodo>/result.csv. Restituisce False se
    qualcosa va storto (l'errore e' gia' stampato), senza sollevare eccezioni.
    """
    subj = str(row["subject_id"])
    cartella = out_root / subj / metodo
    if args.skip_done and (cartella / "result.csv").exists():
        log.info("    [gia' fatto] %s x %s", subj, metodo)
        return True
    try:
        ris = esegui_soggetto(
            anat, pet, str(row["tracer"]), metodo, str(cartella),
            riusa_allineamento=True,
            co_acquisite=int(row.get("co_acquired", 0)) == 1,
            dilatazione_ctx=args.ctx_dilation,
            seg_kwargs=json.loads(args.seg_kwargs),
            id_soggetto=subj, dataset=str(row.get("dataset", "")),
            trasformazione_atlante=args.atlas_transform,
            progresso=lambda fase, frazione, msg: log.info("    %s", msg))
        scrivi_risultato(cartella, ris.record, ris.info)
        log.info("    SUVr=%.4f  CL=%.2f", ris.record["suvr"], ris.record["centiloid"])
        return True
    except Exception as e:
        log.error("    NON RIUSCITO: %s x %s — %s", subj, metodo, e)
        traceback.print_exc()
        return False


def main():
    """Lotto completo: fase 1 su tutti i soggetti, poi fase 2."""
    log = setup_logger("batch")
    args = parse_args()

    # 1. Manifest: si tengono solo le righe con entrambe le immagini.
    manifest = pd.read_csv(args.manifest)
    valid = manifest.dropna(subset=["mri_path", "pet_path"]).reset_index(drop=True)
    if len(valid) < len(manifest):
        log.warning("Salto %d righe del manifest senza MRI o PET.",
                    len(manifest) - len(valid))

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_dir) if args.cache_dir else out_root / "_realigned_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    # 2. Fase 1: allineamento, un soggetto alla volta.
    log.info("=== Fase 1: allineamento (cache in %s) ===", cache_root)
    allineati, falliti = {}, []
    for i, row in valid.iterrows():
        log.info("[%d/%d] allineamento %s", i + 1, len(valid), row["subject_id"])
        r = allinea_soggetto(row, cache_root, log, args.force_realign,
                             args.registration, args.qc)
        if r is None:
            falliti.append({"subject_id": row["subject_id"],
                            "error": "allineamento non riuscito"})
        else:
            allineati[str(row["subject_id"])] = r
    if falliti:
        pd.DataFrame(falliti).to_csv(out_root / "realign_failures.csv", index=False)
        log.warning("%d allineamenti non riusciti: realign_failures.csv", len(falliti))

    # 3. Fase 2: solo per i soggetti allineati, tutti i metodi richiesti.
    log.info("=== Fase 2: segmentazione, VOI, Centiloide ===")
    seg_falliti = []
    totale = len(allineati) * len(args.segmenters)
    n = 0
    for subj, (anat, pet) in allineati.items():
        row = valid[valid["subject_id"].astype(str) == subj].iloc[0]
        for metodo in args.segmenters:
            n += 1
            log.info("[%d/%d] %s x %s", n, totale, subj, metodo)
            if not segmenta(row, metodo, anat, pet, out_root, args, log):
                seg_falliti.append({"subject_id": subj, "segmenter": metodo})
    if seg_falliti:
        pd.DataFrame(seg_falliti).to_csv(out_root / "segmentation_failures.csv",
                                         index=False)
        log.warning("%d coppie (soggetto, metodo) non riuscite: "
                    "segmentation_failures.csv", len(seg_falliti))
    log.info("Fatto. Risultati in %s", out_root)


if __name__ == "__main__":
    main()
