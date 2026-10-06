"""Pipeline completa su un singolo soggetto.

Esempio (MRI e PET come cartelle DICOM, conversione automatica):

    python -m scripts.run_pipeline \\
        --mri    "/percorso/2002_MR" \\
        --pet    "/percorso/2002_PET_FBB" \\
        --tracer FBB --segmenter synthseg --convert auto \\
        --subject-id 2002 --dataset fbb_project \\
        --output data/prove/2002_synthseg

Nella cartella di uscita:
    Realigned_Anatomical/, Realigned_Functional/   immagini allineate (modulo 03)
    qc_allineamento.png                            PET sopra la MRI, da guardare
    segmentation/<metodo>_seg.nii.gz               segmentazione (modulo 04)
    voi/<metodo>_voi_ctx.nii.gz, ..._voi_wc.nii.gz regioni di interesse (05)
    result.csv, info.json                          SUVr e Centiloide (06)

Il `centiloid` di result.csv usa la formula pubblicata del tracciante. A fine
esecuzione viene stampato anche il centiloide ricalibrato (retta di regressione
della tesi per quel metodo e tracciante, da `resources/coefficienti/` o da
`data/comparison_all_methods/recalibration_final.csv`), che e' il valore da
usare: la formula pubblicata, applicata al SUVr in spazio nativo, sottostima.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# Radice del progetto nel path, per poter lanciare lo script anche senza -m.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd                                                # noqa: E402
from shared import percorsi, setup_logger                         # noqa: E402
from shared.pipeline import METODI, esegui_soggetto, scrivi_risultato  # noqa: E402


def parse_args(argv=None):
    """Opzioni da riga di comando (`argv` permette di chiamarlo dai test)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mri", required=True,
                   help="MRI T1: file NIfTI o cartella DICOM.")
    p.add_argument("--pet", required=True,
                   help="PET amiloide: file NIfTI o cartella DICOM.")
    p.add_argument("--tracer", required=True, choices=["FBB", "FMM", "FBP"])
    p.add_argument("--segmenter", required=True, choices=METODI)
    p.add_argument("--output", required=True, help="Cartella di uscita.")
    p.add_argument("--subject-id", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--registration", choices=["ants", "sitk"], default="ants",
                   help="Motore di allineamento PET->MRI. 'ants' (predefinito, "
                        "usato in tutta la tesi) oppure 'sitk' (SimpleITK con "
                        "multi-restart).")
    p.add_argument("--convert", choices=["off", "auto", "dcm2niix", "sitk"],
                   default="off",
                   help="Modulo 00 prima del caricamento. 'off' = carica gli "
                        "ingressi cosi' come sono (DICOM o NIfTI); 'auto' = "
                        "converte con dcm2niix se installato, altrimenti con "
                        "SimpleITK. I file convertiti vanno in <output>/converted.")
    p.add_argument("--co-acquired", action="store_true",
                   help="PET e MRI gia' co-registrate: niente allineamento.")
    p.add_argument("--reuse-realigned", action="store_true",
                   help="--mri e --pet sono gia' le immagini allineate del "
                        "modulo 03: salta i moduli 00-03.")
    p.add_argument("--ctx-dilation", type=int, default=0,
                   help="Voxel di dilatazione del target corticale (0 = "
                        "nessuna, come nei risultati della tesi).")
    p.add_argument("--atlas-transform", default="SyN",
                   help="Trasformazione ANTs template -> soggetto per i "
                        "metodi con atlante ('SyN'; 'Affine' per una prova veloce).")
    p.add_argument("--seg-kwargs", default="{}",
                   help="Opzioni JSON per il segmentatore, per esempio "
                        "'{\"samseg_binary\": \"/percorso/run_samseg\"}'.")
    p.add_argument("--no-qc", action="store_true",
                   help="Non salvare la figura di controllo dell'allineamento.")
    return p.parse_args(argv)


def main(argv=None):
    """Esegue la pipeline sul soggetto, scrive result.csv e stampa SUVr e CL.

    Restituisce il risultato di esegui_soggetto (record, info, avvisi).
    """
    log = setup_logger("pipeline")
    args = parse_args(argv)
    # 1. Pipeline completa (moduli 00-06) e scrittura dei risultati.
    ris = esegui_soggetto(
        args.mri, args.pet, args.tracer, args.segmenter, args.output,
        registrazione=args.registration,
        conversione=args.convert,
        co_acquisite=args.co_acquired,
        riusa_allineamento=args.reuse_realigned,
        dilatazione_ctx=args.ctx_dilation,
        seg_kwargs=json.loads(args.seg_kwargs),
        id_soggetto=args.subject_id,
        dataset=args.dataset,
        trasformazione_atlante=args.atlas_transform,
        figura_qc=not args.no_qc,
        progresso=lambda fase, frazione, msg: log.info(msg),
    )
    scrivi_risultato(args.output, ris.record, ris.info)
    r = ris.record
    log.info("=== soggetto %s, metodo %s, tracciante %s ===",
             r["subject_id"], r["segmenter"], r["tracer"])
    log.info("    SUVr = %.4f", r["suvr"])
    log.info("    CL   = %.2f  (formula pubblicata)", r["centiloid"])
    # 2. Centiloide ricalibrato, solo a schermo: i coefficienti sono stimati
    #    su VOI non dilatate, quindi con --ctx-dilation > 0 non si applicano.
    try:
        c = pd.read_csv(percorsi.COEFFICIENTI)
        c = c[(c.tracciante == args.tracer) & (c.metodo == args.segmenter)
              & (c.calibrazione == "regressione")]
        if len(c) and args.ctx_dilation == 0:
            a, b = float(c.a.iloc[0]), float(c.b.iloc[0])
            log.info("    CL   = %.2f  (ricalibrato, CL = %.2f*SUVr %+.2f)",
                     a * r["suvr"] + b, a, b)
    except Exception as e:                       # il ricalibrato e' un di piu'
        log.warning("    centiloide ricalibrato non calcolato: %s", e)
    for a in ris.avvisi:
        log.warning("    %s", a)
    log.info("Risultati in %s", args.output)
    return ris


if __name__ == "__main__":
    main()
