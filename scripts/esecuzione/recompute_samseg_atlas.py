"""Metodo `samseg_atlas` a partire dalle segmentazioni SAMSEG gia' calcolate.

SAMSEG non parcellizza la corteccia (un'etichetta per emisfero, 3 e 42), quindi
il metodo `samseg` usa come target la corteccia intera. `samseg_atlas` porta
la stessa segmentazione sul target Centiloid standard:

    target      = VOI corticale standard (nello spazio nativo) ∩ corteccia SAMSEG {3, 42}
    riferimento = VOI del cervelletto      (nello spazio nativo) ∩ cervelletto SAMSEG {7, 8, 46, 47}

Lo script riusa la segmentazione e la PET allineata salvate in
<risultati>/<soggetto>/samseg/: non rilancia SAMSEG, e il solo passo costoso e'
la registrazione del template sul soggetto (qualche minuto). Il risultato va
in <risultati>/<soggetto>/samseg_atlas/. Per questo i tempi di segmentazione
di samseg_atlas coincidono con quelli di SAMSEG.

    python -m scripts.esecuzione.recompute_samseg_atlas --results-dir data/results_fbb_ants
    (lo stesso per results_fmm_ants, results_fbp_ants, results_dlbs)

Equivale a `run_pipeline --segmenter samseg_atlas` senza rifare SAMSEG.
"""
from __future__ import annotations
import argparse, glob, importlib, json, os, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared import setup_logger, load_image          # noqa: E402
from shared.percorsi import risolvi                  # noqa: E402
_voi = importlib.import_module("05_VOIExtraction")
_cl = importlib.import_module("06_CentiloidCalculation")


def parse_args():
    """Opzioni da riga di comando (vedi la docstring del modulo)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True,
                   help="Cartella con <soggetto>/samseg/result.csv; i nuovi "
                        "risultati vanno nella stessa cartella.")
    p.add_argument("--source-method", default="samseg")
    p.add_argument("--method-name", default="samseg_atlas")
    p.add_argument("--atlas-transform", default="SyN")
    p.add_argument("--skip-done", action="store_true",
                   help="Salta i soggetti che hanno gia' il risultato.")
    return p.parse_args()


def main():
    """Calcola samseg_atlas per ogni soggetto che ha un risultato SAMSEG."""
    log = setup_logger("samseg_atlas")
    a = parse_args()
    results_dir = Path(a.results_dir)
    cfg = _voi.config_predefinita(per_samseg=True,
                                  type_of_transform=a.atlas_transform)

    csvs = sorted(glob.glob(os.path.join(a.results_dir, "*", a.source_method,
                                         "result.csv")))
    log.info("%d soggetti '%s' in %s", len(csvs), a.source_method, a.results_dir)
    ok, falliti = 0, []
    for f in csvs:
        # La riga SAMSEG fa da base: si cambia il metodo e si sovrascrivono
        # solo i campi ricalcolati, il resto (percorsi, tracciante) resta.
        row = pd.read_csv(f, dtype={"subject_id": str}).iloc[0].to_dict()
        subj = str(row["subject_id"]).strip()
        row["segmenter"] = a.method_name
        subj_dir = results_dir / subj / a.method_name
        if a.skip_done and (subj_dir / "result.csv").exists():
            log.info("[gia' fatto] %s", subj)
            ok += 1
            continue
        try:
            # 1. Immagini gia' prodotte dal run SAMSEG.
            t1 = load_image(risolvi(row["anat_path"]))
            pet = load_image(risolvi(row["pet_path"]))
            seg = load_image(risolvi(row["seg_path"]))      # segmentazione SAMSEG

            # 2. Atlante deformato sul soggetto (il passo lento), poi
            #    intersezione con corteccia e cervelletto SAMSEG, poi SUVr e CL.
            _bm, ctx_a, wc_a, winfo = _voi.warp_atlas_to_native(
                t1, cfg, want_brain_mask=False)
            voi_ctx, voi_wc, vinfo = _voi.intersect_vois_atlas(
                ctx_a, wc_a, seg, cfg, warp_info=winfo)
            ctx_mean, wc_mean, suvr, cl, cinfo = _cl.compute_centiloid(
                pet, voi_ctx, voi_wc, str(row["tracer"]))

            # 3. Salvataggio nello stesso formato della pipeline.
            (subj_dir / "voi").mkdir(parents=True, exist_ok=True)
            cpath, wpath = _voi.save_vois(voi_ctx, voi_wc, str(subj_dir / "voi"),
                                          prefix=f"{a.method_name}_")
            row.update({"ctx_mean": ctx_mean, "wc_mean": wc_mean,
                        "suvr": suvr, "centiloid": cl,
                        "voi_ctx_path": cpath, "voi_wc_path": wpath,
                        "voi_extraction_method": vinfo["method"],
                        "ctx_voxels": vinfo["ctx_voxels"],
                        "wc_voxels": vinfo["wc_voxels"]})
            pd.DataFrame([row]).to_csv(subj_dir / "result.csv", index=False)
            with open(subj_dir / "info.json", "w") as fh:
                json.dump({"voi": vinfo, "centiloid": cinfo}, fh, indent=2, default=str)
            log.info("%-10s  CTX=%d WC=%d  SUVr=%.3f  CL=%.1f", subj,
                     vinfo["ctx_voxels"], vinfo["wc_voxels"], suvr, cl)
            ok += 1
        except Exception as e:
            log.error("NON RIUSCITO %s: %s", subj, e)
            falliti.append({"subject_id": subj, "error": str(e)})

    if falliti:
        pd.DataFrame(falliti).to_csv(results_dir / f"{a.method_name}_failures.csv",
                                     index=False)
    log.info("Fatto: %d riusciti, %d non riusciti.", ok, len(falliti))


if __name__ == "__main__":
    main()
