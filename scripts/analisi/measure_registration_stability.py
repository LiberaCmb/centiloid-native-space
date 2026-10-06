"""Variabilita' dell'allineamento SimpleITK da un seme all'altro.

Per ogni soggetto del manifest esegue l'allineamento SimpleITK con N tentativi
(semi 0..N-1) e registra l'informazione mutua finale di ciascuno, con la loro
sintesi: migliore, media, deviazione standard, peggiore, escursione. Non
scrive immagini e non tocca i risultati esistenti.

E' l'esperimento che motiva il multi-restart (capitolo 3): sulla serie
preliminare di 79 soggetti (archivio PiB 50-70 minuti del Centiloid Project)
l'informazione mutua cambiava da un seme all'altro di 0.31 in mediana e di
0.55 nel soggetto peggiore.

    python -m scripts.analisi.measure_registration_stability \\
        --manifest data/manifest.csv --out data/registration_stability.csv

Uscita: il CSV indicato da --out, una riga per soggetto con la sintesi e i
valori dei singoli tentativi (colonne trial_0 ... trial_{N-1}).
"""
from __future__ import annotations
import argparse
import csv
import importlib
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import setup_logger, ProjectContext     # noqa: E402
_load = importlib.import_module("01_LoadImages")
_realign = importlib.import_module("02_Realignment")

# Formato del messaggio di 02_Realignment/registration.py: tentativi riusciti
# su totali, migliore, media, dev.std e l'elenco dei valori dei tentativi.
_RE_MESSAGGIO = re.compile(
    r"\((\d+)/(\d+) ok\):\s+best=([-\d.]+)\s+mean=([-\d.]+)\s+"
    r"std=([\d.]+)\s+trials=\[(.*?)\]")


def parse_args():
    """Opzioni da riga di comando (vedi la docstring del modulo)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True,
                   help="CSV con subject_id, dataset, tracer, mri_path, pet_path.")
    p.add_argument("--out", required=True, help="CSV di uscita.")
    p.add_argument("--n-restarts", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def misura(row, n_restarts: int, log) -> dict | None:
    """Allinea un soggetto con `n_restarts` tentativi e ne sintetizza la MI.

    row         riga del manifest (subject_id, tracer, mri_path, pet_path, ...)
    n_restarts  numero di tentativi (semi 0..n_restarts-1)
    log         logger su cui riportare avanzamento ed errori

    Restituisce la riga del CSV, o None se caricamento, allineamento o
    lettura del messaggio non riescono.
    """
    # 1. Contesto del soggetto e caricamento di MRI e PET.
    subj = str(row["subject_id"])
    ctx = ProjectContext()
    ctx.subject_id = subj
    ctx.dataset_name = str(row.get("dataset", ""))
    ctx.is_co_acquired = int(row.get("co_acquired", 0)) == 1
    ctx.selected_tracer = str(row["tracer"])
    try:
        for slot, col in (("anat_1", "mri_path"), ("func_1", "pet_path")):
            ok, msg = _load.update_context_with_file(ctx, slot, str(row[col]))
            if not ok:
                log.error("  %s: caricamento non riuscito: %s", subj, msg)
                return None

        # 2. Allineamento multi-restart, cronometrato.
        t0 = time.time()
        transform, msg = _realign.run_registration(ctx, n_restarts=n_restarts)
        runtime_s = time.time() - t0
        if transform is None:
            log.error("  %s: allineamento non riuscito: %s", subj, msg)
            return None

        # 3. Le statistiche si leggono dal messaggio di ritorno del modulo 02.
        #    La metrica di SimpleITK e' l'MI cambiata di segno (piu' bassa =
        #    migliore): per questo il tentativo peggiore e' il massimo.
        m = _RE_MESSAGGIO.search(msg)
        if not m:
            log.warning("  %s: messaggio non interpretabile: %s", subj, msg[:200])
            return None
        trials = [float(x.strip().strip("'\"")) for x in m.group(6).split(",")]
        out = {"subject_id": subj, "tracer": str(row["tracer"]),
               "n_ok": int(m.group(1)), "n_tot": int(m.group(2)),
               "best_MI": float(m.group(3)), "mean_MI": float(m.group(4)),
               "std_MI": float(m.group(5)), "worst_MI": max(trials),
               "MI_range": max(trials) - min(trials),
               "runtime_s": round(runtime_s, 2)}
        for i, t in enumerate(trials):
            out[f"trial_{i}"] = t
        log.info("  %s: migliore=%.3f dev.std=%.3f escursione=%.3f (%.0f s)",
                 subj, out["best_MI"], out["std_MI"], out["MI_range"], runtime_s)
        return out
    except Exception as e:
        log.error("  %s: %s", subj, e)
        return None


def main():
    """Misura tutti i soggetti del manifest e scrive il CSV di uscita."""
    args = parse_args()
    log = setup_logger("stabilita")
    manifest = pd.read_csv(args.manifest)
    if args.limit:
        manifest = manifest.head(args.limit)
    log.info("%d soggetti da %s (%d tentativi ciascuno)",
             len(manifest), args.manifest, args.n_restarts)

    righe = []
    for i, row in manifest.iterrows():
        log.info("[%d/%d]", i + 1, len(manifest))
        r = misura(row, args.n_restarts, log)
        if r is not None:
            righe.append(r)
    if not righe:
        log.error("Nessun soggetto misurato.")
        sys.exit(1)

    # Colonne fisse; i tentativi falliti lasciano vuote le colonne trial_*.
    campi = ["subject_id", "tracer", "n_ok", "n_tot", "best_MI", "mean_MI",
             "std_MI", "worst_MI", "MI_range", "runtime_s"]
    campi += [f"trial_{i}" for i in range(args.n_restarts)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campi)
        w.writeheader()
        for r in righe:
            for k in campi:
                r.setdefault(k, "")
            w.writerow(r)
    log.info("Scritte %d righe in %s", len(righe), out)

    escursioni = [r["MI_range"] for r in righe]
    log.info("Escursione dell'informazione mutua fra i tentativi: mediana=%.3f "
             "massima=%.3f", np.median(escursioni), max(escursioni))


if __name__ == "__main__":
    main()
