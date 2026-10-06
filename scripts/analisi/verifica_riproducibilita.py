"""Riesegue il codice consegnato su alcuni soggetti e confronta i risultati con
quelli gia' calcolati (i result.csv in data/results_*).

Due modalita':

  riusa     (predefinita) parte dalle immagini gia' allineate da ANTs usate
            nella tesi e riesegue segmentazione, VOI e centiloide (moduli
            04-06). Per i metodi con etichette ci si aspettano valori uguali;
            per i metodi con atlante (otsu, kmeans, fsl_fast, samseg_atlas) la
            registrazione SyN del template non e' deterministica e il SUVr puo'
            cambiare di qualche millesimo.
  completo  parte dai NIfTI del manifest e riesegue anche l'allineamento ANTs
            (moduli 01-06). L'allineamento con le impostazioni predefinite non
            e' deterministico: differenze di qualche decimo di centiloide sono
            attese (capitolo 3, ripetibilita' di ANTs).

I risultati nuovi vanno in data/verifica/<modalita>/<soggetto>/<metodo>; i
risultati della tesi non vengono toccati. Alla fine stampa una tabella e la
salva in data/verifica/<modalita>/confronto.csv.

    python -m scripts.analisi.verifica_riproducibilita
    python -m scripts.analisi.verifica_riproducibilita --metodi otsu kmeans synthseg
    python -m scripts.analisi.verifica_riproducibilita --modo completo \\
        --soggetti FBB:1030 FMM:AD_105 --metodi synthseg
"""
from __future__ import annotations
import argparse, importlib, sys, time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared import percorsi, setup_logger                         # noqa: E402
from shared.percorsi import risolvi                               # noqa: E402
from shared.pipeline import METODI, esegui_soggetto, scrivi_risultato  # noqa: E402

_cmp = importlib.import_module("07_Comparison")
MANIFEST = {"FBB": "manifest_fbb_nifti.csv", "FMM": "manifest_fmm_nifti.csv",
            "FBP": "manifest_fbp_nifti.csv"}
# 1030: errore piccolo con tutti i metodi; 1018: PET negativa;
# AD_105: flutemetamol, il soggetto della figura ANTs/SimpleITK
PREDEFINITI = ["FBB:1030", "FBB:1018", "FMM:AD_105"]


def risultato_tesi(ds: str, sid: str, metodo: str) -> pd.Series | None:
    """Riga del result.csv della tesi per soggetto e metodo; None se manca."""
    for cart in _cmp.CARTELLE_RISULTATI[ds]:
        f = percorsi.dati(cart, sid, metodo, "result.csv")
        if f.exists():
            return pd.read_csv(f, dtype={"subject_id": str}).iloc[0]
    return None


def ingressi(modo: str, ds: str, sid: str, vecchio: pd.Series):
    """Percorsi (MRI, PET) da cui ripartire.

    In modo "riusa" sono le immagini gia' allineate registrate nel result.csv
    della tesi; in modo "completo" i NIfTI originali del manifest.
    """
    if modo == "riusa":
        return str(risolvi(vecchio.anat_path)), str(risolvi(vecchio.pet_path))
    man = pd.read_csv(percorsi.dati(MANIFEST[ds]), dtype={"subject_id": str})
    riga = man[man.subject_id.str.strip() == sid].iloc[0]
    return str(risolvi(riga.mri_path)), str(risolvi(riga.pet_path))


def ricalibra(ds: str, metodo: str, suvr: float) -> float:
    """Centiloide ricalibrato per regressione; NaN se mancano i coefficienti."""
    c = pd.read_csv(percorsi.COEFFICIENTI)
    c = c[(c.tracciante == ds) & (c.metodo == metodo) & (c.calibrazione == "regressione")]
    return float(c.a.iloc[0] * suvr + c.b.iloc[0]) if len(c) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--soggetti", nargs="+", default=PREDEFINITI,
                    help="coppie TRACCIANTE:ID, per esempio FBB:1030")
    ap.add_argument("--metodi", nargs="+", default=["otsu", "kmeans", "synthseg"],
                    choices=METODI)
    ap.add_argument("--modo", choices=["riusa", "completo"], default="riusa")
    a = ap.parse_args()
    log = setup_logger("verifica")

    # 1. Per ogni coppia soggetto-metodo: risultato della tesi, nuova
    #    esecuzione della pipeline in data/verifica e confronto di SUVr, CL
    #    ricalibrato e voxel del target.
    base = percorsi.dati("verifica", a.modo)
    righe = []
    for voce in a.soggetti:
        ds, sid = voce.split(":", 1)
        for metodo in a.metodi:
            vecchio = risultato_tesi(ds, sid, metodo)
            if vecchio is None:
                log.warning("%s %s: nessun risultato della tesi, saltato", sid, metodo)
                continue
            mri, pet = ingressi(a.modo, ds, sid, vecchio)
            out = base / sid / metodo
            log.info("=== %s (%s), %s, modo %s ===", sid, ds, metodo, a.modo)
            t0 = time.time()
            try:
                ris = esegui_soggetto(
                    mri, pet, ds, metodo, str(out),
                    riusa_allineamento=(a.modo == "riusa"),
                    id_soggetto=sid, dataset=str(vecchio.dataset),
                    progresso=lambda fase, frazione, msg: log.info(msg))
            except Exception as e:                       # un metodo mancante non
                log.error("%s %s non riuscito: %s", sid, metodo, e)   # ferma gli altri
                righe.append({"soggetto": sid, "tracciante": ds, "metodo": metodo,
                              "esito": f"errore: {e}"})
                continue
            scrivi_risultato(out, ris.record, ris.info)
            nuovo = ris.record
            # stessi coefficienti per vecchio e nuovo: la differenza in CL
            # viene solo dal SUVr
            cl_v = ricalibra(ds, metodo, float(vecchio.suvr))
            cl_n = ricalibra(ds, metodo, float(nuovo["suvr"]))
            righe.append({
                "soggetto": sid, "tracciante": ds, "metodo": metodo,
                "SUVr tesi": round(float(vecchio.suvr), 4),
                "SUVr ora": round(float(nuovo["suvr"]), 4),
                "CL tesi": round(cl_v, 2), "CL ora": round(cl_n, 2),
                "diff CL": round(cl_n - cl_v, 2),
                "voxel target tesi": int(vecchio.ctx_voxels),
                "voxel target ora": int(nuovo["ctx_voxels"]),
                "minuti": round((time.time() - t0) / 60, 1),
                "esito": "ok"})

    # 2. Tabella di confronto, salvata e stampata.
    t = pd.DataFrame(righe)
    base.mkdir(parents=True, exist_ok=True)
    t.to_csv(base / "confronto.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nCentiloide ricalibrato (regressione, coefficienti della tesi):")
    print(t.to_string(index=False))
    print(f"\nTabella salvata in {base / 'confronto.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
