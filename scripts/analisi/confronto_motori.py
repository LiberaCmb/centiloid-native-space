"""ANTs contro SimpleITK: qualita' dell'allineamento e accuratezza del centiloide.

Due piani di confronto (capitolo 3, sezione sull'allineamento):

  qualita' dell'allineamento   NMI e correlazione dei gradienti, dal file di
                               compare_alignment_quality.py;
  accuratezza del centiloide   a parita' di segmentazione e di calibrazione,
                               quanto ciascun motore si avvicina al
                               riferimento GAAIN.

Sul secondo piano i coefficienti di ricalibrazione si ristimano su CIASCUN
motore separatamente (calibrazione per regressione, come nel resto del
lavoro): altrimenti si misurerebbe quanto un motore somiglia all'altro invece
di quanto bene allinea. Il confronto e' appaiato: per ogni metodo si usano i
soli soggetti presenti in entrambi.

Il braccio SimpleITK vive in data/results_{fbb,fmm} (con la sua cache di
allineamento), quello ANTs in data/results_{fbb,fmm}_ants. Si confrontano i
tre metodi a etichette, gli unici calcolati con entrambi i motori.

    python -m scripts.analisi.confronto_motori
Uscite (in data/comparison_all_methods/): confronto_motori.csv e
confronto_motori.xlsx (fogli "Centiloide", "Qualita registrazione",
"Coefficienti per braccio").
"""
from __future__ import annotations
import argparse, importlib, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared import percorsi                       # noqa: E402
_cmp = importlib.import_module("07_Comparison")

METODI = ["synthseg", "fastsurfer", "samseg"]
BRACCI = {
    "FBB": {"sitk": "results_fbb", "ants": "results_fbb_ants"},
    "FMM": {"sitk": "results_fmm", "ants": "results_fmm_ants"},
}


def leggi_suvr(cartella: Path) -> pd.DataFrame:
    """SUVr per soggetto e metodo dai `result.csv` di un braccio.

    I file mancanti o illeggibili si saltano in silenzio; restituisce le
    colonne subject_id, metodo, suvr (vuoto se la cartella non esiste).
    """
    righe = []
    if not cartella.exists():
        return pd.DataFrame(columns=["subject_id", "metodo", "suvr"])
    for sub in sorted(p for p in cartella.iterdir() if p.is_dir()):
        for met in METODI:
            f = sub / met / "result.csv"
            if not f.exists():
                continue
            try:
                r = pd.read_csv(f).iloc[0]
            except Exception:
                continue
            righe.append({"subject_id": str(r["subject_id"]), "metodo": met,
                          "suvr": float(r["suvr"])})
    return pd.DataFrame(righe).drop_duplicates(["subject_id", "metodo"])


def metriche(stima, rif):
    """Numero, bias, MAE, RMSE e r di Pearson dell'errore stima - riferimento."""
    e = stima - rif
    return {"n": len(e), "bias": e.mean(), "mae": np.abs(e).mean(),
            "rmse": float(np.sqrt((e ** 2).mean())),
            "r": float(np.corrcoef(stima, rif)[0, 1]) if len(e) > 2 else np.nan}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qualita", default=str(percorsi.CONFRONTO / "alignment_quality.csv"))
    ap.add_argument("--out", default=str(percorsi.CONFRONTO))
    a = ap.parse_args()

    # ── piano 1: qualita' dell'allineamento ──────────────────────────────────
    q = pd.read_csv(a.qualita)
    print(f"Qualita' dell'allineamento, {len(q)} soggetti "
          f"({', '.join(f'{k}: {v}' for k, v in q.tracer.value_counts().items())})")
    # i pareggi esatti si contano a parte
    righe = []
    for ind, ca, cs in (("NMI", "nmi_ants", "nmi_sitk"), ("GC", "gc_ants", "gc_sitk")):
        meglio = int((q[ca] > q[cs]).sum())
        pari = int((q[ca] == q[cs]).sum())
        peggio = len(q) - meglio - pari
        print(f"  {ind:3s}: ANTs migliore su {meglio}/{len(q)}, pari su {pari}, "
              f"peggiore su {peggio}  (media {q[ca].mean():.4f} contro {q[cs].mean():.4f})")
        righe.append({"indice": ind, "ants_migliore": meglio, "pari": pari,
                      "sitk_migliore": peggio, "n": len(q),
                      "media_ants": round(q[ca].mean(), 4),
                      "media_sitk": round(q[cs].mean(), 4)})
    qual = pd.DataFrame(righe)

    # ── piano 2: accuratezza del centiloide ──────────────────────────────────
    out, coeff = [], []
    for ds, bracci in BRACCI.items():
        gt = _cmp.leggi_ground_truth(percorsi.GROUND_TRUTH / _cmp.GROUND_TRUTH[ds])
        dati = {m: leggi_suvr(percorsi.dati(d)) for m, d in bracci.items()}
        for met in METODI:
            # Confronto appaiato: soli soggetti con riferimento e con un
            # risultato in entrambi i bracci.
            comuni = set(gt.subject_id)
            for m in bracci:
                comuni &= set(dati[m][dati[m].metodo == met].subject_id)
            comuni = sorted(comuni)
            if not comuni:
                continue
            riga = {"dataset": ds, "metodo": met, "n": len(comuni)}
            # Retta di ricalibrazione stimata sul motore stesso (polyfit di
            # grado 1 = minimi quadrati), poi metriche contro il riferimento.
            for m in ("ants", "sitk"):
                d = (dati[m][(dati[m].metodo == met) & (dati[m].subject_id.isin(comuni))]
                     .merge(gt, on="subject_id").sort_values("subject_id"))
                A, B = np.polyfit(d.suvr.values, d.truth_cl.values, 1)
                mm = metriche(A * d.suvr.values + B, d.truth_cl.values)
                riga.update({f"{m}_{k}": v for k, v in mm.items() if k != "n"})
                coeff.append({"dataset": ds, "metodo": met, "motore": m,
                              "a": round(A, 4), "b": round(B, 4), "n": len(d)})
            # negativa = ANTs piu' vicino al riferimento
            riga["differenza_mae"] = riga["ants_mae"] - riga["sitk_mae"]
            out.append(riga)

    # ── uscite: CSV, Excel a tre fogli e sintesi a terminale ───────────────
    res = pd.DataFrame(out)
    num = res.select_dtypes("number").columns
    res[num] = res[num].round(3)
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    res.to_csv(outdir / "confronto_motori.csv", index=False)
    with pd.ExcelWriter(outdir / "confronto_motori.xlsx", engine="openpyxl") as xw:
        res.to_excel(xw, sheet_name="Centiloide", index=False)
        qual.to_excel(xw, sheet_name="Qualita registrazione", index=False)
        pd.DataFrame(coeff).to_excel(xw, sheet_name="Coefficienti per braccio",
                                     index=False)

    print("\nAccuratezza del centiloide (regressione stimata su ciascun motore):\n")
    print(res[["dataset", "metodo", "n", "ants_mae", "sitk_mae", "differenza_mae",
               "ants_r", "sitk_r"]].to_string(index=False))
    print(f"\nANTs ha l'errore minore in {int((res.differenza_mae < 0).sum())} "
          f"delle {len(res)} combinazioni.\nScritti in {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
