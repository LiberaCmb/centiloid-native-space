"""Tabella unica dei centiloidi, grezzi e ricalibrati, soggetto per soggetto.

I `result.csv` dei risultati contengono il centiloide con la formula pubblicata
del tracciante ("grezzo"), che in spazio nativo sottostima. Questo script li
raccoglie tutti e aggiunge il centiloide ricalibrato con le rette di
regressione della tesi (`recalibration_final.csv`):

  * FBB, FMM, FBP: retta stimata sul tracciante stesso, per ciascun metodo;
  * DLBS: retta stimata sul florbetapir del Centiloid Project, applicata
    invariata, come nella validazione del capitolo 4.

Per ogni soggetto si riportano anche il valore di riferimento (GAAIN o
QAmyloid) e se il soggetto e' escluso dal confronto, e perche' (florbetapir
con SUVr di SynthSeg sotto 0.85, DLBS con riferimento inaffidabile).

    python -m scripts.analisi.tabella_centiloidi
Uscita: data/comparison_all_methods/centiloidi_ricalibrati.xlsx
        (un foglio per archivio) e centiloidi_ricalibrati.csv (tutto insieme).
"""
from __future__ import annotations
import argparse, importlib, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared import percorsi                    # noqa: E402

_cmp = importlib.import_module("07_Comparison")

# Tracciante da cui prendere la retta per ciascun archivio: il DLBS e'
# florbetapir e usa le rette stimate sul florbetapir GAAIN.
RETTA_DA = {"FBB": "FBB", "FMM": "FMM", "FBP": "FBP", "DLBS": "FBP"}


def riferimenti(ds: str) -> dict:
    """{soggetto: centiloide di riferimento} per l'archivio `ds`."""
    if ds in _cmp.GROUND_TRUTH:
        gt = _cmp.leggi_ground_truth(percorsi.GROUND_TRUTH / _cmp.GROUND_TRUTH[ds])
        return dict(zip(gt.subject_id.astype(str), gt.truth_cl))
    q = pd.read_excel(percorsi.QAMYLOID_DLBS)
    col_id = [c for c in q.columns if "participant" in c.lower()][0]
    col_cl = [c for c in q.columns if c.strip().lower() == "cl qamyloid"][0]
    return {str(k).strip(): v for k, v in zip(q[col_id], q[col_cl])}


def tabella(ds: str, coeff: pd.DataFrame) -> pd.DataFrame:
    # 1. Tutti i result.csv dell'archivio (metodi con etichette e con atlante).
    r = _cmp.raccogli_risultati([percorsi.dati(c)
                                 for c in _cmp.CARTELLE_RISULTATI[ds]])
    r = r[r.segmenter.isin(_cmp.METODI)].copy()

    # 2. Centiloide ricalibrato: retta di regressione del metodo.
    c = coeff[(coeff.tracciante == RETTA_DA[ds]) & (coeff.calibrazione == "regressione")]
    rette = {m: (float(a), float(b)) for m, a, b in zip(c.metodo, c.a, c.b)}
    r["cl_ricalibrato"] = [rette[m][0] * s + rette[m][1] if m in rette else float("nan")
                           for m, s in zip(r.segmenter, r.suvr)]

    # 3. Riferimento ed esclusioni, con la stessa regola del confronto finale.
    rif = riferimenti(ds)
    r["cl_riferimento"] = r.subject_id.map(rif)
    r["escluso"] = ""
    if ds == "FBP":
        ss = r[r.segmenter == "synthseg"].set_index("subject_id").suvr
        bassi = {s for s, v in ss.items() if not (v >= _cmp.SUVR_MIN_FBP)}
        r.loc[r.subject_id.isin(bassi), "escluso"] = (
            f"SUVr SynthSeg < {_cmp.SUVR_MIN_FBP} o assente")
    if ds == "DLBS":
        r.loc[r.subject_id.isin(_cmp.DLBS_QC_ESCLUSI), "escluso"] = \
            "riferimento inaffidabile (QC)"

    tab = r.rename(columns={"subject_id": "soggetto", "segmenter": "metodo",
                            "centiloid": "cl_grezzo"})
    tab["errore_ricalibrato"] = tab.cl_ricalibrato - tab.cl_riferimento
    cols = ["soggetto", "metodo", "suvr", "cl_grezzo", "cl_ricalibrato",
            "cl_riferimento", "errore_ricalibrato", "escluso"]
    ordine = {m: i for i, m in enumerate(_cmp.METODI)}
    tab = tab[cols].sort_values(["soggetto", "metodo"],
                                key=lambda s: s.map(ordine) if s.name == "metodo" else s)
    tab.insert(0, "archivio", ds)
    return tab.reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(percorsi.CONFRONTO / "centiloidi_ricalibrati.xlsx"))
    a = ap.parse_args()
    coeff = pd.read_csv(percorsi.COEFFICIENTI)
    print(f"Coefficienti da {percorsi.COEFFICIENTI}")

    fogli = {ds: tabella(ds, coeff) for ds in ("FBB", "FMM", "FBP", "DLBS")}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for ds, t in fogli.items():
            t.round(4).to_excel(xw, sheet_name=ds, index=False)
    tutto = pd.concat(fogli.values(), ignore_index=True)
    tutto.round(4).to_csv(out.with_suffix(".csv"), index=False)

    # Riepilogo a schermo: soggetti e MAE ricalibrato sui soggetti inclusi.
    inc = tutto[tutto.escluso == ""].dropna(subset=["errore_ricalibrato"])
    print(inc.groupby(["archivio", "metodo"], sort=False)
             .agg(soggetti=("soggetto", "nunique"),
                  MAE=("errore_ricalibrato", lambda e: e.abs().mean()))
             .round(2).to_string())
    print(f"\nScritti {out} e {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
