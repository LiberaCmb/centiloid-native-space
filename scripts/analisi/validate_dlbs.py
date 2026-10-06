"""Validazione indipendente sul DLBS.

Prende i coefficienti di ricalibrazione stimati per REGRESSIONE sull'archivio
florbetapir del Centiloid Project (GAAIN, 32 soggetti puliti) e li applica
INVARIATI ai SUVr misurati sul Dallas Lifespan Brain Study, confrontando il
risultato con i centiloidi di riferimento QAmyloid.

E' la verifica che la calibrazione trasferisca a una popolazione diversa: se
l'errore sul DLBS con i coefficienti GAAIN e' vicino a quello che si otterrebbe
stimando la retta sul DLBS stesso, la ricalibrazione non e' un fit in-sample.

Uso:
    python -m scripts.analisi.validate_dlbs
    python -m scripts.analisi.validate_dlbs --out data/comparison_all_methods

Ingressi:
    data/results_dlbs/<soggetto>/<metodo>/result.csv      SUVr per soggetto
    data/comparison_all_methods/recalibration_final.csv   coefficienti a, b
    data/ground_truth/QAmyloid_results_DLBS.xlsx          riferimento QAmyloid

Uscite (in data/comparison_all_methods/): validazione_dlbs.csv e
validazione_dlbs.xlsx (fogli "Validazione DLBS" ed "Esclusi QC"). Il codice di
uscita e' 1 se i MAE di controllo non coincidono con quelli attesi.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared.percorsi import risolvi  # noqa: E402

import importlib                       # noqa: E402
_cmp = importlib.import_module("07_Comparison")

# Soggetti con valore di riferimento inaffidabile (criterio in
# 07_Comparison/logic.py). Da non confondere con i sette soggetti del foglio
# (c) di gestione_outlier_abc.xlsx (sub-1136, sub-1891, sub-2044, sub-4575,
# sub-4615, sub-4660, sub-4828): quelli sono coppie metodo-soggetto in cui UN
# metodo sbaglia mentre gli altri no, un problema del metodo e non del dato,
# e NON vanno esclusi. I due elenchi hanno la stessa lunghezza per caso.
DLBS_QC_ESCLUSI = _cmp.DLBS_QC_ESCLUSI
METODI = _cmp.METODI


def leggi_suvr(results_dir: Path) -> pd.DataFrame:
    """SUVr per soggetto e metodo dai result.csv del modulo 06."""
    righe = []
    for sub in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        for met in METODI:
            f = sub / met / "result.csv"
            if not f.exists():
                continue
            try:
                d = pd.read_csv(f).iloc[0]
            except Exception as e:
                print(f"  [salto] {f}: {e}")
                continue
            righe.append({"subject_id": sub.name, "metodo": met,
                          "suvr": float(d["suvr"])})
    return pd.DataFrame(righe)


def leggi_coefficienti(path: Path, tracciante="FBP", calib="regressione") -> dict:
    """{metodo: (a, b)} per un tracciante e una calibrazione."""
    d = pd.read_csv(path)
    d = d[(d["tracciante"].str.upper() == tracciante) &
          (d["calibrazione"] == calib)]
    return {r["metodo"]: (float(r["a"]), float(r["b"])) for _, r in d.iterrows()}


def leggi_riferimento(path: Path) -> pd.DataFrame:
    """Centiloidi QAmyloid (subject_id, cl_rif); colonne trovate per nome."""
    d = pd.read_excel(path)
    col_id = [c for c in d.columns if "participant" in c.lower()][0]
    col_cl = [c for c in d.columns if c.strip().lower() == "cl qamyloid"][0]
    return (d[[col_id, col_cl]]
            .rename(columns={col_id: "subject_id", col_cl: "cl_rif"})
            .dropna())


def metriche(stima: np.ndarray, rif: np.ndarray) -> dict:
    """Numero, bias, MAE, RMSE, dev.std dell'errore e r di Pearson."""
    e = stima - rif
    r = np.corrcoef(stima, rif)[0, 1] if len(stima) > 2 else np.nan
    return {"n": len(e), "bias": e.mean(), "mae": np.abs(e).mean(),
            "rmse": float(np.sqrt((e ** 2).mean())), "sd": e.std(ddof=1),
            "r": r}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="data/results_dlbs")
    ap.add_argument("--coeff", default="data/comparison_all_methods/recalibration_final.csv")
    ap.add_argument("--riferimento", default="data/ground_truth/QAmyloid_results_DLBS.xlsx")
    ap.add_argument("--out", default="data/comparison_all_methods")
    a = ap.parse_args()

    # 1. Ingressi: SUVr del DLBS, coefficienti FBP, riferimento QAmyloid.
    suvr = leggi_suvr(risolvi(a.results))
    coeff = leggi_coefficienti(risolvi(a.coeff))
    rif = leggi_riferimento(risolvi(a.riferimento))
    print(f"SUVr letti: {len(suvr)} righe, {suvr.subject_id.nunique()} soggetti, "
          f"{suvr.metodo.nunique()} metodi")
    print(f"Riferimento QAmyloid: {len(rif)} soggetti")
    print(f"Coefficienti FBP (regressione) per {len(coeff)} metodi\n")

    # 2. Per ogni metodo: coefficienti GAAIN applicati invariati, su tutti i
    #    soggetti e senza i sette del QC, a confronto con la retta in-sample.
    d = suvr.merge(rif, on="subject_id", how="inner")
    out = []
    for met in METODI:
        if met not in coeff:
            print(f"  [salto] nessun coefficiente per {met}")
            continue
        A, B = coeff[met]
        m = d[d.metodo == met].dropna(subset=["suvr", "cl_rif"])
        for etichetta, sel in (("tutti", m),
                               ("senza 7 QC", m[~m.subject_id.isin(DLBS_QC_ESCLUSI)])):
            cl = A * sel.suvr.values + B
            r = metriche(cl, sel.cl_rif.values)
            # confronto: retta ristimata sul DLBS stesso (fit in-sample)
            a2, b2 = np.polyfit(sel.suvr.values, sel.cl_rif.values, 1)
            r_fit = metriche(a2 * sel.suvr.values + b2, sel.cl_rif.values)
            out.append({"metodo": met, "campione": etichetta,
                        "a_FBP": A, "b_FBP": B, **r,
                        "mae_fit_su_DLBS": r_fit["mae"],
                        "delta_mae": r["mae"] - r_fit["mae"]})

    # 3. Tabella arrotondata, CSV ed Excel.
    res = pd.DataFrame(out).round(
        {"a_FBP": 4, "b_FBP": 4, "bias": 2, "mae": 2, "rmse": 2, "sd": 2,
         "r": 4, "mae_fit_su_DLBS": 2, "delta_mae": 2})
    outdir = risolvi(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    res.to_csv(outdir / "validazione_dlbs.csv", index=False)
    with pd.ExcelWriter(outdir / "validazione_dlbs.xlsx", engine="openpyxl") as xw:
        res.to_excel(xw, sheet_name="Validazione DLBS", index=False)
        pd.DataFrame({"soggetti esclusi (riferimento inaffidabile)":
                      DLBS_QC_ESCLUSI}).to_excel(xw, sheet_name="Esclusi QC",
                                                 index=False)

    print(res.to_string(index=False))
    print(f"\nScritti {outdir/'validazione_dlbs.csv'} e .xlsx")

    # ── controllo: i valori attesi dal foglio "Coeff FBP applicati a DLBS"
    #    (tolleranza 0.02 CL, cioe' l'arrotondamento a due decimali)
    attesi = {("synthseg", "tutti"): 4.29, ("synthseg", "senza 7 QC"): 3.96,
              ("fsl_fast", "tutti"): 9.64}
    print("\nControllo contro il foglio del 15 luglio:")
    ok = True
    for (met, camp), atteso in attesi.items():
        got = res[(res.metodo == met) & (res.campione == camp)]["mae"]
        got = float(got.iloc[0]) if len(got) else float("nan")
        segno = "OK " if abs(got - atteso) <= 0.02 else "DIVERSO"
        ok &= segno == "OK "
        print(f"  {segno}  {met:12s} {camp:11s} atteso {atteso:5.2f}  ottenuto {got:5.2f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
