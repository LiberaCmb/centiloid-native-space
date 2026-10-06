"""Metriche complete, depurate e robuste, sotto la calibrazione per regressione.

Il capitolo dei risultati presenta i centiloidi ricalibrati per REGRESSIONE.
Questo script ricalcola, con quella calibrazione, tre viste degli stessi dati:

  completa   tutti i soggetti del dataset;
  depurata   esclusi i soggetti con valore di riferimento inaffidabile
             (solo DLBS: i sette del controllo qualita');
  robusta    esclusi, per ciascun metodo separatamente, i soggetti il cui
             errore e' un valore estremo. Serve a distinguere l'errore tipico
             di un metodo dai pochi casi catastrofici che da soli gonfiano
             media e deviazione standard.

Per «valore estremo» si riportano due criteri, perche' danno letture diverse:

  MAD   |e - mediana(e)| > 2.5 * 1.4826 * MAD(e), cioe' 2.5 deviazioni
        standard robuste dalla mediana degli errori. Si adatta alla
        dispersione del metodo ed e' il criterio da preferire.
  15CL  |e| > 15 CL, soglia fissa. Piu' grossolana, ma indipendente dalla
        dispersione del metodo e piu' facile da dichiarare in tesi.

Uso:  python -m scripts.analisi.robustezza_outlier
Uscite (in data/comparison_all_methods/): robustezza_regressione.csv e
robustezza_regressione.xlsx.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared.percorsi import risolvi  # noqa: E402
import importlib                       # noqa: E402
_cmp = importlib.import_module("07_Comparison")

METODI = _cmp.METODI

# GAAIN: i centiloidi per soggetto si leggono dal file finale prodotto da
# build_final_comparison.py, non si ricalcolano. Cosi' le esclusioni (in
# particolare i 14 florbetapir con acquisizione dimezzata) sono per
# costruzione le stesse del file di confronto, e i numeri non possono
# divergere da quelli riportati in tesi.
FILE_FINALE = ("data/comparison_all_methods/"
               "confronto_metodi_centiloide_ricalibrato_ANTs_finale.xlsx")
GAAIN_FOGLI = ["FBB", "FMM", "FBP"]

DLBS_DIR = "data/results_dlbs"
DLBS_RIF = "data/ground_truth/QAmyloid_results_DLBS.xlsx"
# Il criterio di esclusione e' spiegato in 07_Comparison/logic.py.
DLBS_QC_ESCLUSI = _cmp.DLBS_QC_ESCLUSI


def suvr_da_risultati(dirs) -> pd.DataFrame:
    """SUVr per soggetto e metodo dai `result.csv` delle cartelle `dirs`."""
    righe = []
    for d in dirs:
        base = risolvi(d)
        if not base.exists():
            continue
        for sub in sorted(p for p in base.iterdir() if p.is_dir()):
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


def coefficienti(path: Path) -> dict:
    """{(tracciante, metodo): (a, b)} della sola calibrazione per regressione."""
    d = pd.read_csv(path)
    d = d[d.calibrazione == "regressione"]
    return {(r["tracciante"].upper(), r["metodo"]): (float(r["a"]), float(r["b"]))
            for _, r in d.iterrows()}


def metriche(e: np.ndarray, stima=None, rif=None) -> dict:
    """Metriche dell'errore `e`; con `stima` e `rif` anche la r di Pearson.

    e      errori stima - riferimento (CL)
    stima  centiloidi stimati, solo per la correlazione
    rif    centiloidi di riferimento, solo per la correlazione
    """
    out = {"n": len(e), "bias": e.mean(), "bias_mediano": float(np.median(e)),
           "mae": np.abs(e).mean(),
           "rmse": float(np.sqrt((e ** 2).mean())),
           "sd": e.std(ddof=1) if len(e) > 1 else np.nan}
    if stima is not None and len(stima) > 2:
        out["r"] = float(np.corrcoef(stima, rif)[0, 1])
    return out


def estremi(e: np.ndarray, criterio: str) -> np.ndarray:
    """Maschera booleana dei soggetti da escludere."""
    if criterio == "MAD":
        med = np.median(e)
        mad = np.median(np.abs(e - med))
        if mad == 0:
            return np.zeros(len(e), bool)
        # 1.4826 rende il MAD uno stimatore della deviazione standard per
        # errori gaussiani; 2.5 e' il numero di deviazioni tollerate.
        return np.abs(e - med) > 2.5 * 1.4826 * mad
    return np.abs(e) > 15.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coeff", default="data/comparison_all_methods/recalibration_final.csv")
    ap.add_argument("--riferimento", default=DLBS_RIF,
                    help="Centiloidi QAmyloid del DLBS (xlsx).")
    ap.add_argument("--out", default="data/comparison_all_methods")
    a = ap.parse_args()

    # 1. GAAIN: centiloidi ricalibrati e riferimento letti dal file finale
    #    (colonna "<metodo>\nreg").
    coef = coefficienti(risolvi(a.coeff))
    blocchi = {}

    for ds in GAAIN_FOGLI:
        x = pd.read_excel(risolvi(FILE_FINALE), sheet_name=ds)
        x["subject_id"] = x.subject_id.astype(str)
        righe = []
        for met in METODI:
            col = f"{met}\nreg"
            if col not in x.columns:
                continue
            v = x[["subject_id", "ground_truth", col]].dropna()
            righe.append(pd.DataFrame({"subject_id": v.subject_id, "metodo": met,
                                       "cl": v[col].values,
                                       "rif": v.ground_truth.values}))
        blocchi[ds] = pd.concat(righe, ignore_index=True)
        print(f"[{ds}] {blocchi[ds].subject_id.nunique()} soggetti dal file finale")

    # 2. DLBS: SUVr dai risultati, riferimento QAmyloid e centiloide con i
    #    coefficienti di regressione stimati sul florbetapir GAAIN.
    s = suvr_da_risultati([DLBS_DIR])
    q = pd.read_excel(risolvi(a.riferimento))
    col_id = [c for c in q.columns if "participant" in c.lower()][0]
    col_cl = [c for c in q.columns if c.strip().lower() == "cl qamyloid"][0]
    q = q[[col_id, col_cl]].rename(columns={col_id: "subject_id", col_cl: "rif"})
    d = s.merge(q, on="subject_id")
    d["cl"] = [coef[("FBP", m)][0] * v + coef[("FBP", m)][1]
               for m, v in zip(d.metodo, d.suvr)]
    blocchi["DLBS"] = d
    print(f"[DLBS] {d.subject_id.nunique()} soggetti, coefficienti FBP applicati")

    # 3. Per ogni archivio e metodo le tre viste: completa, depurata (sul
    #    DLBS senza i sette del QC; altrove coincide con la completa) e
    #    robusta con i due criteri.
    out = []
    for ds, d in blocchi.items():
        for met in METODI:
            m = d[d.metodo == met].dropna(subset=["cl", "rif"])
            if m.empty:
                continue
            cl = m.cl.values
            e = cl - m.rif.values
            riga = {"dataset": ds, "metodo": met,
                    **{f"compl_{k}": v for k, v in metriche(e, cl, m.rif.values).items()}}

            if ds == "DLBS":
                sel = ~m.subject_id.isin(DLBS_QC_ESCLUSI).values
                e2, cl2 = e[sel], cl[sel]
                riga.update({f"depur_{k}": v for k, v in
                             metriche(e2, cl2, m.rif.values[sel]).items()})
            else:
                riga.update({f"depur_{k}": v for k, v in
                             metriche(e, cl, m.rif.values).items()})

            for crit in ("MAD", "15CL"):
                fuori = estremi(e, crit)
                keep = ~fuori
                riga[f"rob{crit}_n_esclusi"] = int(fuori.sum())
                riga[f"rob{crit}_mae"] = np.abs(e[keep]).mean() if keep.any() else np.nan
                riga[f"rob{crit}_sd"] = (e[keep].std(ddof=1) if keep.sum() > 1 else np.nan)
            out.append(riga)

    # 4. Scrittura di CSV ed Excel e stampa delle colonne principali.
    res = pd.DataFrame(out)
    num = res.select_dtypes("number").columns
    res[num] = res[num].round(3)
    outdir = risolvi(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    res.to_csv(outdir / "robustezza_regressione.csv", index=False)
    with pd.ExcelWriter(outdir / "robustezza_regressione.xlsx", engine="openpyxl") as xw:
        res.to_excel(xw, sheet_name="Completa depurata robusta", index=False)

    col = ["dataset", "metodo", "compl_n", "compl_mae", "compl_sd",
           "depur_n", "depur_mae", "robMAD_n_esclusi", "robMAD_mae",
           "rob15CL_n_esclusi", "rob15CL_mae"]
    print("\n" + res[col].to_string(index=False))
    print(f"\nScritti in {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
