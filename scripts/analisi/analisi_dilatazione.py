"""Effetto della dilatazione di un voxel del target corticale sul centiloide.

Legge i CSV di `scripts/esecuzione/dilation_recompute.py`
(data/comparison_all_methods/dilation_<DS>.csv,
con SUVr e centiloide senza e con dilatazione) e confronta le due versioni
contro il riferimento, con la calibrazione adottata nel resto del lavoro:

  * per ogni archivio e metodo si stima una retta di regressione per il SUVr
    senza dilatazione e una, separata, per il SUVr con dilatazione: i
    coefficienti pubblicati e quelli ricalibrati sono stimati su VOI non
    dilatate, e applicarli a un SUVr dilatato misurerebbe lo spostamento
    sistematico invece della qualita' del target;
  * riferimento: centiloidi GAAIN per FBB, FMM e FBP (i 32 soggetti puliti,
    stesso criterio del confronto finale); QAmyloid per il DLBS;
  * DLBS: retta stimata sul DLBS stesso (foglio "Riepilogo") e, a parte,
    coefficienti stimati sul florbetapir GAAIN e applicati invariati, su tutti
    i soggetti e senza i sette con riferimento inaffidabile (foglio "DLBS
    coeff. GAAIN", la stessa logica della validazione, usata nel capitolo 4).

Si riportano anche gli errori con la formula pubblicata ("raw"), utili per
vedere quanto la dilatazione sposti il SUVr prima di ogni ricalibrazione.

    python -m scripts.analisi.analisi_dilatazione
Uscita: data/comparison_all_methods/confronto_dilatazione_centiloide_sintesi.xlsx
"""
from __future__ import annotations
import argparse, importlib, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared import percorsi                    # noqa: E402
_cmp = importlib.import_module("07_Comparison")

ARCHIVI = ["FBB", "FMM", "FBP", "DLBS"]


def riferimento(ds: str, qamyloid: Path) -> dict:
    """Mappa soggetto -> centiloide di riferimento (GAAIN, o QAmyloid per il DLBS)."""
    if ds in _cmp.GROUND_TRUTH:
        gt = _cmp.leggi_ground_truth(percorsi.GROUND_TRUTH / _cmp.GROUND_TRUTH[ds])
        return dict(zip(gt.subject_id, gt.truth_cl))
    # DLBS: il riferimento e' il foglio QAmyloid; le colonne si cercano per
    # nome perche' l'intestazione del file originale non e' normalizzata.
    q = pd.read_excel(qamyloid)
    col_id = [c for c in q.columns if "participant" in c.lower()][0]
    col_cl = [c for c in q.columns if c.strip().lower() == "cl qamyloid"][0]
    return {str(k).strip(): v for k, v in zip(q[col_id], q[col_cl])}


def leggi(ds: str, qamyloid: Path, suvr_min_fbp: float) -> pd.DataFrame | None:
    """CSV di dilatazione di un archivio, filtrato e con il riferimento accanto.

    ds            'FBB', 'FMM', 'FBP' o 'DLBS'
    qamyloid      file Excel QAmyloid (usato solo per il DLBS)
    suvr_min_fbp  soglia di SUVr (synthseg, senza dilatazione) sotto cui un
                  soggetto florbetapir e' escluso, come nel confronto finale

    Restituisce le righe con `truth_cl` valorizzato, o None se il CSV manca.
    """
    f = percorsi.CONFRONTO / f"dilation_{ds}.csv"
    if not f.exists():
        print(f"[attenzione] manca {f}: {ds} saltato")
        return None
    d = pd.read_csv(f, dtype={"subject_id": str})
    d["subject_id"] = d.subject_id.str.strip()
    d = d[d.method.isin(_cmp.METODI)]
    if ds == "FBP":          # stessi soggetti del confronto finale
        ref = d[d.method == "synthseg"]
        buoni = set(ref.loc[ref.suvr_dil0 >= suvr_min_fbp, "subject_id"])
        d = d[d.subject_id.isin(buoni)]
    rif = riferimento(ds, qamyloid)
    d["truth_cl"] = d.subject_id.map(rif)
    return d.dropna(subset=["truth_cl"])


def ricalibra(d: pd.DataFrame, col_suvr: str, coeff: tuple | None = None) -> np.ndarray:
    """CL ricalibrato: retta stimata su `d` stesso, oppure `coeff` dato."""
    ok = d[col_suvr].notna()
    out = np.full(len(d), np.nan)
    if ok.sum() < 3:          # troppo pochi punti per una retta affidabile
        return out
    a, b = coeff or _cmp.calibrazione_regressione(d.loc[ok, col_suvr], d.loc[ok, "truth_cl"])
    out[ok.values] = a * d.loc[ok, col_suvr].values + b
    return out


def riga_metriche(ds, met, d, cl0, cl1):
    """Riga del riepilogo: metriche raw e ricalibrate, senza (0) e con (1) dilatazione.

    cl0, cl1  centiloidi ricalibrati senza e con dilatazione; quelli con la
              formula pubblicata si leggono da `d.cl_dil0` e `d.cl_dil1`
    """
    m0 = _cmp.metriche(cl0, d.truth_cl)
    m1 = _cmp.metriche(cl1, d.truth_cl)
    r0 = _cmp.metriche(d.cl_dil0, d.truth_cl)
    r1 = _cmp.metriche(d.cl_dil1, d.truth_cl)
    dmae = m1["mae"] - m0["mae"]
    return {"Dataset": ds, "Metodo": met, "Categoria": _cmp.CATEGORIA[met],
            "N": m0["n"],
            "MAE raw dil0": r0["mae"], "MAE raw dil1": r1["mae"],
            "Bias raw dil0": r0["bias"], "Bias raw dil1": r1["bias"],
            "MAE dil0": m0["mae"], "MAE dil1": m1["mae"], "ΔMAE": dmae,
            "RMSE dil0": m0["rmse"], "RMSE dil1": m1["rmse"],
            "r dil0": m0["r"], "r dil1": m1["r"],
            "ΔSUVr medio": float((d.suvr_dil1 - d.suvr_dil0).mean()),
            "Δvoxel target %": float(100 * (d.ctxvox_dil1 / d.ctxvox_dil0 - 1).mean()),
            # tolleranza minima per non contare come differenze gli errori di
            # arrotondamento in virgola mobile
            "Esito": ("migliora" if dmae < -1e-9 else
                      "peggiora" if dmae > 1e-9 else "uguale")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qamyloid", default=str(percorsi.QAMYLOID_DLBS))
    ap.add_argument("--fbp-suvr-min", type=float, default=_cmp.SUVR_MIN_FBP)
    ap.add_argument("--out", default=str(percorsi.CONFRONTO /
                                         "confronto_dilatazione_centiloide_sintesi.xlsx"))
    a = ap.parse_args()

    # 1. Lettura dei quattro archivi (quelli mancanti restano None e si saltano).
    dati = {ds: leggi(ds, Path(a.qamyloid), a.fbp_suvr_min) for ds in ARCHIVI}
    riep, trasf, per_sogg = [], [], {}
    for ds, d in dati.items():
        if d is None:
            continue
        fogli = []
        # 2. Per ogni metodo, una retta per il SUVr senza dilatazione e una per
        #    quello con dilatazione, stimate sullo stesso archivio.
        for met in _cmp.METODI:
            dm = d[d.method == met].dropna(subset=["suvr_dil0", "suvr_dil1"]).copy()
            if len(dm) < 3:
                continue
            dm["cl_rec0"] = ricalibra(dm, "suvr_dil0")
            dm["cl_rec1"] = ricalibra(dm, "suvr_dil1")
            riep.append(riga_metriche(ds, met, dm, dm.cl_rec0, dm.cl_rec1))
            fogli.append(dm)
            # 3. Solo DLBS: coefficienti stimati sul florbetapir GAAIN (stesso
            #    metodo, stessa dilatazione) e applicati invariati al DLBS.
            if ds == "DLBS" and dati.get("FBP") is not None:
                fbp = dati["FBP"][dati["FBP"].method == met].dropna(
                    subset=["suvr_dil0", "suvr_dil1"])
                c0 = _cmp.calibrazione_regressione(fbp.suvr_dil0, fbp.truth_cl)
                c1 = _cmp.calibrazione_regressione(fbp.suvr_dil1, fbp.truth_cl)
                # tutti i 226 soggetti e, come nella validazione, i 219 senza
                # i sette con riferimento inaffidabile
                for campione, sel in (("tutti", dm),
                                      ("senza 7 QC",
                                       dm[~dm.subject_id.isin(_cmp.DLBS_QC_ESCLUSI)])):
                    riga = riga_metriche(ds, met, sel, ricalibra(sel, "suvr_dil0", c0),
                                         ricalibra(sel, "suvr_dil1", c1))
                    trasf.append({"Campione": campione, **riga})
        per_sogg[ds] = pd.concat(fogli, ignore_index=True)

    # 4. Sintesi per archivio: MAE medio sui metodi e quanti metodi migliorano
    #    o peggiorano con la dilatazione.
    R = pd.DataFrame(riep)
    T = pd.DataFrame(trasf)
    sint = (R.groupby("Dataset", sort=False)
             .agg(**{"MAE dil0 (media)": ("MAE dil0", "mean"),
                     "MAE dil1 (media)": ("MAE dil1", "mean"),
                     "ΔMAE medio": ("ΔMAE", "mean"),
                     "n migliora": ("Esito", lambda s: int((s == "migliora").sum())),
                     "n peggiora": ("Esito", lambda s: int((s == "peggiora").sum()))})
             .reset_index())

    # 5. Excel con sintesi, riepilogo, foglio DLBS e dettaglio per soggetto,
    #    poi stampa a terminale delle colonne principali.
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        sint.round(3).to_excel(xw, sheet_name="Sintesi", index=False)
        R.round(3).to_excel(xw, sheet_name="Riepilogo", index=False)
        if not T.empty:
            T.round(3).to_excel(xw, sheet_name="DLBS coeff. GAAIN", index=False)
        for ds, d in per_sogg.items():
            d.round(4).to_excel(xw, sheet_name=f"{ds} per soggetto", index=False)

    pd.set_option("display.width", 200)
    print(R[["Dataset", "Metodo", "N", "MAE raw dil0", "MAE raw dil1",
             "MAE dil0", "MAE dil1", "ΔMAE", "ΔSUVr medio", "Esito"]]
          .round(2).to_string(index=False))
    print("\n" + sint.round(2).to_string(index=False))
    if not T.empty:
        print("\nDLBS con i coefficienti stimati sul florbetapir GAAIN:")
        print(T[["Campione", "Metodo", "N", "MAE dil0", "MAE dil1", "ΔMAE"]]
              .round(2).to_string(index=False))
    print(f"\nScritto {out}")


if __name__ == "__main__":
    main()
