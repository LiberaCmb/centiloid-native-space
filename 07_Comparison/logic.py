"""Raccolta dei risultati, confronto con i valori di riferimento e ricalibrazione.

Il modulo legge i `result.csv` scritti dal modulo 06 per ogni combinazione di
soggetto, metodo e tracciante, li riunisce in un'unica tabella, esclude i
soggetti che il controllo qualita' dichiara inutilizzabili e affianca a ogni
stima il centiloide di riferimento pubblicato con l'archivio.

Ricalibrazione
--------------
Il SUVr misurato in spazio nativo non e' quello su cui sono state stimate le
formule pubblicate, quindi la retta CL = a*SUVr + b va ristimata per ogni
coppia metodo-tracciante. Due strategie:

  regressione   minimi quadrati su tutti i soggetti, contro il loro
                centiloide di riferimento. E' la calibrazione adottata.
  two-point     la retta per i due baricentri (SUVr medio, CL medio) dei
                controlli giovani e degli anziani: la procedura originale del
                Centiloid Project, riportata come termine di paragone.

Le funzioni sono usate dagli script di `scripts/analisi` e `scripts/figure`;
in particolare `tabelle_confronto` da `build_final_comparison.py`, che stima
i coefficienti ricalibrati.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd

# ── Metodi della tesi, nell'ordine di presentazione ─────────────────────────
METODI = ["synthseg", "fastsurfer", "samseg", "samseg_atlas",
          "otsu", "kmeans", "fsl_fast"]

# Famiglia di ciascun metodo, riportata nella colonna "Categoria" del riepilogo.
CATEGORIA = {
    "synthseg": "AI FreeSurfer",
    "fastsurfer": "AI FreeSurfer",
    "samseg": "AI FreeSurfer",
    "samseg_atlas": "AI + atlante",
    "otsu": "Classico+atlante",
    "kmeans": "Classico+atlante",
    "fsl_fast": "Classico+atlante",
}

# ── Dove stanno i risultati di ciascun archivio (relativi alla cartella dati) ─
# I metodi a etichette e samseg_atlas nella cartella *_ants, i classici (VOI
# dall'atlante) nella cartella *_atlas. Allineamento ANTs per tutti.
CARTELLE_RISULTATI = {
    "FBB": ["results_fbb_ants", "results_fbb_atlas"],
    "FMM": ["results_fmm_ants", "results_fmm_atlas"],
    "FBP": ["results_fbp_ants", "results_fbp_atlas"],
    "DLBS": ["results_dlbs"],
}
# File con i centiloidi di riferimento degli archivi GAAIN (relativi alla
# cartella dati), letti con `leggi_ground_truth`.
GROUND_TRUTH = {
    "FBB": "gaain_fbb.csv",
    "FMM": "gaain_fmm.csv",
    "FBP": "gaain_fbp.csv",
}

# ── Controllo qualita' ──────────────────────────────────────────────────────
# Florbetapir GAAIN: 14 dei 46 soggetti sono inutilizzabili. Uno (181120) non
# produce risultato; 13 hanno PET con una scala di intensita' anomala e SUVr
# fra 0.59 e 0.81, non fisiologici. Il taglio a SUVr >= 0.85 sul metodo di
# riferimento li separa nettamente (il primo soggetto valido sta a 0.922).
# E' una scelta operativa, non un dato: lo script la stampa a ogni esecuzione.
SUVR_MIN_FBP = 0.85
# metodo il cui SUVr decide l'esclusione (per tutti i metodi insieme)
METODO_RIFERIMENTO_QC = "synthseg"

# DLBS: soggetti con valore di RIFERIMENTO inaffidabile. Su questi sette tutti
# i metodi, che sbagliano in modo indipendente, si discostano dal riferimento
# nello stesso verso e di molto (errore mediano 11-17 CL contro i ~3 tipici):
# quando metodi indipendenti concordano fra loro e divergono dal riferimento,
# il sospetto cade sul riferimento.
DLBS_QC_ESCLUSI = ["sub-1007", "sub-1225", "sub-1600", "sub-2035",
                   "sub-4288", "sub-4467", "sub-4507"]


def formule_pubblicate() -> dict:
    """Le formule pubblicate dei traccianti, dall'unica fonte (modulo 06)."""
    return dict(importlib.import_module("06_CentiloidCalculation").CL_FORMULAS)


# ════════════════════════════════════════════════════════════════════════════
# Gruppi per la calibrazione two-point
# ════════════════════════════════════════════════════════════════════════════
def gruppo_soggetto(sid: str) -> str:
    """'YC' (controllo giovane), 'AD' (anziano/paziente) o '' se non deducibile.

    Convenzioni degli archivi GAAIN:
        FBB   giovani 'Y1001', ...      anziani '1008', '2032' (solo cifre)
        FMM   giovani 'YHV_001', ...    anziani 'AD_100', ...
    Il florbetapir ha identificativi numerici per tutti: il gruppo si legge
    dalla colonna `group` del manifest (`gruppi_da_manifest`).
    """
    s = str(sid).strip().upper()
    if not s:
        return ""
    if s.startswith("YHV") or s.startswith("YC"):   # prima della regola 'Y'+cifre
        return "YC"
    if s.startswith("AD"):
        return "AD"
    if s[0] == "Y" and len(s) > 1 and s[1].isdigit():
        return "YC"
    if s.isdigit():
        return "AD"
    return ""


def gruppi_da_manifest(path: Path) -> dict:
    """{soggetto: 'YC' | 'AD'} dalla colonna `group` di un manifest, se c'e'."""
    if not Path(path).is_file():
        return {}
    try:
        m = pd.read_csv(path)
    except Exception:
        return {}
    if "group" not in m.columns or "subject_id" not in m.columns:
        return {}
    # Tutto cio' che non e' marcato come controllo giovane conta come anziano.
    fuori = {}
    for _, r in m.iterrows():
        g = str(r["group"]).strip().upper()
        fuori[str(r["subject_id"]).strip()] = (
            "YC" if g in ("YC", "YHC", "YHV") else "AD")
    return fuori


# ════════════════════════════════════════════════════════════════════════════
# Lettura dei risultati
# ════════════════════════════════════════════════════════════════════════════
def raccogli_risultati(cartelle) -> pd.DataFrame:
    """Legge ogni <cartella>/<soggetto>/<metodo>/result.csv in un'unica tabella.

    Cartelle assenti e file illeggibili vengono segnalati e saltati; senza
    alcun risultato si restituisce un DataFrame vuoto.
    """
    frames = []
    for base in cartelle:
        base = Path(base)
        if not base.is_dir():
            print(f"  [attenzione] cartella assente: {base}")
            continue
        for subj_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            for meth_dir in sorted(p for p in subj_dir.iterdir() if p.is_dir()):
                f = meth_dir / "result.csv"
                if not f.is_file():
                    continue
                try:
                    frames.append(pd.read_csv(f))
                except Exception as e:
                    print(f"  [attenzione] {f} illeggibile: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # identificativi come stringhe ripulite, per unire con il riferimento
    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    df["segmenter"] = df["segmenter"].astype(str).str.strip()
    return df


def leggi_ground_truth(path) -> pd.DataFrame:
    """CSV di riferimento con colonne subject_id, truth_suvr, truth_cl."""
    gt = pd.read_csv(path)
    gt["subject_id"] = gt["subject_id"].astype(str).str.strip()
    return gt


def filtro_qualita(df: pd.DataFrame, suvr_min: float, etichetta: str,
                   metodo_rif: str = METODO_RIFERIMENTO_QC) -> pd.DataFrame:
    """Tiene i soggetti il cui SUVr, misurato con `metodo_rif`, e' plausibile.

    Esclude per soggetto (tutti i metodi insieme), cosi' il confronto resta
    appaiato. Stampa l'elenco degli esclusi con il loro SUVr.

    df           tabella di `raccogli_risultati`
    suvr_min     SUVr minimo ammesso (per il florbetapir SUVR_MIN_FBP)
    etichetta    nome del dataset, solo per i messaggi
    metodo_rif   metodo su cui si misura il SUVr

    Un soggetto senza risultato per `metodo_rif` viene escluso.
    """
    ref = df[df.segmenter == metodo_rif]
    tutti = set(df.subject_id)
    buoni = set(ref.loc[ref.suvr >= suvr_min, "subject_id"])
    esclusi = sorted(tutti - buoni)
    if esclusi:
        print(f"  [{etichetta}] esclusi {len(esclusi)} soggetti su {len(tutti)} "
              f"(SUVr {metodo_rif} < {suvr_min} o risultato assente):")
        valori = ref.set_index("subject_id")["suvr"].to_dict()
        for s in esclusi:
            v = valori.get(s)
            print(f"      {s}: SUVr = " + ("assente" if v is None or pd.isna(v)
                                           else f"{v:.3f}"))
    return df[df.subject_id.isin(buoni)].copy()


# ════════════════════════════════════════════════════════════════════════════
# Metriche e calibrazioni
# ════════════════════════════════════════════════════════════════════════════
def metriche(stima, riferimento) -> dict:
    """Bias medio, errore assoluto medio, RMSE e correlazione di Pearson."""
    p = np.asarray(stima, float)
    t = np.asarray(riferimento, float)
    # si confrontano solo le coppie con entrambi i valori finiti (no NaN)
    ok = np.isfinite(p) & np.isfinite(t)
    p, t = p[ok], t[ok]
    if len(p) == 0:
        return dict(n=0, bias=np.nan, mae=np.nan, rmse=np.nan, r=np.nan)
    # errore con segno: bias positivo = sovrastima rispetto al riferimento
    e = p - t
    # Pearson solo con almeno tre punti e varianza non nulla, altrimenti NaN
    r = (np.corrcoef(p, t)[0, 1]
         if len(p) > 2 and p.std() > 0 and t.std() > 0 else np.nan)
    return dict(n=int(len(p)), bias=float(e.mean()),
                mae=float(np.abs(e).mean()),
                rmse=float(np.sqrt((e ** 2).mean())), r=float(r))


def calibrazione_regressione(suvr, cl_rif) -> tuple[float, float]:
    """(a, b) ai minimi quadrati di CL_rif = a*SUVr + b."""
    a, b = np.polyfit(np.asarray(suvr, float), np.asarray(cl_rif, float), 1)
    return float(a), float(b)


def calibrazione_two_point(suvr, cl_rif, gruppi) -> tuple[float, float] | None:
    """(a, b) della retta per i baricentri dei controlli giovani e degli anziani.

    Restituisce None se uno dei due gruppi ha meno di due soggetti.
    """
    suvr = np.asarray(suvr, float)
    cl_rif = np.asarray(cl_rif, float)
    g = np.asarray(gruppi)
    yc, ad = g == "YC", g == "AD"
    if yc.sum() < 2 or ad.sum() < 2:
        return None
    # baricentri dei due gruppi nel piano (SUVr, CL)
    sy, sa = suvr[yc].mean(), suvr[ad].mean()
    cy, ca = cl_rif[yc].mean(), cl_rif[ad].mean()
    # SUVr medi coincidenti: retta verticale, pendenza non definita
    if abs(sa - sy) <= 1e-9:
        return None
    a = (ca - cy) / (sa - sy)
    return float(a), float(cy - a * sy)


def leggi_coefficienti(path, metodo=None, tracciante=None,
                       calibrazione="regressione") -> dict:
    """Legge `recalibration_final.csv`.

    Restituisce {(tracciante, metodo): (a, b)}, filtrato per gli argomenti
    indicati.
    """
    d = pd.read_csv(path)
    d = d[d["calibrazione"].astype(str).str.strip() == calibrazione]
    if metodo is not None:
        d = d[d["metodo"].astype(str).str.strip() == metodo]
    if tracciante is not None:
        d = d[d["tracciante"].astype(str).str.upper() == tracciante.upper()]
    return {(str(r["tracciante"]).upper(), str(r["metodo"]).strip()):
            (float(r["a"]), float(r["b"])) for _, r in d.iterrows()}


# ════════════════════════════════════════════════════════════════════════════
# Tabelle del confronto finale
# ════════════════════════════════════════════════════════════════════════════
def tabelle_confronto(per_dataset: dict, gruppi_extra: dict | None = None):
    """Calcola, per ogni tracciante e metodo, le tre versioni del centiloide
    (formula pubblicata, two-point, regressione) e le loro metriche.

    per_dataset    {tracciante: DataFrame con subject_id, segmenter, suvr,
                   centiloid, truth_cl}
    gruppi_extra   {tracciante: {soggetto: 'YC' | 'AD'}} letti dai manifest;
                   dove manca si deduce il gruppo dall'identificativo

    Restituisce (riepilogo, {tracciante: foglio per soggetto}, coefficienti).
    Le metriche della regressione sono misurate sugli stessi soggetti usati
    per stimarla (adattamento, non validazione).
    """
    summary, sheets, coeffs = [], {}, []
    gruppi_extra = gruppi_extra or {}

    for ds, m in per_dataset.items():
        # 1. Foglio per soggetto: una riga per soggetto con il riferimento,
        #    a cui si aggiungono le colonne dei metodi.
        mappa_gruppi = gruppi_extra.get(ds, {})
        subjects = (m[["subject_id", "truth_cl"]]
                    .drop_duplicates("subject_id")
                    .sort_values("subject_id")
                    .reset_index(drop=True))
        sheet = pd.DataFrame({"subject_id": subjects.subject_id,
                              "ground_truth": subjects.truth_cl.round(1)})

        for meth in METODI:
            s = m[m.segmenter == meth].dropna(subset=["suvr", "truth_cl"])
            # con meno di tre punti la retta e la correlazione non hanno senso
            if len(s) < 3:
                print(f"  [{ds}] metodo {meth}: solo {len(s)} soggetti, saltato")
                continue
            suvr = s.suvr.values.astype(float)
            truth = s.truth_cl.values.astype(float)

            # 2. Calibrazione per regressione (sempre possibile qui).
            a, b = calibrazione_regressione(suvr, truth)
            coeffs.append(dict(tracciante=ds, metodo=meth,
                               calibrazione="regressione",
                               a=round(a, 4), b=round(b, 4),
                               n=int(len(s)), allineamento="ANTs"))

            # 3. Calibrazione two-point: gruppo dal manifest, altrimenti
            #    dall'identificativo. La colonna n riporta "n_YC+n_AD".
            g = s.subject_id.map(
                lambda x: mappa_gruppi.get(str(x).strip()) or gruppo_soggetto(x))
            tp = calibrazione_two_point(suvr, truth, g.values)
            if tp is not None:
                a2, b2 = tp
                coeffs.append(dict(tracciante=ds, metodo=meth,
                                   calibrazione="two-point",
                                   a=round(a2, 4), b=round(b2, 4),
                                   n=f"{int((g == 'YC').sum())}+{int((g == 'AD').sum())}",
                                   allineamento="ANTs"))
            else:
                print(f"  [{ds}] {meth}: poli troppo scarsi "
                      f"(YC={int((g == 'YC').sum())}, AD={int((g == 'AD').sum())}), "
                      "two-point saltata")

            # 4. Centiloidi nelle tre versioni (raw = formula pubblicata, gia'
            #    nel result.csv) e loro colonne nel foglio per soggetto.
            raw = s.centiloid.astype(float).values
            reg = a * suvr + b
            sheet[f"{meth}\nraw"] = sheet.subject_id.map(dict(zip(s.subject_id, raw))).round(1)
            sheet[f"{meth}\nreg"] = sheet.subject_id.map(dict(zip(s.subject_id, reg))).round(1)
            calibrazioni = [("raw", raw), ("regressione", reg)]
            if tp is not None:
                two = a2 * suvr + b2
                sheet[f"{meth}\n2pt"] = sheet.subject_id.map(dict(zip(s.subject_id, two))).round(1)
                calibrazioni.append(("two-point", two))

            # 5. Metriche di ciascuna versione contro il riferimento.
            for cal, pred in calibrazioni:
                mt = metriche(pred, truth)
                summary.append({
                    "Dataset": ds, "Metodo": meth, "Categoria": CATEGORIA[meth],
                    "Calibrazione": cal, "N": mt["n"],
                    "Bias (CL)": round(mt["bias"], 2),
                    "MAE (CL)": round(mt["mae"], 2),
                    "RMSE (CL)": round(mt["rmse"], 2),
                    "Pearson r": round(mt["r"], 3),
                    "Allineamento": "ANTs"})

        sheets[ds] = sheet

    return pd.DataFrame(summary), sheets, pd.DataFrame(coeffs)
