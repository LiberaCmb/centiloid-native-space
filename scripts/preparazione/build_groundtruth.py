"""Centiloidi di riferimento dei tre archivi GAAIN, dai fogli Excel pubblicati.

Riferimento: regione di normalizzazione = cervelletto intero (WC), come nella
pipeline. Per ogni tracciante si scrive data/ground_truth/gaain_<tr>.csv con
le colonne subject_id, truth_suvr, truth_cl.

  FBB  FBBproject_SupplementaryTable.xlsx, foglio '18F-FBB':
       colonna 0 = soggetto, 3 = SUVr WC, 7 = centiloide WC.
       Le righe di intestazione e i titoli di sezione ('Elderly', 'Mean', ...)
       si scartano: si tengono gli identificativi numerici ('1008') e quelli
       dei controlli giovani ('Y1001').
  FMM  GEHealthcare[18F]flutemetamol.xlsx, foglio 'GE PiB & Flutemetamol',
       dalla riga 3: colonna 2 = soggetto ('AD_100', 'YHV_001'),
       16 = SUVr flutemetamol WC, 19 = centiloide (convertito via PiB) WC.
  FBP  Avid_Centiloid_standard_method.xlsx, dalla riga 2: colonna 0 =
       soggetto, 3 = SUVr florbetapir, 8 = centiloide florbetapir.

    python -m scripts.preparazione.build_groundtruth            # tutti e tre
    python -m scripts.preparazione.build_groundtruth --tracer FBP
"""
from __future__ import annotations
import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared import percorsi                        # noqa: E402

# Per ogni tracciante: cartella dell'archivio GAAIN, file Excel e foglio,
# prima riga utile e indici (da 0) delle colonne id / SUVr / centiloide.
# `formato` None = valori scritti a piena precisione, nell'ordine del foglio.
FONTI = {
    "FBB": dict(archivio="set di dati PIB da 50-70 minuti e FBB da 90-110 minuti (dati DICOM)",
                file="FBBproject_SupplementaryTable.xlsx", foglio="18F-FBB",
                prima_riga=0, col_id=0, col_suvr=3, col_cl=7, formato="%.6f"),
    "FMM": dict(archivio="Set di dati PIBflutemetamolo da 50-70 minuti (dati NIFTI e DICOM)",
                file="GEHealthcare[18F]flutemetamol.xlsx", foglio="GE PiB & Flutemetamol",
                prima_riga=3, col_id=2, col_suvr=16, col_cl=19, formato="%.6f"),
    "FBP": dict(archivio="50-60 minute florbetapir and 50-70 minute PiB dataset DICOM data",
                file="Avid_Centiloid_standard_method.xlsx", foglio=0,
                prima_riga=2, col_id=0, col_suvr=3, col_cl=8, formato=None),
}


def _id_valido(raw, tracciante) -> str | None:
    """Identificativo del soggetto dalla cella, o None se la riga va scartata."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    if re.match(r"^\d+\.0$", s):              # '1008.0' -> '1008'
        s = s.split(".")[0]
    if not s:
        return None
    if tracciante == "FBB":                   # scarta titoli di sezione
        if s[0].isdigit() or (s[0] in "Yy" and len(s) > 1 and s[1].isdigit()):
            return s
        return None
    return s


def estrai(tr: str) -> list[tuple[str, float, float]]:
    """Righe (soggetto, SUVr, centiloide) dal foglio Excel del tracciante.

    Le righe senza identificativo valido o con valori non numerici si saltano.
    """
    f = FONTI[tr]
    src = percorsi.DATASETS / f["archivio"] / f["file"]
    if not src.exists():
        sys.exit(f"File non trovato: {src}")
    df = pd.read_excel(src, sheet_name=f["foglio"], header=None)
    righe = []
    for i in range(f["prima_riga"], len(df)):
        sid = _id_valido(df.iat[i, f["col_id"]], tr)
        if sid is None:
            continue
        try:
            suvr, cl = float(df.iat[i, f["col_suvr"]]), float(df.iat[i, f["col_cl"]])
        except (TypeError, ValueError):
            continue
        if pd.isna(suvr) or pd.isna(cl):
            continue
        righe.append((sid, suvr, cl))
    return righe


def main():
    """Scrive un gaain_<tracciante>.csv per ciascun tracciante richiesto."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracer", choices=list(FONTI), nargs="*", default=list(FONTI))
    ap.add_argument("--out-dir", default=str(percorsi.GROUND_TRUTH))
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for tr in a.tracer:
        righe = estrai(tr)
        fmt = FONTI[tr]["formato"]
        if fmt:                               # FBB e FMM: ordinati, 6 decimali
            righe = sorted(righe)
        out = out_dir / f"gaain_{tr.lower()}.csv"
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["subject_id", "truth_suvr", "truth_cl"])
            for sid, suvr, cl in righe:
                w.writerow([sid, fmt % suvr if fmt else repr(suvr),
                            fmt % cl if fmt else repr(cl)])
        print(f"[{tr}] {len(righe)} soggetti -> {out}")


if __name__ == "__main__":
    main()
