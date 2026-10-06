"""Confronto finale dei sette metodi sui tre archivi GAAIN (risultato principale).

Dai `result.csv` della pipeline costruisce

    data/comparison_all_methods/confronto_metodi_centiloide_ricalibrato_ANTs_finale.xlsx
    data/comparison_all_methods/recalibration_final.csv

Il secondo contiene i coefficienti (a, b) di ricalibrazione per ogni coppia
tracciante-metodo: lo leggono l'interfaccia e gli script di validazione.

Metodologia
-----------
* Per ogni metodo e tracciante tre versioni del centiloide: con la formula
  pubblicata ("raw"), ricalibrata two-point (retta per i baricentri di
  controlli giovani e anziani) e ricalibrata per regressione (minimi quadrati
  su tutti i soggetti). La regressione e' la calibrazione adottata; la
  two-point e' il termine di paragone. In fondo lo script stampa quale delle
  due vince per ogni combinazione.
* Riferimento: i centiloidi pubblicati da GAAIN (data/ground_truth/gaain_*.csv).
* Allineamento ANTs per tutti i metodi.
* Florbetapir: si tengono i soggetti con SUVr di SynthSeg >= 0.85 (32 su 46);
  la soglia e' il parametro --fbp-suvr-min e gli esclusi vengono stampati.

    python -m scripts.analisi.build_final_comparison
    python -m scripts.analisi.build_final_comparison --fbp-suvr-min 0.90
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import percorsi                                    # noqa: E402
_cmp = importlib.import_module("07_Comparison")

OUT_XLSX = "confronto_metodi_centiloide_ricalibrato_ANTs_finale.xlsx"
OUT_COEFFS = "recalibration_final.csv"
ARCHIVI = ("FBB", "FMM", "FBP")


def note(n_fbp: int) -> list[str]:
    """Righe di nota scritte sotto la tabella del foglio "Riepilogo"."""
    f = _cmp.formule_pubblicate()
    formule = ", ".join(f"{t} {a}/{b}" for t, (a, b) in f.items())
    return [
        "NOTE:",
        "• Tre colonne per metodo: grezzo (formula pubblicata), two-point (poli YC/AD), regressione (minimi quadrati).",
        "• La calibrazione di riferimento e' la REGRESSIONE; la two-point e' riportata come termine di paragone.",
        f"• Dataset: FBB, FMM, FBP (GAAIN florbetapir, {n_fbp} puliti). Riferimento: centiloidi GAAIN. Allineamento ANTs.",
        f"• Grezzo = formula pubblicata del tracciante ({formule}).",
        "• Rigenerato da scripts/analisi/build_final_comparison.py: non modificare a mano.",
    ]


def write_xlsx(path: Path, summary, sheets, coeffs, n_fbp: int):
    """Scrive l'Excel del confronto finale con un'impaginazione minima.

    path      file .xlsx di uscita
    summary   riepilogo per archivio, metodo e calibrazione (foglio "Riepilogo")
    sheets    {archivio: tabella per soggetto}, un foglio per archivio
    coeffs    coefficienti di ricalibrazione (foglio "Coefficienti")
    n_fbp     numero di soggetti florbetapir tenuti, citato nelle note
    """
    import pandas as pd
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Riepilogo", index=False)
        ws = xw.sheets["Riepilogo"]
        for i, txt in enumerate(note(n_fbp)):
            ws.cell(row=len(summary) + 4 + i, column=1, value=txt)

        for ds, sheet in sheets.items():
            sheet.to_excel(xw, sheet_name=ds, index=False)

        cf = coeffs.rename(columns={"tracciante": "Dataset", "metodo": "Metodo",
                                    "calibrazione": "Calibrazione",
                                    "a": "a (pendenza)", "b": "b (intercetta)"})
        cf[["Dataset", "Metodo", "Calibrazione", "a (pendenza)",
            "b (intercetta)", "n"]].to_excel(xw, sheet_name="Coefficienti", index=False)

        # Intestazioni in evidenza, prima riga bloccata e larghezza delle
        # colonne stimata sulle prime 60 righe (limitata fra 10 e 46 caratteri).
        head = Font(name="Arial", bold=True, color="FFFFFF", size=10)
        fill = PatternFill("solid", fgColor="1F3864")
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for c in ws[1]:
                if c.value is not None:
                    c.font, c.fill = head, fill
                    c.alignment = Alignment(horizontal="center", wrap_text=True)
            for j in range(1, ws.max_column + 1):
                width = max((len(str(ws.cell(row=r, column=j).value or ""))
                             for r in range(1, min(ws.max_row, 60) + 1)), default=10)
                ws.column_dimensions[get_column_letter(j)].width = min(max(width + 2, 10), 46)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(percorsi.CONFRONTO),
                   help="Cartella di destinazione.")
    p.add_argument("--fbp-suvr-min", type=float, default=_cmp.SUVR_MIN_FBP,
                   help="SUVr minimo di un soggetto FBP valido (predefinito 0.85).")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per ogni archivio: risultati della pipeline, filtro di qualita' sul
    #    florbetapir e unione con i centiloidi GAAIN di riferimento.
    per_dataset, gruppi_extra, n_fbp = {}, {}, 0
    for ds in ARCHIVI:
        print(f"[{ds}] lettura dei risultati…")
        df = _cmp.raccogli_risultati(percorsi.dati(c) for c in _cmp.CARTELLE_RISULTATI[ds])
        if df.empty:
            print(f"  [{ds}] nessun risultato, archivio saltato")
            continue
        if ds == "FBP":
            df = _cmp.filtro_qualita(df, args.fbp_suvr_min, ds)
            n_fbp = df.subject_id.nunique()
            # gli identificativi FBP sono numerici per tutti: i gruppi YC/AD
            # per la two-point si leggono dal manifest
            gruppi_extra[ds] = _cmp.gruppi_da_manifest(
                percorsi.dati("manifest_fbp_nifti.csv"))
            if gruppi_extra[ds]:
                n_yc = sum(1 for v in gruppi_extra[ds].values() if v == "YC")
                print(f"  [{ds}] gruppi dal manifest: {n_yc} controlli giovani, "
                      f"{len(gruppi_extra[ds]) - n_yc} anziani")

        gt = _cmp.leggi_ground_truth(percorsi.GROUND_TRUTH / _cmp.GROUND_TRUTH[ds])
        m = df.merge(gt[["subject_id", "truth_cl"]], on="subject_id", how="inner")
        print(f"  [{ds}] {m.subject_id.nunique()} soggetti con riferimento, "
              f"{m.segmenter.nunique()} metodi")
        per_dataset[ds] = m

    if not per_dataset:
        raise SystemExit("Nessun archivio elaborabile: controllare la cartella "
                         f"dei dati ({percorsi.DATI}).")

    # 2. Tabelle del confronto (raw, two-point, regressione) e coefficienti.
    summary, sheets, coeffs = _cmp.tabelle_confronto(per_dataset, gruppi_extra)

    # 3. Scrittura dell'Excel e del CSV dei coefficienti.
    xlsx = out_dir / OUT_XLSX
    write_xlsx(xlsx, summary, sheets, coeffs, n_fbp)
    csv = out_dir / OUT_COEFFS
    coeffs.to_csv(csv, index=False)
    print(f"\nScritto {xlsx}")
    print(f"Scritto {csv}   <- letto da ui/pipeline_service.py e dalla validazione")

    # 4. Resoconto a terminale: quale ricalibrazione ha il MAE piu' basso per
    #    ogni archivio e metodo, e classifica dei metodi con la regressione.
    print("\nConfronto fra le due ricalibrazioni (MAE in CL):")
    conf = (summary[summary.Calibrazione.isin(["two-point", "regressione"])]
            .pivot_table(index=["Dataset", "Metodo"], columns="Calibrazione",
                         values="MAE (CL)"))
    if {"two-point", "regressione"}.issubset(conf.columns):
        conf["vince"] = np.where(conf["two-point"] < conf["regressione"],
                                 "two-point", "regressione")
        print(conf.round(2).to_string())
        print("\nvittorie:", conf["vince"].value_counts().to_dict())

    print("\nClassifica per MAE (calibrazione per regressione):")
    best = (summary[summary.Calibrazione == "regressione"]
            .sort_values(["Dataset", "MAE (CL)"]))
    print(best[["Dataset", "Metodo", "Categoria", "N",
                "MAE (CL)", "RMSE (CL)", "Pearson r"]].to_string(index=False))


if __name__ == "__main__":
    main()
