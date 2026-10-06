"""Tempi di segmentazione, ricalcolati dai result.csv dei run finali.

Ogni result.csv scritto dal modulo 06 porta con se' `seg_runtime_s`, il tempo
impiegato dal modulo 04 su quel soggetto. Questo script li raccoglie tutti e
produce la tabella dei costi computazionali, cosi' che i tempi riportati in
tesi vengano dagli stessi run dei risultati e non da esecuzioni precedenti.

Uso:  python -m scripts.analisi.report_runtimes_final
Uscite (in data/comparison_all_methods/): tempi_per_dataset.csv,
tempi_sintesi.csv e tempi_calcolo.xlsx (fogli "Sintesi", "Per dataset",
"Per soggetto").
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared.percorsi import risolvi  # noqa: E402

# Cartelle dei run finali: quelle e nient'altro.
SORGENTI = {
    "FBB":  ["data/results_fbb_ants", "data/results_fbb_atlas"],
    "FMM":  ["data/results_fmm_ants", "data/results_fmm_atlas"],
    "FBP":  ["data/results_fbp_ants", "data/results_fbp_atlas"],
    "DLBS": ["data/results_dlbs"],
}
# Famiglia di ciascun metodo, per raggruppare le righe della tabella.
FAMIGLIA = {"synthseg": "apprendimento", "fastsurfer": "apprendimento",
            "samseg": "bayesiano", "samseg_atlas": "bayesiano",
            "otsu": "classico", "kmeans": "classico", "fsl_fast": "classico",
            "recon_all": "riferimento"}


def raccogli() -> pd.DataFrame:
    """Un record (dataset, metodo, soggetto, secondi) per ogni result.csv valido."""
    righe = []
    for dataset, dirs in SORGENTI.items():
        for d in dirs:
            base = risolvi(d)
            if not base.exists():
                continue
            for sub in sorted(p for p in base.iterdir() if p.is_dir()):
                for met in sorted(p for p in sub.iterdir() if p.is_dir()):
                    f = met / "result.csv"
                    if not f.exists():
                        continue
                    try:
                        r = pd.read_csv(f).iloc[0]
                    except Exception:
                        continue
                    t = r.get("seg_runtime_s")
                    if pd.isna(t):
                        continue
                    righe.append({"dataset": dataset, "metodo": met.name,
                                  "soggetto": sub.name, "secondi": float(t)})
    return pd.DataFrame(righe)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/comparison_all_methods")
    a = ap.parse_args()

    d = raccogli()
    if d.empty:
        print("nessun result.csv trovato")
        return 1
    print(f"{len(d)} esecuzioni lette da {d.soggetto.nunique()} soggetti\n")

    # Statistiche per dataset e metodo, piu' il tempo complessivo in ore.
    g = (d.groupby(["dataset", "metodo"])["secondi"]
           .agg(n="count", mediana="median", media="mean",
                minimo="min", massimo="max", dev_std="std")
           .reset_index())
    g["ore_totali"] = (d.groupby(["dataset", "metodo"])["secondi"].sum()
                        .values / 3600)
    g["famiglia"] = g.metodo.map(FAMIGLIA)
    g = g[["dataset", "famiglia", "metodo", "n", "mediana", "media",
           "minimo", "massimo", "dev_std", "ore_totali"]].round(2)

    # sintesi su tutti i dataset: la mediana e' la statistica da citare,
    # la media e' gonfiata dai primi run (download dei modelli).
    tot = (d.groupby("metodo")["secondi"]
             .agg(n="count", mediana="median", media="mean",
                  minimo="min", massimo="max").reset_index())
    tot["famiglia"] = tot.metodo.map(FAMIGLIA)
    tot = tot[["famiglia", "metodo", "n", "mediana", "media",
               "minimo", "massimo"]].sort_values("mediana").round(2)

    outdir = risolvi(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    g.to_csv(outdir / "tempi_per_dataset.csv", index=False)
    tot.to_csv(outdir / "tempi_sintesi.csv", index=False)
    with pd.ExcelWriter(outdir / "tempi_calcolo.xlsx", engine="openpyxl") as xw:
        tot.to_excel(xw, sheet_name="Sintesi", index=False)
        g.to_excel(xw, sheet_name="Per dataset", index=False)
        d.to_excel(xw, sheet_name="Per soggetto", index=False)

    print("NOTA: samseg_atlas riusa la segmentazione di SAMSEG, quindi ha lo")
    print("stesso seg_runtime_s; il costo aggiuntivo e' la registrazione SyN")
    print("dell'atlante, che non entra in questo campo. Vale anche per i metodi")
    print("classici in modalita' atlante.\n")
    print("— sintesi su tutti i dataset (secondi per soggetto) —")
    print(tot.to_string(index=False))
    print("\n— per dataset —")
    print(g.to_string(index=False))
    print(f"\nScritti in {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
