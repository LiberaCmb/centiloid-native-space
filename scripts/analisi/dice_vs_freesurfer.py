"""Sovrapposizione delle VOI con la segmentazione di riferimento (DLBS).

Sul solo archivio di validazione esiste, per quasi tutti i soggetti, la
parcellazione anatomica di FreeSurfer. Qui si misura quanto le VOI di ciascun
metodo si sovrappongano a quelle costruite su di essa, con il coefficiente di
Dice e l'indice di Jaccard.

(Sul disco la cartella di FreeSurfer per ogni soggetto si chiama `recon_all`,
dal nome del comando che la produce: e' solo il nome della directory.)

Il numero va letto in due gruppi distinti, perche' misura due cose diverse:

  * metodi a etichette (synthseg, fastsurfer, samseg): confronto simmetrico,
    entrambe le VOI puntano allo stesso bersaglio anatomico. E' una vera
    differenza di segmentazione.
  * metodi ad atlante (samseg_atlas, otsu, kmeans, fsl_fast): la loro VOI e'
    per costruzione un sottoinsieme della VOI atlante standard, quindi il
    confronto misura la COPERTURA del bersaglio da parte del tessuto
    segmentato. Per questi si riporta anche il Dice rispetto alla VOI atlante,
    che si ricava dai conteggi in info.json senza rileggere le maschere:
        Dice = 2 * ctx_voxels / (ctx_voxels + ctx_atlas_only_voxels)
    perche' la VOI del metodo e' contenuta in quella atlante.

Uso:  python -m scripts.analisi.dice_vs_freesurfer
Uscite (in data/comparison_all_methods/): dice_per_soggetto.csv,
dice_copertura.csv, dice_sintesi.csv e dice_vs_freesurfer.xlsx. I due CSV
per soggetto fanno anche da punto di ripresa: rilanciando si salta chi c'e'.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError:
    sys.exit("serve nibabel:  pip install nibabel")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared.percorsi import risolvi  # noqa: E402

A_ETICHETTE = ["synthseg", "fastsurfer", "samseg"]
AD_ATLANTE = ["samseg_atlas", "otsu", "kmeans", "fsl_fast"]
# Nome della sottocartella di FreeSurfer dentro results_dlbs/<soggetto>/
CARTELLA_RIFERIMENTO = "recon_all"
RIFERIMENTO = "FreeSurfer"


def maschera(path: Path) -> np.ndarray | None:
    """Maschera booleana da un NIfTI; None se manca o non si legge."""
    if not path.exists():
        return None
    try:
        return np.asanyarray(nib.load(str(path)).dataobj) > 0
    except Exception as e:
        print(f"  [salto] {path}: {e}")
        return None


def dice_jaccard(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(Dice, Jaccard) di due maschere; NaN se le griglie differiscono o sono vuote."""
    if a.shape != b.shape:
        return float("nan"), float("nan")
    inter = int(np.logical_and(a, b).sum())
    sa, sb = int(a.sum()), int(b.sum())
    if sa + sb == 0:
        return float("nan"), float("nan")
    unione = sa + sb - inter
    return 2 * inter / (sa + sb), (inter / unione if unione else float("nan"))


def voi(sub: Path, metodo: str, quale: str) -> Path:
    """Percorso della VOI `quale` ('ctx' o 'wc') di un metodo per un soggetto."""
    return sub / metodo / "voi" / f"{metodo}_voi_{quale}.nii.gz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/results_dlbs")
    ap.add_argument("--out", default="data/comparison_all_methods")
    ap.add_argument("--max-soggetti", type=int, default=0,
                    help="elabora al piu' N soggetti per esecuzione (0 = tutti). "
                         "Serve solo per spezzare l'elaborazione: i soggetti "
                         "gia' presenti nel CSV vengono saltati.")
    a = ap.parse_args()

    base = risolvi(a.results)
    outdir = risolvi(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    f_sog, f_cop = outdir / "dice_per_soggetto.csv", outdir / "dice_copertura.csv"

    # 1. Ripresa: si ricaricano i risultati gia' scritti e si elaborano solo i
    #    soggetti che mancano (al piu' --max-soggetti).
    soggetti = sorted(p for p in base.iterdir()
                      if p.is_dir() and p.name.startswith("sub-"))
    senza_riferimento = []
    righe = (pd.read_csv(f_sog).to_dict("records") if f_sog.exists() else [])
    copertura = (pd.read_csv(f_cop).to_dict("records") if f_cop.exists() else [])
    fatti = {r["soggetto"] for r in righe}
    restano = [s for s in soggetti if s.name not in fatti]
    if a.max_soggetti:
        restano = restano[:a.max_soggetti]
    print(f"{len(soggetti)} soggetti in {a.results}; "
          f"gia' elaborati {len(fatti)}, ne elaboro {len(restano)}\n")
    # 2. Per ogni soggetto con parcellazione FreeSurfer: Dice e Jaccard di
    #    ogni metodo sulle due VOI.
    for i, sub in enumerate(restano, 1):
        rif_ctx = maschera(voi(sub, CARTELLA_RIFERIMENTO, "ctx"))
        rif_wc = maschera(voi(sub, CARTELLA_RIFERIMENTO, "wc"))
        if rif_ctx is None or rif_wc is None:
            senza_riferimento.append(sub.name)
            continue
        for met in A_ETICHETTE + AD_ATLANTE:
            m_ctx, m_wc = maschera(voi(sub, met, "ctx")), maschera(voi(sub, met, "wc"))
            if m_ctx is None or m_wc is None:
                continue
            d_ctx, j_ctx = dice_jaccard(m_ctx, rif_ctx)
            d_wc, j_wc = dice_jaccard(m_wc, rif_wc)
            righe.append({"soggetto": sub.name, "metodo": met,
                          "dice_ctx": d_ctx, "jaccard_ctx": j_ctx,
                          "dice_wc": d_wc, "jaccard_wc": j_wc})
        # copertura della VOI atlante, dai conteggi gia' salvati
        for met in AD_ATLANTE:
            f = sub / met / "info.json"
            if not f.exists():
                continue
            try:
                v = json.load(open(f))["voi"]
                c, ca = v["ctx_voxels"], v["ctx_atlas_only_voxels"]
                w, wa = v["wc_voxels"], v["wc_atlas_only_voxels"]
            except Exception:
                continue
            copertura.append({"soggetto": sub.name, "metodo": met,
                              "dice_ctx_vs_atlante": 2 * c / (c + ca) if c + ca else np.nan,
                              "dice_wc_vs_atlante": 2 * w / (w + wa) if w + wa else np.nan})
        if i % 25 == 0:
            print(f"  ...{i}/{len(restano)}", flush=True)

    # 3. Salvataggio dei per-soggetto e avvisi sui soggetti esclusi o mancanti.
    d = pd.DataFrame(righe)
    cop = pd.DataFrame(copertura)
    if d.empty:
        print("nessun confronto possibile")
        return 1
    d.to_csv(f_sog, index=False)
    if not cop.empty:
        cop.to_csv(f_cop, index=False)
    mancano = len(soggetti) - d.soggetto.nunique() - len(senza_riferimento)
    if senza_riferimento:
        print(f"\n{len(senza_riferimento)} soggetti senza parcellazione "
              f"FreeSurfer, esclusi dal confronto: "
              f"{', '.join(sorted(senza_riferimento))}")
    if mancano > 0:
        print(f"\nATTENZIONE: mancano ancora {mancano} soggetti. "
              f"Rilancia lo script per continuare da dove si e' fermato.")

    # 4. Sintesi per metodo (media, dev.std, numero), separando i metodi a
    #    etichette da quelli ad atlante, e scrittura di CSV ed Excel.
    s1 = (d.groupby("metodo")[["dice_ctx", "dice_wc", "jaccard_ctx", "jaccard_wc"]]
            .agg(["mean", "std", "count"]).round(3))
    s1.columns = ["_".join(c) for c in s1.columns]
    s1 = s1.reset_index()
    s1["gruppo"] = s1.metodo.map(lambda m: "vs FreeSurfer (DKT)" if m in A_ETICHETTE
                                 else "vs FreeSurfer (non omologo)")
    s2 = (cop.groupby("metodo")[["dice_ctx_vs_atlante", "dice_wc_vs_atlante"]]
             .agg(["mean", "std", "count"]).round(3)) if not cop.empty else pd.DataFrame()
    if not s2.empty:
        s2.columns = ["_".join(c) for c in s2.columns]
        s2 = s2.reset_index()

    s1.to_csv(outdir / "dice_sintesi.csv", index=False)
    with pd.ExcelWriter(outdir / "dice_vs_freesurfer.xlsx", engine="openpyxl") as xw:
        s1.to_excel(xw, sheet_name="Sintesi vs FreeSurfer", index=False)
        if not s2.empty:
            s2.to_excel(xw, sheet_name="Copertura VOI atlante", index=False)
        d.to_excel(xw, sheet_name="Per soggetto", index=False)

    print("\n— Dice medio rispetto a FreeSurfer —")
    print(s1[["gruppo", "metodo", "dice_ctx_mean", "dice_ctx_std",
              "dice_wc_mean", "dice_wc_count"]].sort_values(
              ["gruppo", "dice_ctx_mean"], ascending=[True, False]).to_string(index=False))
    if not s2.empty:
        print("\n— Copertura della VOI atlante standard (metodi ad atlante) —")
        print(s2.to_string(index=False))
    print(f"\nScritti in {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
