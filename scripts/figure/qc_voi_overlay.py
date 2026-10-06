"""Controllo visivo delle regioni di interesse sopra la MRI, in PNG.

Per ogni (soggetto, metodo) tre sezioni (coronale, sagittale, assiale) a
coordinate mondo fisse (predefinite x=15, y=0, z=0), con il target corticale
in rosso e il cervelletto in azzurro, nello stile dei controlli di
FreeSurfer. I percorsi di MRI e VOI si leggono dal `result.csv` di ciascun
metodo. Il disegno e' in `03_VisualCheck/figure_qc.py`.

    python -m scripts.figure.qc_voi_overlay --results-dir data/results_fbb_ants \\
        --out data/qc_overlays/fbb
"""
from __future__ import annotations
import argparse, csv, glob, importlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared.percorsi import risolvi           # noqa: E402
_visual = importlib.import_module("03_VisualCheck")


def parse_args():
    """Argomenti da riga di comando (vedi la docstring del modulo)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--coords", nargs=3, type=float, default=[15.0, 0.0, 0.0],
                   metavar=("X", "Y", "Z"), help="Coordinate mondo (mm, RAS).")
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main():
    """Una figura per ogni result.csv trovato in <results-dir>/<soggetto>/<metodo>/."""
    a = parse_args()
    out = risolvi(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fatti = errori = 0
    for fp in sorted(glob.glob(str(risolvi(a.results_dir) / "*" / "*" / "result.csv"))):
        sid, metodo = Path(fp).parts[-3], Path(fp).parts[-2]
        if a.subjects and sid not in a.subjects:
            continue
        if a.methods and metodo not in a.methods:
            continue
        png = out / f"{sid}_{metodo}.png"
        # Le figure gia' presenti non si rifanno: si puo' rilanciare dopo
        # un'interruzione senza ricominciare da capo.
        if png.exists():
            continue
        try:
            r = next(csv.DictReader(open(fp)))
            if not (r.get("anat_path") and r.get("voi_ctx_path") and r.get("voi_wc_path")):
                print(sid, metodo, "SALTO (percorsi assenti in result.csv)")
                continue
            _visual.figura_voi(risolvi(r["anat_path"]), risolvi(r["voi_ctx_path"]),
                               risolvi(r["voi_wc_path"]), png,
                               titolo=f"QC: {sid}  -  {metodo}", coords=tuple(a.coords))
            fatti += 1
            print(sid, metodo, "OK", flush=True)
        except Exception as e:
            errori += 1
            print(sid, metodo, "ERRORE", e, flush=True)
        # --limit conta solo le figure fatte, utile per una prova veloce.
        if a.limit and fatti >= a.limit:
            break
    print(f"Fatto: {fatti} figure, {errori} errori")


if __name__ == "__main__":
    main()
