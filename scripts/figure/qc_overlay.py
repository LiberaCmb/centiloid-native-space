"""Controllo visivo dell'allineamento: PET allineata sopra la MRI, in PNG.

Per ogni soggetto un mosaico con tre righe (assiale, coronale, sagittale) e
cinque fette per riga: MRI in grigio, PET in scala "hot" con un contorno
giallo. Se il contorno segue i bordi del cervello l'allineamento e' buono; se
ne esce, la PET e' spostata o ruotata. Evita di aprire ogni soggetto in
ITK-SNAP. Il disegno e' in `03_VisualCheck/figure_qc.py`.

    python -m scripts.figure.qc_overlay --all
    python -m scripts.figure.qc_overlay --subjects 1008 2002 --root data/aligned_ants
    # cache SimpleITK, con un'etichetta per non sovrascrivere le figure ANTs:
    python -m scripts.figure.qc_overlay --all --tag sitk \\
        --root data/results_fbb/_realigned_cache data/results_fmm/_realigned_cache
"""
from __future__ import annotations
import argparse
import importlib
import sys
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
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--subjects", nargs="+", help="Soggetti da disegnare.")
    g.add_argument("--all", action="store_true",
                   help="Tutti i soggetti trovati nelle cartelle --root.")
    p.add_argument("--root", nargs="+", default=["data/aligned_ants"],
                   help="Cartelle con una sottocartella per soggetto; un "
                        "soggetto si prende dalla prima che lo contiene.")
    p.add_argument("--out", default="data/qc_overlays")
    p.add_argument("--tag", default="", help="Etichetta aggiunta al nome del file.")
    p.add_argument("--pet-threshold-pct", type=float, default=55.0,
                   help="Percentile della PET sotto cui la sovrapposizione e' "
                        "trasparente (predefinito 55).")
    return p.parse_args()


def main():
    """Disegna un mosaico di controllo per ogni soggetto richiesto."""
    args = parse_args()
    roots = [risolvi(r) for r in args.root]
    out_dir = risolvi(args.out)

    # 1. Elenco dei soggetti: con --all si uniscono le sottocartelle di tutte le
    #    radici, senza doppioni; quelle che iniziano con "_" sono cache o
    #    cartelle di servizio, non soggetti.
    if args.all:
        visti, soggetti = set(), []
        for root in roots:
            if root.exists():
                for p in sorted(root.iterdir()):
                    if p.is_dir() and not p.name.startswith("_") and p.name not in visti:
                        visti.add(p.name)
                        soggetti.append(p.name)
    else:
        soggetti = args.subjects
    print(f"{len(soggetti)} soggetti -> {out_dir}")

    # 2. Per ogni soggetto vale la prima radice che ha sia MRI sia PET allineata;
    #    un errore su un soggetto non ferma gli altri.
    n_ok = 0
    for sid in soggetti:
        coppia = None
        for root in roots:
            mr = root / sid / "Realigned_Anatomical" / "Realigned_Anatomical.nii.gz"
            pet = root / sid / "Realigned_Functional" / "Realigned_Functional.nii.gz"
            if mr.exists() and pet.exists():
                coppia = (mr, pet)
                break
        if coppia is None:
            print(f"  [salto] {sid}: manca la MRI o la PET")
            continue
        # L'etichetta --tag distingue nel nome e nel titolo le figure di
        # cartelle diverse (ad esempio ANTs e SimpleITK) dello stesso soggetto.
        suffisso = f"_{args.tag}" if args.tag else ""
        titolo = sid + (f" [{args.tag}]" if args.tag else "")
        try:
            png = _visual.figura_allineamento(
                coppia[0], coppia[1], out_dir / f"overlay_{sid}{suffisso}.png",
                titolo=titolo, pet_threshold_pct=args.pet_threshold_pct)
            print(f"  [ok] {sid} -> {png}")
            n_ok += 1
        except Exception as e:
            print(f"  [errore] {sid}: {e}")
    print(f"Fatto: {n_ok}/{len(soggetti)} figure in {out_dir}")


if __name__ == "__main__":
    main()
