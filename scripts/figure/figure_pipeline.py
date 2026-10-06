"""Schema dei moduli della pipeline, per il capitolo 3.

Rifa' in inglese il vecchio fig_pipeline.pdf, che era in italiano e aveva la
casella 06 fuori squadra. Produce data/figure_tesi/fig_pipeline_en.pdf.

Uso:  python -m scripts.figure.figure_pipeline
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared import percorsi                 # noqa: E402
OUT = percorsi.dati("figure_tesi")

INK, INK2 = "#0b0b0b", "#52514e"
CONDIVISO = "#dfe7ef"   # moduli di preparazione, comuni a ogni pipeline
NOSTRO    = "#e9dff0"   # dove sta il contributo di questa tesi
BORDO     = "#8f9aa6"

MODULI = [
    ("00", "DICOM $\\rightarrow$ NIfTI", None, CONDIVISO),
    ("01", "Image loading", None, CONDIVISO),
    ("02", "PET$\\rightarrow$MR registration", "SimpleITK / ANTs", CONDIVISO),
    ("03", "Alignment quality control", None, CONDIVISO),
    ("04", "Segmentation", "classical / deep learning / Bayesian", NOSTRO),
    ("05", "VOI extraction", "label-based / atlas-based", NOSTRO),
    ("06", "Centiloid", "SUVr $\\rightarrow$ CL, recalibration", NOSTRO),
    ("07", "Comparison and synthesis", None, NOSTRO),
]

plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def disegna():
    """Schema verticale dei moduli 00-07 con legenda (capitolo 3)."""
    n = len(MODULI)
    # alt = altezza di una casella, passo = distanza verticale fra due caselle;
    # le coordinate sono unita' dati arbitrarie, scelte a occhio.
    alt, passo = 0.62, 1.0
    fig, ax = plt.subplots(figsize=(4.2, 6.9))
    ax.set_xlim(2.3, 8.3); ax.set_ylim(-1.0, (n - 1) * passo + alt + 0.1)
    ax.axis("off")

    x0, larghezza = 2.6, 5.4
    for i, (num, titolo, sotto, colore) in enumerate(MODULI):
        # Il modulo 00 sta in alto: si scende al crescere dell'indice.
        y = (n - 1 - i) * passo
        ax.add_patch(FancyBboxPatch(
            (x0, y), larghezza, alt, boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=0.9, edgecolor=BORDO, facecolor=colore, zorder=2))
        cx = x0 + larghezza / 2
        # Con il sottotitolo le due righe di testo si spostano verso l'alto e
        # verso il basso della casella; senza, il titolo resta centrato.
        if sotto:
            ax.text(cx, y + alt * 0.63, f"{num}  {titolo}", ha="center",
                    va="center", fontsize=10, color=INK, zorder=3)
            ax.text(cx, y + alt * 0.25, sotto, ha="center", va="center",
                    fontsize=8, color=INK2, zorder=3)
        else:
            ax.text(cx, y + alt / 2, f"{num}  {titolo}", ha="center",
                    va="center", fontsize=10, color=INK, zorder=3)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (cx, y - 0.02), (cx, y - passo + alt + 0.02),
                arrowstyle="-|>", mutation_scale=11, linewidth=0.9,
                color=INK2, zorder=1))

    # Legenda sotto lo schema (le y negative lasciate libere da set_ylim).
    for j, (colore, testo) in enumerate((
            (CONDIVISO, "shared preparation"),
            (NOSTRO, "where this thesis contributes"))):
        yy = -0.55 - j * 0.28
        ax.add_patch(FancyBboxPatch(
            (x0, yy), 0.42, 0.18, boxstyle="round,pad=0.01,rounding_size=0.04",
            linewidth=0.8, edgecolor=BORDO, facecolor=colore))
        ax.text(x0 + 0.56, yy + 0.09, testo, fontsize=8.5, color=INK2,
                ha="left", va="center")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_pipeline_en.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    """Scrive fig_pipeline_en.pdf."""
    # Nessun argomento: il parser serve solo a dare un errore se se ne passano.
    argparse.ArgumentParser().parse_args()
    disegna()
    print("fig_pipeline_en.pdf")
    print(f"\nScritta in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
