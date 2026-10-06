"""Figure del capitolo dei risultati, generate dai file di confronto correnti.

Produce sei PDF vettoriali in data/figure_tesi. Le etichette, i titoli e le
legende sono in inglese, perche' la versione finale della tesi e' in inglese.
Sono pensati per la stampa in bianco e nero oltre che a colori: ogni barra
porta il proprio valore scritto accanto, quindi il colore distingue le
famiglie di metodi ma non e' mai l'unico modo di leggere il grafico.

I nomi hanno tutti il prefisso fig_res_ per non collidere con le figure
didattiche del capitolo 2, che si chiamano fig_dice.pdf e fig_bland_altman.pdf
e stanno nella stessa cartella.

  fig_res_mae_methods.pdf        errore assoluto medio per metodo, un pannello
                                 per tracciante, calibrazione per regressione
  fig_res_mae_calibrations.pdf   effetto della ricalibrazione: formula
                                 pubblicata, due punti, regressione
  fig_res_bland_altman.pdf       Bland-Altman di SynthSeg sui tre traccianti
  fig_res_dice.pdf               sovrapposizione delle VOI, due gruppi separati
  fig_res_runtime.pdf            tempo di segmentazione per soggetto, scala log
  fig_res_dlbs_transfer.pdf      validazione: coefficienti GAAIN contro fit sul
                                 DLBS

Uso:  python -m scripts.figure.figure_risultati
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared import percorsi                 # noqa: E402
from shared.percorsi import risolvi         # noqa: E402
D = percorsi.CONFRONTO
USCITA = percorsi.dati("figure_tesi")

# Palette categoriale validata (tre famiglie, tre slot): blu, arancio, verde-acqua.
COL = {"apprendimento": "#2a78d6", "bayesiano": "#eb6834", "classico": "#1baf7a"}
FAMIGLIA_EN = {"apprendimento": "Learning-based", "bayesiano": "Bayesian",
               "classico": "Classical"}
FAM = {"synthseg": "apprendimento", "fastsurfer": "apprendimento",
       "samseg": "bayesiano", "samseg_atlas": "bayesiano",
       "otsu": "classico", "kmeans": "classico", "fsl_fast": "classico"}
# NOME ha "_" protetto per LaTeX (testi destinati alle tabelle), ETICHETTA e'
# il nome in chiaro usato sugli assi delle figure.
NOME = {"synthseg": "SynthSeg", "fastsurfer": "FastSurfer", "samseg": "SAMSEG",
        "samseg_atlas": "samseg\\_atlas", "otsu": "Otsu", "kmeans": "k-means",
        "fsl_fast": "FSL FAST"}
ETICHETTA = {"synthseg": "SynthSeg", "fastsurfer": "FastSurfer", "samseg": "SAMSEG",
             "samseg_atlas": "samseg_atlas", "otsu": "Otsu", "kmeans": "k-means",
             "fsl_fast": "FSL FAST"}
TRACCIANTE = {"FBB": "florbetaben", "FMM": "flutemetamol", "FBP": "florbetapir"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlesize": 9.5,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "pdf.fonttype": 42,
})


def _pulisci(ax, griglia="x"):
    """Griglia leggera dietro i dati, sull'asse dei valori."""
    ax.grid(axis=griglia, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)


def _barre_orizzontali(ax, etichette, valori, famiglie, fmt="{:.2f}", xmax=None):
    """Barre orizzontali colorate per famiglia, con il valore scritto accanto.

    Il primo elemento sta in alto. Senza xmax si lascia il 22% di spazio
    oltre la barra piu' lunga per il numero.
    """
    y = np.arange(len(valori))[::-1]
    ax.barh(y, valori, height=0.62, zorder=2,
            color=[COL[f] for f in famiglie], edgecolor="white", linewidth=0.8)
    lim = xmax if xmax else max(valori) * 1.22
    for yi, v in zip(y, valori):
        ax.text(v + lim * 0.015, yi, fmt.format(v), va="center", ha="left",
                fontsize=8.2, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(etichette)
    ax.set_xlim(0, lim)
    _pulisci(ax)


def _legenda_famiglie(fig, ncol=3, y=-0.02):
    """Legenda comune delle tre famiglie di metodi, sotto la figura."""
    from matplotlib.patches import Patch
    voci = [Patch(facecolor=COL[k], label=FAMIGLIA_EN[k]) for k in
            ("apprendimento", "bayesiano", "classico")]
    fig.legend(handles=voci, loc="lower center", ncol=ncol, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, y), labelcolor=INK)


# ── 1. MAE per metodo e tracciante ────────────────────────────────────────────
def fig_mae_metodi(rie: pd.DataFrame):
    """fig_res_mae_methods: MAE per metodo dopo la regressione, per tracciante."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    for ax, ds in zip(axes, ["FBB", "FMM", "FBP"]):
        d = (rie[(rie.Dataset == ds) & (rie.Calibrazione == "regressione")]
             .sort_values("MAE (CL)"))
        _barre_orizzontali(ax, [ETICHETTA[m] for m in d.Metodo],
                           d["MAE (CL)"].values,
                           [FAM[m] for m in d.Metodo], xmax=13.5)
        # xmax fisso: i tre pannelli hanno la stessa scala e si confrontano.
        ax.set_title(TRACCIANTE[ds], color=INK)
        # Etichetta dell'asse solo sul pannello centrale, vale per tutti.
        ax.set_xlabel("mean absolute error (CL)" if ds == "FMM" else "")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _legenda_famiglie(fig, y=0.0)
    fig.savefig(USCITA / "fig_res_mae_methods.pdf")
    plt.close(fig)


# ── 2. Effetto della calibrazione ─────────────────────────────────────────────
def fig_mae_calibrazioni(rie: pd.DataFrame):
    """fig_res_mae_calibrations: MAE dei metodi con le tre calibrazioni."""
    cal = [("raw", "published formula"), ("two-point", "two-point anchoring"),
           ("regressione", "least-squares regression")]
    metodi = ["synthseg", "fastsurfer", "samseg", "samseg_atlas",
              "otsu", "kmeans", "fsl_fast"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), sharey=True)
    x = np.arange(len(metodi))
    for ax, ds in zip(axes, ["FBB", "FMM", "FBP"]):
        for i, (c, lab) in enumerate(cal):
            v = [rie[(rie.Dataset == ds) & (rie.Metodo == m) &
                     (rie.Calibrazione == c)]["MAE (CL)"].iloc[0] for m in metodi]
            ax.plot(x, v, marker="o", markersize=4.5, linewidth=1.6,
                    color=["#52514e", "#eb6834", "#2a78d6"][i],
                    label=lab if ds == "FBB" else None, zorder=3)
            # Etichette solo dal primo pannello, per non triplicare la legenda.
        ax.set_xticks(x)
        ax.set_xticklabels([ETICHETTA[m] for m in metodi], rotation=45,
                           ha="right", fontsize=7.6)
        ax.set_title(TRACCIANTE[ds], color=INK)
        _pulisci(ax, "y")
    axes[0].set_ylabel("mean absolute error (CL)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0), labelcolor=INK)
    fig.savefig(USCITA / "fig_res_mae_calibrations.pdf")
    plt.close(fig)


# ── 3. Bland-Altman di SynthSeg ───────────────────────────────────────────────
def fig_bland_altman(fogli: dict):
    """fig_res_bland_altman: SynthSeg ricalibrato contro il riferimento.

    fogli     un DataFrame per tracciante (fogli dell'Excel finale), con le
              colonne "ground_truth" e "synthseg\\nreg"
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    for ax, ds in zip(axes, ["FBB", "FMM", "FBP"]):
        x = fogli[ds]
        v = x[["ground_truth", "synthseg\nreg"]].dropna()
        rif, st = v.ground_truth.values, v["synthseg\nreg"].values
        # Differenza e media per soggetto; limiti di accordo a +-1.96 sd.
        d, m = st - rif, (st + rif) / 2
        md, sd = d.mean(), d.std(ddof=1)
        ax.axhline(0, color=GRID, linewidth=0.8, zorder=1)
        ax.scatter(m, d, s=16, color=COL["apprendimento"], alpha=0.85,
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.axhline(md, color=INK2, linewidth=1.2, zorder=2)
        for q in (md + 1.96 * sd, md - 1.96 * sd):
            ax.axhline(q, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
        # banda libera in alto per l'annotazione, cosi' non tocca ne' i punti
        # ne' la linea tratteggiata del limite superiore
        lo = min(d.min(), md - 1.96 * sd)
        hi = max(d.max(), md + 1.96 * sd)
        r = hi - lo
        ax.set_ylim(lo - 0.10 * r, hi + 0.42 * r)
        # Con la regressione il bias medio e' circa zero: si evita di stampare
        # "-0.00".
        mostra = 0.0 if abs(md) < 0.005 else md
        ax.text(0.03, 0.97, f"mean {mostra:+.2f} CL\nlimits ±{1.96*sd:.1f} CL",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.8,
                color=INK2)
        ax.set_title(TRACCIANTE[ds], color=INK)
        ax.set_xlabel("mean of estimate and reference (CL)" if ds == "FMM" else "")
        _pulisci(ax, "y")
    axes[0].set_ylabel("estimate − reference (CL)")
    fig.tight_layout()
    fig.savefig(USCITA / "fig_res_bland_altman.pdf")
    plt.close(fig)


# ── 4. Dice ───────────────────────────────────────────────────────────────────
def fig_dice(sint: pd.DataFrame, cop: pd.DataFrame):
    """fig_res_dice: Dice delle VOI in due pannelli.

    sint      Dice medi contro FreeSurfer (metodi che producono etichette)
    cop       Dice medi contro la VOI dell'atlante standard (gli altri metodi)

    I due gruppi hanno riferimenti diversi, quindi stanno in pannelli separati.
    """
    g1 = ["fastsurfer", "synthseg", "samseg"]
    g2 = ["samseg_atlas", "fsl_fast", "otsu", "kmeans"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    d1 = sint.set_index("metodo").loc[g1]
    y = np.arange(len(g1))[::-1]
    # Due barre per metodo, sfalsate di +-0.19: target (colore pieno della
    # famiglia) sopra, cervelletto (tinta chiara) sotto.
    for off, col, lab in ((0.19, COL["apprendimento"], "target cortex"),
                          (-0.19, "#9bbfe8", "cerebellum")):
        v = d1["dice_ctx_mean"] if off > 0 else d1["dice_wc_mean"]
        axes[0].barh(y + off, v.values, height=0.34, color=col, zorder=2,
                     edgecolor="white", linewidth=0.8, label=lab)
        for yi, x in zip(y + off, v.values):
            axes[0].text(x + 0.012, yi, f"{x:.3f}", va="center", fontsize=7.8, color=INK)
    axes[0].set_yticks(y); axes[0].set_yticklabels([ETICHETTA[m] for m in g1])
    axes[0].set_xlim(0, 1.12); axes[0].set_title("against FreeSurfer", color=INK)
    _pulisci(axes[0])

    d2 = cop.set_index("metodo").loc[g2]
    y = np.arange(len(g2))[::-1]
    for off, col, lab in ((0.19, COL["classico"], "target cortex"),
                          (-0.19, "#a3ddc5", "cerebellum")):
        v = (d2["dice_ctx_vs_atlante_mean"] if off > 0
             else d2["dice_wc_vs_atlante_mean"])
        axes[1].barh(y + off, v.values, height=0.34, color=col, zorder=2,
                     edgecolor="white", linewidth=0.8, label=lab)
        for yi, x in zip(y + off, v.values):
            axes[1].text(x + 0.012, yi, f"{x:.3f}", va="center", fontsize=7.8, color=INK)
    axes[1].set_yticks(y); axes[1].set_yticklabels([ETICHETTA[m] for m in g2])
    axes[1].set_xlim(0, 1.12)
    axes[1].set_title("coverage of the standard atlas VOI", color=INK)
    _pulisci(axes[1])
    for ax in axes:
        ax.set_xlabel("Dice coefficient")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    # Legenda in grigio: indica solo scuro = target e chiaro = cervelletto,
    # valido per entrambe le famiglie di colore.
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="#6f6f6f", label="target cortex"),
                        Patch(facecolor="#bdbdbd", label="cerebellum")],
               loc="lower center", ncol=2, frameon=False, fontsize=8.2,
               bbox_to_anchor=(0.5, 0.0), labelcolor=INK)
    fig.savefig(USCITA / "fig_res_dice.pdf")
    plt.close(fig)


# ── 5. Tempi ──────────────────────────────────────────────────────────────────
def fig_tempi(t: pd.DataFrame):
    """fig_res_runtime: tempo per soggetto (mediana, da minimo a massimo)."""
    t = t[t.metodo.isin(FAM)].sort_values("mediana")
    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    y = np.arange(len(t))[::-1]
    for yi, lo, hi in zip(y, t.minimo.values, t.massimo.values):
        ax.plot([lo, hi], [yi, yi], color=GRID, linewidth=2.2, zorder=1,
                solid_capstyle="round")
    ax.scatter(t.mediana.values, y, s=46, zorder=3,
               color=[COL[FAM[m]] for m in t.metodo],
               edgecolor="white", linewidth=0.8)
    # Mediane scritte in colonna sul bordo destro (scala log fino a 3e4 s),
    # cosi' non si sovrappongono agli intervalli.
    for yi, v in zip(y, t.mediana.values):
        ax.text(2.6e4, yi, f"{v:.1f} s", va="center", ha="right",
                fontsize=8.2, color=INK)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels([ETICHETTA[m] for m in t.metodo])
    ax.set_xlabel("seconds per subject, log scale")
    ax.set_xlim(0.2, 3.0e4)
    _pulisci(ax)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _legenda_famiglie(fig, y=0.0)
    fig.savefig(USCITA / "fig_res_runtime.pdf")
    plt.close(fig)


# ── 6. Trasferimento al DLBS ──────────────────────────────────────────────────
def fig_dlbs(v: pd.DataFrame):
    """fig_res_dlbs_transfer: MAE sul DLBS, coefficienti GAAIN contro rifit."""
    # Si usa il campione senza i sette soggetti esclusi al controllo qualita'.
    d = v[v.campione == "senza 7 QC"].sort_values("mae")
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    y = np.arange(len(d))[::-1]
    # Il segmento grigio unisce, per ogni metodo, il MAE con la retta stimata
    # sul DLBS stesso a quello con i coefficienti GAAIN trasferiti.
    for yi, a, b in zip(y, d.mae_fit_su_DLBS.values, d.mae.values):
        ax.plot([a, b], [yi, yi], color=GRID, linewidth=1.6, zorder=1)
    ax.scatter(d.mae_fit_su_DLBS.values, y, s=26, color=INK2, zorder=3,
               label="line refitted on DLBS")
    ax.scatter(d.mae.values, y, s=34, zorder=3,
               color=[COL[FAM[m]] for m in d.metodo], edgecolor="white",
               linewidth=0.6, label="GAAIN coefficients applied (colour by family)")
    for yi, b in zip(y, d.mae.values):
        ax.text(b + 0.18, yi, f"{b:.2f}", va="center", fontsize=8.2, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([ETICHETTA[m] for m in d.metodo])
    ax.set_xlabel("mean absolute error on DLBS (CL), 219 subjects")
    ax.set_xlim(3, 11.2)
    _pulisci(ax)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=8.2,
               bbox_to_anchor=(0.5, 0.0), labelcolor=INK)
    fig.savefig(USCITA / "fig_res_dlbs_transfer.pdf")
    plt.close(fig)


def main() -> int:
    """Legge i file di confronto e scrive i sei PDF."""
    # 1. Lettura dei file di confronto (moduli 07 e scripts/analisi).
    argparse.ArgumentParser().parse_args()
    finale = D / "confronto_metodi_centiloide_ricalibrato_ANTs_finale.xlsx"
    rie = pd.read_excel(finale, sheet_name="Riepilogo").dropna(subset=["Metodo"])
    fogli = {ds: pd.read_excel(finale, sheet_name=ds) for ds in ("FBB", "FMM", "FBP")}
    sint = pd.read_csv(D / "dice_sintesi.csv")
    # dice_copertura e' per soggetto: qui si riduce alla media per metodo.
    cop = (pd.read_csv(D / "dice_copertura.csv")
             .groupby("metodo")[["dice_ctx_vs_atlante", "dice_wc_vs_atlante"]]
             .mean().reset_index()
             .rename(columns={"dice_ctx_vs_atlante": "dice_ctx_vs_atlante_mean",
                              "dice_wc_vs_atlante": "dice_wc_vs_atlante_mean"}))
    tempi = pd.read_csv(D / "tempi_sintesi.csv")
    val = pd.read_csv(D / "validazione_dlbs.csv")

    # 2. Le sei figure.
    USCITA.mkdir(parents=True, exist_ok=True)
    fig_mae_metodi(rie)
    print("fig_res_mae_methods.pdf")
    fig_mae_calibrazioni(rie)
    print("fig_res_mae_calibrations.pdf")
    fig_bland_altman(fogli)
    print("fig_res_bland_altman.pdf")
    fig_dice(sint, cop)
    print("fig_res_dice.pdf")
    fig_tempi(tempi)
    print("fig_res_runtime.pdf")
    fig_dlbs(val)
    print("fig_res_dlbs_transfer.pdf")
    print(f"\nScritte in {USCITA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
