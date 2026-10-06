"""Dove la pipeline cerca i dati, l'atlante e i dataset originali.

Tutti i percorsi sono relativi alla cartella del progetto, cosi' il codice gira
su qualunque macchina senza modifiche. Ciascuno si puo' spostare altrove con
una variabile d'ambiente:

    CENTILOID_DATI       cartella dei risultati e dei file intermedi
                         (predefinita: <progetto>/data)
    CENTILOID_ATLANTE    template ICBM152 e VOI standard del Centiloid Project
                         (predefinita: <progetto>/resources/atlante)
    CENTILOID_DATASETS   archivi originali scaricati da GAAIN
                         (predefinita: la cartella `datasets` accanto al progetto)
    CENTILOID_DLBS       dataset DLBS in formato BIDS
                         (predefinita: la cartella `DLBS` accanto al progetto)
"""

from __future__ import annotations
import os
from pathlib import Path

# Radice del progetto: la cartella che contiene shared/.
ROOT = Path(__file__).resolve().parents[1]


def _da_ambiente(nome: str, predefinito: Path) -> Path:
    """Percorso dalla variabile d'ambiente `nome`, se definita e non vuota."""
    valore = os.environ.get(nome, "").strip()
    return Path(valore).expanduser() if valore else predefinito


DATI = _da_ambiente("CENTILOID_DATI", ROOT / "data")
ATLANTE = _da_ambiente("CENTILOID_ATLANTE", ROOT / "resources" / "atlante")
DATASETS = _da_ambiente("CENTILOID_DATASETS", ROOT.parent / "datasets")
DLBS = _da_ambiente("CENTILOID_DLBS", ROOT.parent / "DLBS")

# ── Atlante: template ICBM152 lineare e VOI standard del Centiloid Project ──
# Le VOI GAAIN vivono nello spazio ICBM152 *lineare* (griglia 181x217x181):
# il template deve essere quello, non l'MNI152 di FSL (182x218x182).
TEMPLATE_ICBM = ATLANTE / "icbm_avg_152_t1_tal_lin.nii.gz"
MASCHERA_ICBM = ATLANTE / "icbm_avg_152_t1_tal_lin_mask.nii.gz"
VOI_CTX_STD = ATLANTE / "voi_ctx_1mm.nii.gz"
VOI_WC_STD = ATLANTE / "voi_WhlCbl_1mm.nii.gz"

# ── File di riferimento ─────────────────────────────────────────────────────
GROUND_TRUTH = DATI / "ground_truth"
CONFRONTO = DATI / "comparison_all_methods"
# Coefficienti di ricalibrazione CL = a*SUVr + b. Si usa la copia in data/ se
# c'e' (e' quella che scripts/analisi/build_final_comparison.py riscrive);
# altrimenti quella della tesi, inclusa nel progetto, cosi' l'interfaccia e
# run_pipeline danno il centiloide ricalibrato anche senza la cartella dati.
COEFFICIENTI_TESI = ROOT / "resources" / "coefficienti" / "recalibration_final.csv"
COEFFICIENTI = (CONFRONTO / "recalibration_final.csv"
                if (CONFRONTO / "recalibration_final.csv").exists()
                else COEFFICIENTI_TESI)
# Valori di riferimento del dataset DLBS.
QAMYLOID_DLBS = GROUND_TRUTH / "QAmyloid_results_DLBS.xlsx"


def dati(*parti: str) -> Path:
    """Percorso dentro la cartella dei dati: `dati("results_fbb_ants")`."""
    return DATI.joinpath(*parti)


def risolvi(percorso) -> Path:
    """Trasforma un percorso letto da un `result.csv` in un percorso esistente.

    I `result.csv` registrano i percorsi relativi alla radice del progetto
    (`data/aligned_ants/...`). Se la cartella dei dati e' stata spostata con
    CENTILOID_DATI, il prefisso `data/` viene sostituito con quella cartella.
    """
    p = Path(str(percorso))
    if p.is_absolute():
        return p
    parti = p.parts
    if parti and parti[0] == "data":
        return DATI.joinpath(*parti[1:])
    return ROOT / p
