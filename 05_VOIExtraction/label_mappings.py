"""Etichette FreeSurfer che compongono le due regioni di interesse.

Il target corticale segue la definizione standard del Centiloid Project
(Klunk et al. 2015) letta sulle etichette dell'atlante DKT: corteccia
frontale, temporale e parietale, cingolo anteriore e posteriore, precuneo,
nei due emisferi. Il riferimento e' il cervelletto intero.

Cambiare questi insiemi cambia il SUVr, e i coefficienti di ricalibrazione
sono stimati su questi: ogni modifica va documentata. L'interfaccia
(`ui/pipeline_service.py`) legge le liste da qui e non le duplica.
"""

from __future__ import annotations
from typing import Set

# ════════════════════════════════════════════════════════════════════════════
# Target corticale: parcellazione aparc/DKT (1000 + k a sinistra, 2000 + k a
# destra). 19 regioni per emisfero, 38 etichette in tutto.
# ════════════════════════════════════════════════════════════════════════════
FREESURFER_CTX_LABELS: Set[int] = {
    # ── Frontale ─────────────────────────────────────────────────────────
    1003, 2003,  # frontale medio caudale
    1027, 2027,  # frontale medio rostrale
    1028, 2028,  # frontale superiore
    1018, 2018,  # pars opercularis
    1019, 2019,  # pars orbitalis
    1020, 2020,  # pars triangularis
    1012, 2012,  # orbitofrontale laterale
    1014, 2014,  # orbitofrontale mediale
    # ── Temporale ────────────────────────────────────────────────────────
    1009, 2009,  # temporale inferiore
    1015, 2015,  # temporale medio
    1030, 2030,  # temporale superiore
    # ── Parietale ────────────────────────────────────────────────────────
    1008, 2008,  # parietale inferiore
    1029, 2029,  # parietale superiore
    1031, 2031,  # sopramarginale
    # ── Cingolo ──────────────────────────────────────────────────────────
    1002, 2002,  # cingolo anteriore caudale
    1026, 2026,  # cingolo anteriore rostrale
    1023, 2023,  # cingolo posteriore
    1010, 2010,  # istmo del cingolo
    # ── Precuneo ─────────────────────────────────────────────────────────
    1025, 2025,
}

# Riferimento: cervelletto intero = corteccia + sostanza bianca cerebellare.
FREESURFER_WC_LABELS: Set[int] = {
    7,    # Left-Cerebellum-White-Matter
    8,    # Left-Cerebellum-Cortex
    46,   # Right-Cerebellum-White-Matter
    47,   # Right-Cerebellum-Cortex
}

# ════════════════════════════════════════════════════════════════════════════
# Corteccia senza parcellazione (SAMSEG): un'etichetta per emisfero.
# ════════════════════════════════════════════════════════════════════════════
# SAMSEG restituisce l'atlante `aseg`, dove tutta la corteccia di un emisfero
# e' un'unica etichetta. E' anatomicamente corretta ma molto piu' ampia del
# target Centiloid: e' il target del metodo `samseg`, mentre `samseg_atlas` la
# interseca con la VOI standard.
FREESURFER_CTX_LABELS_ASEG: Set[int] = {3, 42}   # Left/Right-Cerebral-Cortex

# Etichette usate da `samseg_atlas` nell'intersezione con l'atlante (passate
# ai campi `cortex_labels` e `reference_labels` di `AtlasConfig`): corteccia
# dei due emisferi e cervelletto intero, gli stessi valori degli insiemi sopra.
SAMSEG_CORTEX_LABELS = (3, 42)
SAMSEG_CEREB_LABELS = (7, 8, 46, 47)

# ════════════════════════════════════════════════════════════════════════════
# Schema 'tissue3' dei metodi classici: 1 = liquor, 2 = grigia, 3 = bianca
# ════════════════════════════════════════════════════════════════════════════
# Valori predefiniti di `AtlasConfig.gm_label` e `wm_label` (atlas_voi.py).
TISSUE3_GM_LABEL = 2
TISSUE3_WM_LABEL = 3
