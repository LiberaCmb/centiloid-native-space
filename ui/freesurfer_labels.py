"""Catalogo COMPLETO delle regioni FreeSurfer selezionabili nell'interfaccia.

Il professore vuole poter controllare la segmentazione e comporre liberamente
sia il target (corteccia bersaglio) sia il reference (regione di
normalizzazione). Perciò qui sono elencate TUTTE le etichette che SynthSeg e
FastSurfer producono:

  * la parcellazione corticale aparc/DKT (per lobo),
  * le strutture sottocorticali aseg (talamo, nuclei della base, ippocampo…),
  * cervelletto, sostanza bianca, ventricoli/CSF, tronco encefalico e altro.

Convenzione FreeSurfer (aparc): una regione corticale con indice base `k` è
`1000 + k` a SINISTRA e `2000 + k` a DESTRA. Il Centiloide è bilaterale, quindi
in interfaccia ogni regione è UNA casella che porta con sé entrambi gli
emisferi. Le strutture aseg pari (es. talamo) sono anch'esse bilaterali; quelle
mediane (tronco, 3°/4° ventricolo, corpo calloso, chiasma) sono singole.

`in_std_ctx` / `in_std_wc` marcano le regioni del target / reference del
Centiloide standard (Klunk et al. 2015) e devono restare IDENTICHE a
05_VOIExtraction/label_mappings.py (verificato dal blocco __main__).
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple


class Region:
    """Una regione selezionabile (porta con sé una o più label FreeSurfer)."""

    def __init__(self, name_it: str, ids: Tuple[int, ...], category: str,
                 in_std_ctx: bool = False, in_std_wc: bool = False):
        self.name_it = name_it
        self.ids: Tuple[int, ...] = tuple(ids)
        self.category = category
        self.in_std_ctx = in_std_ctx
        self.in_std_wc = in_std_wc

    @property
    def key(self) -> str:
        """Identificativo stabile, usato come chiave delle caselle Streamlit."""
        return "r_" + "_".join(str(i) for i in self.ids)

    @property
    def label_for_ui(self) -> str:
        """Testo della casella: nome italiano seguito dalle etichette."""
        idstr = "/".join(str(i) for i in self.ids)
        return f"{self.name_it}  (id {idstr})"


# ════════════════════════════════════════════════════════════════════════════
# 1) Parcellazione corticale aparc / DKT  (bilaterale: 1000+k e 2000+k)
# ════════════════════════════════════════════════════════════════════════════
# (indice_base, nome_it, lobo, in_target_standard)
# Nessuna regione corticale fa parte del riferimento standard, che e' solo
# cervelletto: per questo manca la colonna in_std_wc.
_CORTICAL: List[tuple] = [
    # ── Frontale ─────────────────────────────────────────────────────────
    (3,  "Frontale medio caudale",     "Frontale",   True),
    (27, "Frontale medio rostrale",    "Frontale",   True),
    (28, "Frontale superiore",         "Frontale",   True),
    (18, "Pars opercularis",           "Frontale",   True),
    (19, "Pars orbitalis",             "Frontale",   True),
    (20, "Pars triangularis",          "Frontale",   True),
    (12, "Orbitofrontale laterale",    "Frontale",   True),
    (14, "Orbitofrontale mediale",     "Frontale",   True),
    (24, "Precentrale",                "Frontale",   False),
    (17, "Paracentrale",               "Frontale",   False),
    (32, "Polo frontale",              "Frontale",   False),
    # ── Temporale ────────────────────────────────────────────────────────
    (9,  "Temporale inferiore",        "Temporale",  True),
    (15, "Temporale medio",            "Temporale",  True),
    (30, "Temporale superiore",        "Temporale",  True),
    (7,  "Fusiforme",                  "Temporale",  False),
    (6,  "Entorinale",                 "Temporale",  False),
    (16, "Paraippocampale",            "Temporale",  False),
    (33, "Polo temporale",             "Temporale",  False),
    (34, "Temporale trasverso",        "Temporale",  False),
    (1,  "Banks STS",                  "Temporale",  False),
    # ── Parietale ────────────────────────────────────────────────────────
    (8,  "Parietale inferiore",        "Parietale",  True),
    (29, "Parietale superiore",        "Parietale",  True),
    (31, "Sopramarginale",             "Parietale",  True),
    (25, "Precuneo",                   "Parietale",  True),
    (22, "Postcentrale",               "Parietale",  False),
    # ── Cingolo ──────────────────────────────────────────────────────────
    (2,  "Cingolo anteriore caudale",  "Cingolo",    True),
    (26, "Cingolo anteriore rostrale", "Cingolo",    True),
    (23, "Cingolo posteriore",         "Cingolo",    True),
    (10, "Cingolo istmo",              "Cingolo",    True),
    # ── Occipitale ───────────────────────────────────────────────────────
    (11, "Occipitale laterale",        "Occipitale", False),
    (5,  "Cuneo",                      "Occipitale", False),
    (13, "Linguale",                   "Occipitale", False),
    (21, "Pericalcarina",              "Occipitale", False),
    # ── Insula ───────────────────────────────────────────────────────────
    (35, "Insula",                     "Insula",     False),
]

# ════════════════════════════════════════════════════════════════════════════
# 2) Strutture aseg (sottocorticale, cervelletto, ventricoli, tronco, ...)
# ════════════════════════════════════════════════════════════════════════════
# (nome_it, ids, categoria, in_std_ctx, in_std_wc)
_ASEG: List[tuple] = [
    # ── Corteccia intera (aseg, senza parcellazione: es. SAMSEG) ──────────
    ("Corteccia cerebrale (intera)",    (3, 42),   "Corteccia intera (aseg)",   False, False),
    # ── Sottocorticale ───────────────────────────────────────────────────
    ("Talamo",                          (10, 49),  "Sottocorticale",            False, False),
    ("Caudato",                         (11, 50),  "Sottocorticale",            False, False),
    ("Putamen",                         (12, 51),  "Sottocorticale",            False, False),
    ("Pallido",                         (13, 52),  "Sottocorticale",            False, False),
    ("Ippocampo",                       (17, 53),  "Sottocorticale",            False, False),
    ("Amigdala",                        (18, 54),  "Sottocorticale",            False, False),
    ("Nucleo accumbens",                (26, 58),  "Sottocorticale",            False, False),
    ("Diencefalo ventrale (VentralDC)", (28, 60),  "Sottocorticale",            False, False),
    # ── Cervelletto (reference standard = corteccia + SB, sx+dx) ──────────
    ("Cervelletto - corteccia",         (8, 47),   "Cervelletto",               False, True),
    ("Cervelletto - sostanza bianca",   (7, 46),   "Cervelletto",               False, True),
    # ── Sostanza bianca cerebrale ─────────────────────────────────────────
    ("Sostanza bianca cerebrale",       (2, 41),   "Sostanza bianca",           False, False),
    ("Iperintensità sostanza bianca",   (77,),     "Sostanza bianca",           False, False),
    # ── Ventricoli e CSF ──────────────────────────────────────────────────
    ("Ventricolo laterale",             (4, 43),   "Ventricoli e CSF",          False, False),
    ("Corno temporale (Inf-Lat-Vent)",  (5, 44),   "Ventricoli e CSF",          False, False),
    ("Terzo ventricolo",                (14,),     "Ventricoli e CSF",          False, False),
    ("Quarto ventricolo",               (15,),     "Ventricoli e CSF",          False, False),
    ("CSF",                             (24,),     "Ventricoli e CSF",          False, False),
    ("Plesso corioideo",                (31, 63),  "Ventricoli e CSF",          False, False),
    # ── Tronco encefalico e altro ─────────────────────────────────────────
    ("Tronco encefalico (brain-stem)",  (16,),     "Tronco e altro",            False, False),
    ("Chiasma ottico",                  (85,),     "Tronco e altro",            False, False),
    ("Corpo calloso",         (251, 252, 253, 254, 255), "Tronco e altro",      False, False),
    ("Vaso (vessel)",                   (30, 62),  "Tronco e altro",            False, False),
]


# ── Costruzione del catalogo unificato ───────────────────────────────────────
def _build_regions() -> List[Region]:
    """Unisce le due tabelle in un'unica lista di Region, corticali per prime."""
    regs: List[Region] = []
    for base_idx, name_it, lobe, in_std in _CORTICAL:
        regs.append(Region(name_it, (1000 + base_idx, 2000 + base_idx),
                           f"Corticale — {lobe}", in_std_ctx=in_std))
    for name_it, ids, cat, in_ctx, in_wc in _ASEG:
        regs.append(Region(name_it, ids, cat, in_std_ctx=in_ctx, in_std_wc=in_wc))
    return regs


REGIONS: List[Region] = _build_regions()

# Ordine di visualizzazione delle categorie
CATEGORY_ORDER: List[str] = [
    "Corticale — Frontale",
    "Corticale — Temporale",
    "Corticale — Parietale",
    "Corticale — Cingolo",
    "Corticale — Occipitale",
    "Corticale — Insula",
    "Corteccia intera (aseg)",
    "Sottocorticale",
    "Cervelletto",
    "Sostanza bianca",
    "Ventricoli e CSF",
    "Tronco e altro",
]


def regions_by_category() -> Dict[str, List[Region]]:
    """Regioni raggruppate per categoria, nell'ordine di CATEGORY_ORDER.

    Una categoria assente da CATEGORY_ORDER viene comunque accodata in fondo.
    """
    out: Dict[str, List[Region]] = {c: [] for c in CATEGORY_ORDER}
    for r in REGIONS:
        out.setdefault(r.category, []).append(r)
    return out


# ════════════════════════════════════════════════════════════════════════════
# Preset Centiloide standard — deve coincidere con label_mappings.py
# ════════════════════════════════════════════════════════════════════════════
def _std_ctx_ids() -> Set[int]:
    """Etichette del target standard, ricavate dai flag in_std_ctx."""
    ids: Set[int] = set()
    for r in REGIONS:
        if r.in_std_ctx:
            ids.update(r.ids)
    return ids


def _std_wc_ids() -> Set[int]:
    """Etichette del riferimento standard, ricavate dai flag in_std_wc."""
    ids: Set[int] = set()
    for r in REGIONS:
        if r.in_std_wc:
            ids.update(r.ids)
    return ids


CENTILOID_STANDARD_CTX: Set[int] = _std_ctx_ids()
CENTILOID_STANDARD_WC: Set[int] = _std_wc_ids()


# ── Lookup id -> nome leggibile (per ri-mostrare i gruppi scelti) ────────────
def id_to_name() -> Dict[int, str]:
    """Dizionario etichetta -> nome; le corticali distinguono sx e dx."""
    out: Dict[int, str] = {}
    for r in REGIONS:
        if len(r.ids) == 2 and (r.ids[0] >= 1000):        # coppia corticale L/R
            out[r.ids[0]] = f"{r.name_it} (sx)"
            out[r.ids[1]] = f"{r.name_it} (dx)"
        else:
            for i in r.ids:
                out[i] = r.name_it
    return out


ID_TO_NAME: Dict[int, str] = id_to_name()


# Controllo di coerenza: `python ui/freesurfer_labels.py` verifica che i
# preset dell'interfaccia coincidano con le etichette usate dal modulo 05.
if __name__ == "__main__":
    import importlib
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    lm = importlib.import_module("05_VOIExtraction.label_mappings")
    assert CENTILOID_STANDARD_CTX == lm.FREESURFER_CTX_LABELS, (
        "CTX preset mismatch:\n"
        f"  ui-only : {sorted(CENTILOID_STANDARD_CTX - lm.FREESURFER_CTX_LABELS)}\n"
        f"  lm-only : {sorted(lm.FREESURFER_CTX_LABELS - CENTILOID_STANDARD_CTX)}")
    assert CENTILOID_STANDARD_WC == lm.FREESURFER_WC_LABELS, "WC preset mismatch"
    print("OK: preset Centiloide coincide con label_mappings.py")
    print(f"  Regioni totali selezionabili: {len(REGIONS)}")
    print(f"  Target standard: {len(CENTILOID_STANDARD_CTX)} label")
    print(f"  Reference standard: {len(CENTILOID_STANDARD_WC)} label")
