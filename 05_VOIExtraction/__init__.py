"""Modulo 05 — Regioni di interesse nello spazio nativo.

Dalla segmentazione si ottengono le due maschere che servono al SUVr: il
target corticale e il riferimento (cervelletto intero), sulla griglia della
MRI. Due strade, a seconda del metodo:

  etichette (logic.py)       SynthSeg, FastSurfer, SAMSEG: le regioni si
                             leggono dalle etichette FreeSurfer;
  atlante (atlas_voi.py)     Otsu, k-means, FSL FAST, samseg_atlas: le VOI
                             standard del Centiloid Project, portate nello
                             spazio del soggetto, intersecate con la
                             segmentazione.
"""

from .logic import extract_vois, save_vois, dilata_target
from .label_mappings import (
    FREESURFER_CTX_LABELS,
    FREESURFER_WC_LABELS,
    SAMSEG_CORTEX_LABELS,
    SAMSEG_CEREB_LABELS,
)
from .atlas_voi import (
    AtlasConfig,
    config_predefinita,
    warp_atlas_to_native,
    intersect_vois_atlas,
    check_template_matches_voi,
)

__all__ = [
    "extract_vois",
    "save_vois",
    "dilata_target",
    "FREESURFER_CTX_LABELS",
    "FREESURFER_WC_LABELS",
    "SAMSEG_CORTEX_LABELS",
    "SAMSEG_CEREB_LABELS",
    "AtlasConfig",
    "config_predefinita",
    "warp_atlas_to_native",
    "intersect_vois_atlas",
    "check_template_matches_voi",
]
