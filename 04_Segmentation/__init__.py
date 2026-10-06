"""Modulo 04 — Segmentazione della MRI nello spazio nativo del soggetto.

E' il punto in cui questa pipeline si separa da quella standard del
Centiloid Project: invece di deformare il soggetto sul cervello standard, si
segmenta la sua MRI cosi' com'e'. Ogni metodo riceve la stessa MRI e restituisce una mappa di etichette
sulla stessa griglia; e' l'unico modulo che cambia da un metodo all'altro.

Metodi (identificativo -> classe)
---------------------------------
  classici (schema 'tissue3')   otsu, kmeans, fsl_fast
  FreeSurfer ('freesurfer')     synthseg, fastsurfer, samseg

Il settimo metodo della tesi, `samseg_atlas`, non e' un segmentatore a se':
e' la segmentazione di SAMSEG letta attraverso le VOI standard dell'atlante
(vedi `05_VOIExtraction` e `shared/pipeline.py`).

Ogni segmentatore si registra in `SEGMENTERS` con un decoratore. I moduli si
importano uno alla volta in un try/except, cosi' una dipendenza facoltativa
mancante (per esempio scikit-learn per k-means) rende indisponibile solo quel
metodo e non gli altri.
"""

from .base import BaseSegmenter, SegmentationResult, register_segmenter, SEGMENTERS

import importlib as _importlib
import logging as _logging

_logger = _logging.getLogger(__name__)

# Moduli da importare (relativi a questo pacchetto): l'import basta, perche'
# il decoratore `register_segmenter` registra la classe in SEGMENTERS.
_SEGMENTER_MODULES = [
    "classical.otsu", "classical.kmeans", "classical.fsl_fast",
    "ai.synthseg", "ai.fastsurfer", "ai.samseg",
]

for _mod in _SEGMENTER_MODULES:
    try:
        _importlib.import_module("." + _mod, __name__)
    except Exception as _exc:                                    # noqa: BLE001
        _logger.warning("Segmentatore '%s' non caricato (%s): il metodo non "
                        "sara' disponibile finche' la dipendenza non viene "
                        "installata.", _mod, _exc)


def get_segmenter(name: str) -> BaseSegmenter:
    """Istanzia un segmentatore dal suo identificativo ('otsu', 'synthseg', ...)."""
    if name not in SEGMENTERS:
        raise KeyError(f"Segmentatore sconosciuto '{name}'. Disponibili: "
                       f"{sorted(SEGMENTERS.keys())}")
    return SEGMENTERS[name]()


def list_segmenters() -> list[str]:
    """Identificativi dei segmentatori caricati correttamente, in ordine alfabetico."""
    return sorted(SEGMENTERS.keys())


__all__ = [
    "BaseSegmenter",
    "SegmentationResult",
    "register_segmenter",
    "SEGMENTERS",
    "get_segmenter",
    "list_segmenters",
]
