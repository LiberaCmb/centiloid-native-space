"""Modulo 01 — Caricamento delle immagini.

Carica una serie DICOM oppure un singolo volume NIfTI in un `ImageVolume`, lo
riorienta in RAS e ne legge la geometria.
"""

from .logic import (
    ImageLoader,
    detect_tracer,
    update_context_with_file,
)

__all__ = ["ImageLoader", "detect_tracer", "update_context_with_file"]
