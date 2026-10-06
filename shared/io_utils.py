"""Lettura delle immagini da disco.

Il caricamento completo (DICOM compreso) sta nel modulo 01; qui c'e' solo la
lettura di un volume gia' convertito, usata dai moduli successivi.
"""

from __future__ import annotations
import SimpleITK as sitk


def load_image(path: str) -> sitk.Image:
    """Legge un'immagine e la riorienta in RAS, la convenzione di tutta la pipeline."""
    img = sitk.ReadImage(str(path))
    orient = sitk.DICOMOrientImageFilter()
    orient.SetDesiredCoordinateOrientation("RAS")
    return orient.Execute(img)
