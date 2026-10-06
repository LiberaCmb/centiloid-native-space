"""Soglie di Otsu multilivello sull'intensita' della MRI T1 (tre classi).

Dentro la maschera del cervello, l'istogramma della T1 corretta con N4 viene
diviso da due soglie di Otsu (scikit-image, `threshold_multiotsu`) in tre
classi, che dalla piu' scura alla piu' chiara sono liquor, sostanza grigia e
sostanza bianca. Le soglie massimizzano la varianza fra le classi (capitolo 2).
"""
from __future__ import annotations
import logging
import numpy as np
import SimpleITK as sitk
from ..base import BaseSegmenter, register_segmenter, n4_bias_correct

logger = logging.getLogger(__name__)


@register_segmenter("otsu")
class OtsuSegmenter(BaseSegmenter):
    """Otsu a tre classi: 1 = liquor, 2 = sostanza grigia, 3 = sostanza bianca."""

    label_scheme = "tissue3"

    def _segment(self, mri, external_brain_mask=None,
                 n4_correct: bool = True, n4_shrink: int = 4, **_):
        """Mappa a tre tessuti della MRI dentro la maschera del cervello.

        mri                   MRI T1 (sitk.Image)
        external_brain_mask   maschera comune dei metodi classici (ICBM152
                              portato nello spazio nativo); se None, ripiego
                              di `_brain_mask`
        n4_correct            applica la correzione N4 prima delle soglie
        n4_shrink             fattore di riduzione per la stima N4

        Restituisce (mappa di etichette, extras con soglie e conteggi).
        """
        # 1. Maschera del cervello e correzione della disomogeneita' di campo.
        #    Se N4 fallisce si prosegue sulle intensita' originali.
        mask = self._brain_mask(mri, use_external_mask=external_brain_mask)
        if mask.sum() == 0:
            raise RuntimeError("Otsu: la maschera del cervello e' vuota.")

        if n4_correct:
            try:
                mri_corr = n4_bias_correct(mri, mask=mask, shrink=n4_shrink)
                logger.info("Otsu: correzione N4 applicata (shrink=%d).", n4_shrink)
            except Exception as e:
                logger.warning("Otsu: N4 non riuscita (%s), uso le intensita' "
                               "originali.", e)
                mri_corr = sitk.Cast(mri, sitk.sitkFloat32)
        else:
            mri_corr = sitk.Cast(mri, sitk.sitkFloat32)

        # 2. Due soglie calcolate solo sull'istogramma dei voxel del cervello:
        #    il fondo nero, molto piu' numeroso, sposterebbe le soglie.
        arr = sitk.GetArrayFromImage(mri_corr).astype(np.float32)
        in_brain = arr[mask > 0]
        if in_brain.size == 0:
            raise RuntimeError("Otsu: nessun voxel nella maschera del cervello.")

        try:
            from skimage.filters import threshold_multiotsu
            thresholds = threshold_multiotsu(in_brain, classes=3)
        except Exception as e:
            logger.warning("Otsu: threshold_multiotsu non riuscita (%s), uso i "
                           "terzili.", e)
            # ripiego: soglie al 33.3 e al 66.7 percentile (tre classi uguali)
            thresholds = np.percentile(in_brain, np.linspace(0, 100, 4)[1:-1])

        # 3. Etichettatura: le classi di intensita' diventano 1/2/3 solo dentro
        #    la maschera; fuori resta 0.
        # digitize: 0 sotto la prima soglia, 1 fra le due, 2 sopra la seconda
        raw = np.digitize(arr, thresholds, right=False).astype(np.int16)
        labels_arr = np.zeros_like(raw, dtype=np.int16)
        labels_arr[(mask > 0) & (raw == 0)] = 1
        labels_arr[(mask > 0) & (raw == 1)] = 2
        labels_arr[(mask > 0) & (raw >= 2)] = 3

        # 4. Controllo: una classe sotto il 2% del cervello indica soglie
        #    degeneri (per esempio N4 o maschera sbagliate). Solo un avviso.
        unique, counts = np.unique(labels_arr[mask > 0], return_counts=True)
        for u, c in zip(unique, counts):
            if u in (1, 2, 3) and c / max(in_brain.size, 1) < 0.02:
                logger.warning("Otsu: la classe %d copre solo il %.1f%% del "
                               "cervello: segmentazione forse degenere.",
                               int(u), 100 * c / max(in_brain.size, 1))

        out = self._np_to_sitk_like(labels_arr, mri)
        return out, {
            "thresholds": [float(t) for t in thresholds],
            "n4_correct": bool(n4_correct),
            "label_counts": {int(u): int(c) for u, c in zip(unique, counts)},
        }
