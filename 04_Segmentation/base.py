"""Classe base, registro dei metodi e funzioni comuni ai segmentatori.

Fa parte del modulo 04: definisce il contratto che ogni metodo di
segmentazione rispetta (`BaseSegmenter`), il contenitore del risultato
(`SegmentationResult`), il registro `SEGMENTERS` con il suo decoratore e due
funzioni usate da piu' metodi: la correzione N4 (metodi classici) e il
ricampionamento di una mappa di etichette sulla griglia della MRI (metodi
FreeSurfer, che lavorano su una griglia propria).
"""

from __future__ import annotations
import abc
import dataclasses
import logging
import time
from typing import Optional, Dict, Type, Any
import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Risultato di una segmentazione
# ════════════════════════════════════════════════════════════════════════════
@dataclasses.dataclass
class SegmentationResult:
    """Uscita di `BaseSegmenter.segment`, passata al modulo 05."""
    label_map: sitk.Image        # mappa di etichette sulla griglia della MRI
    label_scheme: str            # 'freesurfer' oppure 'tissue3'
    method_name: str
    runtime_s: float             # durata di `_segment`, in secondi
    # informazioni diagnostiche proprie del metodo (soglie, centroidi, file)
    extras: Dict[str, Any] = dataclasses.field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════════
# Classe base
# ════════════════════════════════════════════════════════════════════════════
class BaseSegmenter(abc.ABC):
    """Contratto comune: ogni metodo implementa `_segment(...)`.

    `segment(...)` lo avvolge misurandone il tempo (e' il `seg_runtime_s` dei
    risultati) e controllando che l'uscita sia una mappa di etichette intere.

    Schemi di etichette:
      'freesurfer'  etichette anatomiche della tabella FreeSurfer
                    (SynthSeg, FastSurfer, SAMSEG);
      'tissue3'     tre classi di tessuto, 1 = liquor, 2 = sostanza grigia,
                    3 = sostanza bianca (Otsu, k-means, FSL FAST).
    """

    # Attributi di classe ridefiniti dalle sottoclassi; `name` lo imposta il
    # decoratore di registrazione. `requires_gpu` e `requires_external_tool`
    # sono solo descrittivi (li mostra l'interfaccia).
    name: str = "base"
    label_scheme: str = "tissue3"
    requires_gpu: bool = False
    requires_external_tool: Optional[str] = None

    def segment(self, mri, **kwargs):
        """Esegue il metodo sulla MRI e ne impacchetta l'uscita.

        mri       MRI T1 del soggetto (sitk.Image), spazio nativo
        kwargs    opzioni passate tali e quali a `_segment` (per esempio
                  `external_brain_mask` per i metodi classici)

        Restituisce un `SegmentationResult`.
        """
        if mri is None:
            raise ValueError(f"{self.name}: serve la MRI.")

        logger.info("Segmentazione: %s", self.name)
        t0 = time.perf_counter()
        label_map, extras = self._segment(mri=mri, **kwargs)
        dt = time.perf_counter() - t0
        logger.info("%s completata in %.1f s", self.name, dt)

        if not isinstance(label_map, sitk.Image):
            raise TypeError(f"{self.name}: _segment deve restituire un sitk.Image.")
        # Le etichette devono essere intere: un'uscita di altro tipo viene
        # convertita, cosi' il confronto con i valori delle etichette nel
        # modulo 05 e' esatto.
        if label_map.GetPixelID() not in (sitk.sitkUInt8, sitk.sitkInt16,
                                          sitk.sitkUInt16, sitk.sitkInt32,
                                          sitk.sitkUInt32):
            label_map = sitk.Cast(label_map, sitk.sitkInt16)

        return SegmentationResult(
            label_map=label_map,
            label_scheme=self.label_scheme,
            method_name=self.name,
            runtime_s=dt,
            extras=extras or {},
        )

    @abc.abstractmethod
    def _segment(self, mri, **kwargs):
        """Restituisce (label_map: sitk.Image, extras: dict)."""

    @staticmethod
    def _np_to_sitk_like(arr, ref):
        """Array numpy -> sitk.Image con la geometria di `ref`."""
        out = sitk.GetImageFromArray(arr)
        out.SetSpacing(ref.GetSpacing())
        out.SetOrigin(ref.GetOrigin())
        out.SetDirection(ref.GetDirection())
        return out

    @staticmethod
    def _brain_mask(mri, use_external_mask=None):
        """Maschera del cervello per i metodi classici.

        Nella pipeline la maschera arriva dall'esterno: e' la maschera del
        template ICBM152 portata nello spazio del soggetto dalla stessa
        registrazione che trasporta le VOI standard (vedi `05_VOIExtraction/
        atlas_voi.py`), uguale per i tre metodi classici.

        Senza maschera esterna (uso isolato del segmentatore, test) si ripiega
        su una soglia di Otsu globale seguita dalla componente connessa piu'
        grande: grezza, include scalpo e collo, ma basta a far girare il metodo.
        """
        if use_external_mask is not None:
            arr_mask = sitk.GetArrayFromImage(use_external_mask)
            return (arr_mask > 0).astype(np.uint8)

        # Otsu a due classi: 0 sotto la soglia, 1 sopra, istogramma a 200 bin.
        try:
            mask_img = sitk.OtsuThreshold(sitk.Cast(mri, sitk.sitkFloat32), 0, 1, 200)
            mask = sitk.GetArrayFromImage(mask_img).astype(np.uint8)
        except Exception:
            # Ripiego: si scarta il 20% piu' scuro dei voxel positivi.
            arr = sitk.GetArrayFromImage(mri).astype(np.float32)
            thr = float(np.quantile(arr[arr > 0], 0.20)) if (arr > 0).any() else 0.0
            mask = (arr > thr).astype(np.uint8)

        # Si tiene solo la componente connessa piu' grande, che di norma e'
        # la testa: cade il rumore di fondo isolato.
        from scipy import ndimage
        labels, n = ndimage.label(mask)
        if n > 0:
            sizes = ndimage.sum(mask, labels, index=range(1, n + 1))
            mask = (labels == int(np.argmax(sizes)) + 1).astype(np.uint8)
        return mask


def n4_bias_correct(mri: sitk.Image, mask: np.ndarray | None = None,
                    shrink: int = 4) -> sitk.Image:
    """Correzione della disomogeneita' di campo con N4 (Tustison et al. 2010).

    La sensibilita' delle bobine non e' uniforme e la luminosita' della MRI
    deriva lentamente nel volume: la stessa sostanza grigia risulta piu' chiara
    in una regione e piu' scura in un'altra. N4 stima questa deriva come un
    campo moltiplicativo liscio e la divide via. La stima si fa su
    un'immagine ridotta di un fattore `shrink` (il campo e' liscio, la perdita
    e' trascurabile) e poi si applica a piena risoluzione.

    mri       MRI T1 (sitk.Image)
    mask      maschera del cervello (array numpy, ordine z,y,x); se None si
              usa una soglia di Otsu
    shrink    fattore di riduzione per la stima del campo

    Restituisce la MRI corretta, in float32, sulla griglia originale.
    """
    img = sitk.Cast(mri, sitk.sitkFloat32)
    if mask is not None:
        mask_img = sitk.GetImageFromArray(mask.astype(np.uint8))
        mask_img.CopyInformation(img)
    else:
        mask_img = sitk.OtsuThreshold(img, 0, 1, 200)
    if shrink > 1:
        small = sitk.Shrink(img, [shrink] * img.GetDimension())
        small_mask = sitk.Shrink(mask_img, [shrink] * img.GetDimension())
    else:
        small, small_mask = img, mask_img
    corr = sitk.N4BiasFieldCorrectionImageFilter()
    # Quattro livelli di risoluzione del campo (dal piu' grossolano al piu'
    # fine) con le iterazioni massime per ciascuno.
    corr.SetMaximumNumberOfIterations([50, 50, 30, 20])
    _ = corr.Execute(small, small_mask)
    # Il campo stimato sull'immagine ridotta si valuta sulla griglia piena e
    # si divide via (il filtro lo restituisce in forma logaritmica).
    log_bias = corr.GetLogBiasFieldAsImage(img)
    return sitk.Cast(img / sitk.Exp(log_bias), sitk.sitkFloat32)


def resample_labels_like(seg: sitk.Image, ref: sitk.Image) -> sitk.Image:
    """Porta una mappa di etichette sulla griglia di `ref` (vicino piu' prossimo)."""
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(ref)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetDefaultPixelValue(0)
    return rs.Execute(seg)


# ════════════════════════════════════════════════════════════════════════════
# Registro dei metodi
# ════════════════════════════════════════════════════════════════════════════
# identificativo -> classe; popolato all'import dei singoli moduli
SEGMENTERS: Dict[str, Type[BaseSegmenter]] = {}


def register_segmenter(name: str):
    """Decoratore: @register_segmenter('otsu') rende la classe disponibile."""
    def decorator(cls):
        cls.name = name
        SEGMENTERS[name] = cls
        return cls
    return decorator
