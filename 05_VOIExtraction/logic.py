"""Regioni di interesse dalle etichette FreeSurfer (SynthSeg, FastSurfer, SAMSEG).

Dalla segmentazione si ricavano due maschere binarie sulla griglia della MRI:
  voi_ctx  target corticale (le 38 etichette DKT del Centiloid);
  voi_wc   riferimento, il cervelletto intero (etichette 7, 8, 46, 47).

Se la segmentazione non contiene la parcellazione corticale (SAMSEG), il
target diventa la corteccia intera dei due emisferi (etichette 3 e 42).

I metodi classici non hanno etichette anatomiche e passano invece per
`atlas_voi.py`.

Dilatazione del target
----------------------
Il nastro corticale e' spesso circa 3 mm, quanto la risoluzione della PET: il
segnale letto al suo interno e' diluito dalla sostanza bianca e dal liquor
adiacenti (effetto di volume parziale). `ctx_dilation_iters` allarga il target
di quel numero di voxel (dilatazione binaria di scipy, restando dentro il
cervello) per recuperare parte del segnale disperso, al prezzo di includere
tessuto che non appartiene alla regione. E' spenta per impostazione
predefinita; l'esperimento e' nel capitolo 4.
"""

from __future__ import annotations
import os
import logging
import numpy as np
import SimpleITK as sitk

from .label_mappings import (
    FREESURFER_CTX_LABELS,
    FREESURFER_CTX_LABELS_ASEG,
    FREESURFER_WC_LABELS,
)

logger = logging.getLogger(__name__)


def _labels_to_mask(seg_arr, labels):
    """Maschera binaria (uint8) dei voxel la cui etichetta e' in `labels`."""
    return np.isin(seg_arr, list(labels)).astype(np.uint8)


def dilata_target(ctx_arr: np.ndarray, brain_arr: np.ndarray,
                  iterazioni: int) -> np.ndarray:
    """Dilata il target di `iterazioni` voxel restando dentro `brain_arr`.

    E' la stessa operazione usata da `scripts/esecuzione/dilation_recompute.py`
    per l'esperimento sulle VOI gia' calcolate.
    """
    from scipy import ndimage
    grown = ndimage.binary_dilation(ctx_arr.astype(bool), iterations=int(iterazioni))
    return (grown & (brain_arr > 0)).astype(np.uint8)


def extract_vois(segmentation, label_scheme="freesurfer",
                 ctx_dilation_iters: int = 0,
                 ctx_labels=None, wc_labels=None):
    """Target corticale e cervelletto da una segmentazione con etichette FreeSurfer.

    Parametri
    ---------
    segmentation : sitk.Image
    label_scheme : str
        Deve essere 'freesurfer'. I metodi classici ('tissue3') non hanno
        etichette anatomiche: le loro VOI vengono da `atlas_voi.py`.
    ctx_dilation_iters : int
        Voxel di dilatazione del target (0 = nessuna).
    ctx_labels, wc_labels : insiemi di etichette, facoltativi
        Sostituiscono le definizioni standard. Li usa l'interfaccia per
        comporre regioni a scelta; con None si usa lo standard Centiloid.

    Restituisce (voi_ctx, voi_wc, info).
    """
    if label_scheme != "freesurfer":
        raise ValueError(
            f"Schema '{label_scheme}' senza etichette anatomiche: per i metodi "
            "classici le VOI si ricavano dall'atlante (atlas_voi.py).")

    # 1. Insiemi di etichette: quelli passati dall'interfaccia o lo standard.
    seg_arr = sitk.GetArrayFromImage(segmentation).astype(np.int32)

    custom_labels = (ctx_labels is not None) or (wc_labels is not None)
    ctx_label_set = set(ctx_labels) if ctx_labels is not None else FREESURFER_CTX_LABELS
    wc_label_set = set(wc_labels) if wc_labels is not None else FREESURFER_WC_LABELS

    info = {"label_scheme": label_scheme, "method": None,
            "ctx_voxels": 0, "wc_voxels": 0,
            "custom_labels": custom_labels,
            "ctx_labels": sorted(ctx_label_set),
            "wc_labels": sorted(wc_label_set),
            "ctx_dilation_iters": int(ctx_dilation_iters)}

    # 2. Maschere dalle etichette; se il target DKT e' vuoto e le etichette
    #    sono quelle standard, si ripiega sulla corteccia intera.
    ctx_arr = _labels_to_mask(seg_arr, ctx_label_set)
    wc_arr = _labels_to_mask(seg_arr, wc_label_set)
    info["method"] = ("freesurfer_custom_labels" if custom_labels
                      else "freesurfer_label_lookup_aparc")
    if int(ctx_arr.sum()) == 0 and not custom_labels:
        # nessuna parcellazione corticale (SAMSEG): corteccia intera
        ctx_arr = _labels_to_mask(seg_arr, FREESURFER_CTX_LABELS_ASEG)
        info["method"] = "freesurfer_label_lookup_aseg"
        logger.info("Nessuna etichetta aparc: uso la corteccia intera "
                    "(etichette 3 e 42) come target.")

    # 3. Dilatazione facoltativa del target. Il "cervello" che la limita e'
    #    l'insieme dei voxel con un'etichetta qualsiasi (seg_arr > 0).
    if ctx_dilation_iters and ctx_dilation_iters > 0 and ctx_arr.sum() > 0:
        ctx_before = int(ctx_arr.sum())
        ctx_arr = dilata_target(ctx_arr, seg_arr, ctx_dilation_iters)
        info["ctx_voxels_before_dilation"] = ctx_before
        info["method"] += f"+dil{ctx_dilation_iters}"
        logger.info("Target dilatato di %d voxel: %d -> %d voxel",
                    ctx_dilation_iters, ctx_before, int(ctx_arr.sum()))

    # 4. Conteggi e avvisi: una VOI vuota non e' un errore qui, il modulo 06
    #    la trasforma in SUVr NaN.
    info["ctx_voxels"] = int(ctx_arr.sum())
    info["wc_voxels"] = int(wc_arr.sum())
    if info["ctx_voxels"] == 0:
        logger.warning("Il target corticale e' vuoto: controllare le etichette.")
    if info["wc_voxels"] == 0:
        logger.warning("Il cervelletto e' vuoto: controllare le etichette.")

    # 5. Ritorno a sitk.Image con la geometria della segmentazione.
    voi_ctx = sitk.GetImageFromArray(ctx_arr.astype(np.uint8))
    voi_ctx.CopyInformation(segmentation)
    voi_wc = sitk.GetImageFromArray(wc_arr.astype(np.uint8))
    voi_wc.CopyInformation(segmentation)
    return voi_ctx, voi_wc, info


def save_vois(voi_ctx, voi_wc, output_dir, prefix=""):
    """Scrive le due VOI come <prefix>voi_ctx.nii.gz e <prefix>voi_wc.nii.gz.

    Restituisce i due percorsi (ctx, wc).
    """
    os.makedirs(output_dir, exist_ok=True)
    ctx_path = os.path.join(output_dir, f"{prefix}voi_ctx.nii.gz")
    wc_path = os.path.join(output_dir, f"{prefix}voi_wc.nii.gz")
    sitk.WriteImage(voi_ctx, ctx_path)
    sitk.WriteImage(voi_wc, wc_path)
    return ctx_path, wc_path
