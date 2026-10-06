"""Modulo 06 - SUVr e Centiloide nello spazio nativo del soggetto.

Riceve la PET gia' allineata alla MRI (modulo 03) e le due regioni prodotte
dal modulo 05, e calcola

    SUVr = media della PET nel target corticale / media nel cervelletto
    CL   = a * SUVr + b

(tesi, capitolo 2 "The Centiloid scale" e capitolo 3 "Computation of the
centiloid").

I coefficienti (a, b) di CL_FORMULAS sono quelli pubblicati per ciascun
tracciante, stimati su pipeline che misurano in spazio standard. Applicati al
SUVr nativo danno centiloidi sistematicamente troppo bassi (bias medio di
-8.05 CL su FBB, -9.27 su FMM, -14.54 su FBP): per questo i risultati della
tesi usano coefficienti ricalibrati per regressione, stimati in
`scripts/analisi/build_final_comparison.py`. Il `centiloid` scritto nei
`result.csv` e' quello con la formula pubblicata ("grezzo"); quello
ricalibrato si ottiene dal SUVr con i coefficienti di
`resources/coefficienti/recalibration_final.csv`.
"""

from __future__ import annotations
import logging
import math
import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)

# Formule pubblicate dei traccianti, CL = pendenza * SUVr + intercetta
# (florbetaben, flutemetamol, florbetapir; regione di riferimento: cervelletto
# intero).
CL_FORMULAS = {
    'FBB': (153.4,  -154.9),
    'FMM': (121.42, -121.16),
    'FBP': (175.4,  -182.3),
}

# Intervallo di SUVr fisiologicamente plausibile: un negativo sta intorno a 1,
# una malattia avanzata arriva a circa 3. Fuori da qui il numero e' quasi
# sempre il sintomo di un problema (cervelletto fuori dal campo di vista della
# PET, etichette sbagliate) e il soggetto viene segnalato.
SUVR_PLAUSIBILE = (0.5, 4.0)


def suvr_to_cl(suvr: float, tracer: str) -> float:
    """Centiloide dal SUVr con la formula pubblicata del tracciante."""
    if tracer not in CL_FORMULAS:
        raise ValueError(f"Tracciante sconosciuto '{tracer}'. "
                         f"Disponibili: {sorted(CL_FORMULAS)}")
    a, b = CL_FORMULAS[tracer]
    return a * suvr + b


def _ensure_same_grid(mask, ref, atol: float = 1e-5):
    """Riporta `mask` sulla griglia di `ref` se non coincidono gia'.

    Dimensione, passo, origine e direzione si confrontano con tolleranza, cosi'
    le minime differenze dovute a scrittura e rilettura di un NIfTI non
    provocano un ricampionamento inutile. Si usa il vicino piu' prossimo: la
    maschera deve restare binaria.
    """
    if (mask.GetSize() == ref.GetSize()
            and np.allclose(mask.GetSpacing(), ref.GetSpacing(), atol=atol)
            and np.allclose(mask.GetOrigin(), ref.GetOrigin(), atol=atol)
            and np.allclose(mask.GetDirection(), ref.GetDirection(), atol=atol)):
        return mask
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(ref)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetDefaultPixelValue(0)
    return rs.Execute(mask)


def compute_centiloid(pet, voi_ctx, voi_wc, tracer: str, strict: bool = False):
    """SUVr e Centiloide con la formula pubblicata del tracciante.

    pet       PET allineata (sitk.Image), nelle sue unita' originali
    voi_ctx   maschera binaria del target corticale
    voi_wc    maschera binaria del cervelletto intero (regione di riferimento)
    tracer    'FBB', 'FMM' o 'FBP'
    strict    se True, i casi patologici sollevano un'eccezione invece di
              restituire NaN

    Restituisce (media_ctx, media_wc, suvr, cl, info), dove `info` raccoglie
    numero di voxel, formula usata ed eventuali avvisi.

    Casi patologici: se una VOI e' vuota (la segmentazione non ha prodotto le
    etichette attese) il valore non viene forzato ma restituito come NaN, cosi'
    l'errore resta riconoscibile. Se il SUVr esce dall'intervallo plausibile,
    il soggetto viene segnalato in `info["warning"]`.
    """
    # 1. Maschere e PET sulla stessa griglia, poi in array numpy.
    voi_ctx_r = _ensure_same_grid(voi_ctx, pet)
    voi_wc_r = _ensure_same_grid(voi_wc, pet)
    pet_arr = sitk.GetArrayFromImage(pet).astype(np.float32)
    ctx_arr = sitk.GetArrayFromImage(voi_ctx_r) > 0
    wc_arr = sitk.GetArrayFromImage(voi_wc_r) > 0

    info = {"tracer": tracer, "n_ctx_vox": int(ctx_arr.sum()),
            "n_wc_vox": int(wc_arr.sum()),
            "formula": CL_FORMULAS.get(tracer), "warning": None}

    def _empty():
        info["warning"] = "VOI vuota: SUVr non calcolabile"
        return math.nan, math.nan, math.nan, math.nan, info

    # 2. Una regione vuota rende il rapporto privo di senso: NaN, non zero.
    if not ctx_arr.any():
        msg = "Il target corticale e' vuoto sulla griglia della PET."
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
        return _empty()
    if not wc_arr.any():
        msg = "Il cervelletto e' vuoto sulla griglia della PET."
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
        return _empty()

    # 3. Medie della PET nelle due regioni. Media semplice di tutti i voxel
    #    della maschera, senza pesi ne' soglie.
    ctx_voxels = pet_arr[ctx_arr]
    wc_voxels = pet_arr[wc_arr]
    ctx_mean = float(ctx_voxels.mean())
    wc_mean = float(wc_voxels.mean())

    # Il denominatore deve essere positivo: una media nulla o negativa nel
    # cervelletto indica una PET non calibrata o una regione fuori immagine.
    if wc_mean <= 0:
        info["warning"] = f"Media nel cervelletto = {wc_mean}: PET non calibrata."
        if strict:
            raise ValueError(info["warning"])
        logger.warning(info["warning"])
        return ctx_mean, wc_mean, math.nan, math.nan, info

    # 4. SUVr e centiloide con la formula pubblicata ("grezzo").
    suvr = ctx_mean / wc_mean
    cl = suvr_to_cl(suvr, tracer)
    info.update({"ctx_min": float(ctx_voxels.min()),
                 "ctx_max": float(ctx_voxels.max()),
                 "wc_min": float(wc_voxels.min()),
                 "wc_max": float(wc_voxels.max())})

    # 5. Controlli di plausibilita': non cambiano il risultato, lo annotano.
    suvr_lo, suvr_hi = SUVR_PLAUSIBILE
    if suvr < suvr_lo or suvr > suvr_hi:
        msg = (f"SUVr={suvr:.3f} fuori dall'intervallo plausibile "
               f"[{suvr_lo}, {suvr_hi}] (CTX={ctx_mean:.3g}, WC={wc_mean:.3g}): "
               "probabile cervelletto fuori dal campo di vista della PET o "
               "etichette sbagliate.")
        info["warning"] = msg
        info["suvr_out_of_range"] = True
        logger.warning(msg)

    # Scala della PET insolita (conteggi grezzi o immagine gia' normalizzata):
    # il SUVr e' un rapporto e non ne risente, ma lo si annota perche' aiuta a
    # riconoscere immagini di provenienza diversa (tesi, capitolo 4: i 13
    # soggetti florbetapir esclusi hanno una scala anomala).
    pet_global_mean = float(np.mean(np.abs(pet_arr)))
    if pet_global_mean < 0.1 or pet_global_mean > 200:
        logger.warning("Media globale della PET = %.3g: scala insolita.",
                       pet_global_mean)
        info["pet_unusual_scale"] = True
        info["pet_global_mean"] = pet_global_mean

    logger.info("CTX=%.4f  WC=%.4f  SUVr=%.4f  CL=%.2f", ctx_mean, wc_mean, suvr, cl)
    return ctx_mean, wc_mean, suvr, cl, info
