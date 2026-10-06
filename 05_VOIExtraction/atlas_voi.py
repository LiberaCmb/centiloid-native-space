"""VOI dall'atlante per i metodi senza parcellazione (classici e samseg_atlas).

I metodi classici distinguono solo tre tessuti (liquor, sostanza grigia,
sostanza bianca): non sanno dove finisce la corteccia e dove comincia il
cervelletto. L'informazione anatomica arriva quindi dall'esterno, dalle VOI
standard del Centiloid Project:

  1. il template ICBM152 viene registrato sulla MRI del soggetto (affine +
     SyN, perche' servono deformazioni non lineari per far combaciare due
     anatomie diverse);
  2. le VOI standard e la maschera del cervello del template vengono portate
     nello spazio nativo con la trasformazione ottenuta;
  3. le VOI vengono intersecate con la segmentazione del metodo:

        target      = VOI corticale standard  ∩  sostanza grigia
        riferimento = VOI del cervelletto      ∩  (sostanza grigia ∪ bianca)

Il paziente non viene mai deformato: si muove il template. La VOI dice *dove*
sta la regione, la segmentazione dice *quali* voxel di quella regione sono
sostanza grigia.

La stessa costruzione, applicata alle etichette di SAMSEG (corteccia 3/42,
cervelletto 7/8/46/47) invece che ai tessuti, da' il metodo `samseg_atlas`.

GEOMETRIA
---------
Le VOI GAAIN (voi_ctx_1mm, voi_WhlCbl_1mm) sono nello spazio ICBM152 lineare a
1 mm: griglia 181x217x181, orientazione LAS. Il template di registrazione deve
quindi essere `icbm_avg_152_t1_tal_lin`, non l'MNI152 di FSL (182x218x182);
`check_template_matches_voi()` lo verifica.

ANTsPy viene importato solo dentro le funzioni che lo usano.
"""
from __future__ import annotations
import logging
import os
import tempfile
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

from .label_mappings import TISSUE3_GM_LABEL, TISSUE3_WM_LABEL

logger = logging.getLogger(__name__)

# Griglia delle VOI GAAIN (ICBM152 lineare, 1 mm), vedi GEOMETRIA sopra.
_EXPECTED_VOI_SHAPE = (181, 217, 181)


@dataclass
class AtlasConfig:
    """File dell'atlante e opzioni dell'intersezione."""
    template_path: str            # template ICBM152 lineare T1
    voi_ctx_path: str             # VOI corticale standard
    voi_wc_path: str              # VOI del cervelletto intero
    # Maschera del cervello del template: portata nello spazio nativo diventa
    # la maschera comune dei tre metodi classici. Vuota per samseg_atlas, che
    # segmenta il cervello da solo.
    brain_mask_path: str = ""
    gm_label: int = TISSUE3_GM_LABEL
    wm_label: int = TISSUE3_WM_LABEL
    # Il cervelletto e' sostanza grigia + bianca; True lo limita alla grigia.
    restrict_wc_to_gm: bool = False
    # Etichette esplicite (samseg_atlas): se `cortex_labels` non e' vuoto si
    # usano queste invece della convenzione a tre tessuti.
    cortex_labels: tuple = ()
    reference_labels: tuple = ()
    # Trasformazione ANTs template -> soggetto ("SyN" = affine + non lineare)
    type_of_transform: str = "SyN"

    def validate_files(self):
        """Solleva FileNotFoundError se manca template, VOI ctx o VOI wc."""
        for label, p in (("template", self.template_path),
                         ("voi_ctx", self.voi_ctx_path),
                         ("voi_wc", self.voi_wc_path)):
            if not p or not os.path.exists(p):
                raise FileNotFoundError(
                    f"File dell'atlante non trovato ({label}): {p!r}. Servono "
                    "il template ICBM152 lineare e le VOI standard del "
                    "Centiloid Project (vedi resources/atlante/LEGGIMI.md).")


def _orient(img, code="RAS"):
    """Riorienta senza perdita (usa la matrice affine). Il template e' RAS, le
    VOI GAAIN sono LAS: stesso spazio fisico, asse x memorizzato al contrario.
    """
    f = sitk.DICOMOrientImageFilter()
    f.SetDesiredCoordinateOrientation(code)
    return f.Execute(img)


def check_template_matches_voi(cfg: AtlasConfig) -> dict:
    """Verifica che template e VOI stiano sulla stessa griglia.

    Errore se le forme non coincidono; avviso se meno del 90% della VOI
    corticale cade dentro il cervello del template.

    Restituisce un dizionario con dimensioni, passi, percentuale della VOI
    dentro il cervello e l'esito (`ok`).
    """
    # 1. Lettura nella stessa orientazione (RAS) e confronto delle griglie.
    t = _orient(sitk.ReadImage(cfg.template_path))
    ctx = _orient(sitk.ReadImage(cfg.voi_ctx_path))
    wc = _orient(sitk.ReadImage(cfg.voi_wc_path))

    rep = {
        "template_size": t.GetSize(), "voi_ctx_size": ctx.GetSize(),
        "voi_wc_size": wc.GetSize(),
        "template_spacing": tuple(round(x, 3) for x in t.GetSpacing()),
        "voi_spacing": tuple(round(x, 3) for x in ctx.GetSpacing()),
    }
    if ctx.GetSize() != wc.GetSize():
        raise ValueError(f"Le VOI ctx {ctx.GetSize()} e wc {wc.GetSize()} "
                         "hanno griglie diverse.")
    if tuple(ctx.GetSize()) != _EXPECTED_VOI_SHAPE:
        logger.warning("Griglia delle VOI %s diversa da quella ICBM152 "
                       "lineare %s.", ctx.GetSize(), _EXPECTED_VOI_SHAPE)
    if t.GetSize() != ctx.GetSize():
        raise ValueError(
            f"Griglia del template {t.GetSize()} diversa da quella delle VOI "
            f"{ctx.GetSize()}: serve icbm_avg_152_t1_tal_lin (181x217x181), "
            "non l'MNI152 di FSL.")
    # 2. Sovrapposizione: la VOI corticale deve cadere quasi tutta nel
    #    cervello del template. Il "cervello" e' approssimato dai voxel sopra
    #    il 40-esimo percentile delle intensita' positive: soglia grezza, ma
    #    basta per un controllo. Si richiede almeno il 90%: con uno spazio
    #    sbagliato una parte consistente della VOI esce dal cervello.
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(t)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetDefaultPixelValue(0)
    voi_on_t = sitk.GetArrayFromImage(rs.Execute(ctx)) > 0
    t_arr = sitk.GetArrayFromImage(sitk.Cast(t, sitk.sitkFloat32))
    pos = t_arr[t_arr > 0]
    brain = t_arr > float(np.percentile(pos, 40)) if pos.size else t_arr > 0
    inside = 100.0 * float((voi_on_t & brain).sum()) / max(1, int(voi_on_t.sum()))
    rep["ctx_voi_inside_brain_pct"] = round(inside, 1)
    if inside < 90.0:
        logger.warning("Solo il %.1f%% della VOI corticale cade nel cervello "
                       "del template: possibile spazio sbagliato.", inside)
    rep["ok"] = inside >= 90.0
    return rep


# Il passaggio fra SimpleITK e ANTsPy avviene tramite un NIfTI su disco, che
# porta con se' la geometria (origine, passo, direzione) nei due sensi.
def _sitk_to_ants(img_sitk, tmpdir, name):
    """sitk.Image -> ants.ANTsImage, attraverso <tmpdir>/<name>.nii.gz."""
    import ants
    p = os.path.join(tmpdir, f"{name}.nii.gz")
    sitk.WriteImage(img_sitk, p)
    return ants.image_read(p)


def _ants_to_sitk(img_ants, tmpdir, name):
    """ants.ANTsImage -> sitk.Image, attraverso <tmpdir>/<name>.nii.gz."""
    p = os.path.join(tmpdir, f"{name}.nii.gz")
    img_ants.to_filename(p)
    return sitk.ReadImage(p)


def warp_atlas_to_native(native_t1_sitk, cfg: AtlasConfig, want_brain_mask=True):
    """Registra il template sulla MRI del soggetto UNA volta e porta nello
    spazio nativo VOI corticale, VOI del cervelletto e (se richiesta) maschera
    del cervello. Interpolazione al vicino piu' prossimo: le maschere restano
    binarie.

    native_t1_sitk    MRI T1 del soggetto (immagine fissa della registrazione)
    cfg               file dell'atlante e tipo di trasformazione
    want_brain_mask   porta anche la maschera del cervello, se configurata

    Restituisce (maschera_cervello_o_None, ctx_atlante, wc_atlante, info).
    """
    import ants
    cfg.validate_files()

    with tempfile.TemporaryDirectory() as tmp:
        # 1. Registrazione template (mobile) -> MRI nativa (fissa). Template
        #    e VOI passano entrambi per `_orient`, cosi' condividono lo spazio.
        fixed = _sitk_to_ants(sitk.Cast(native_t1_sitk, sitk.sitkFloat32),
                              tmp, "native_t1")
        template = _sitk_to_ants(
            _orient(sitk.ReadImage(cfg.template_path, sitk.sitkFloat32)),
            tmp, "template")

        logger.info("ANTs %s: template ICBM152 -> MRI nativa ...",
                    cfg.type_of_transform)
        reg = ants.registration(fixed=fixed, moving=template,
                                type_of_transform=cfg.type_of_transform)
        # trasformazioni dirette template -> soggetto, nell'ordine in cui
        # `apply_transforms` le vuole
        tl = reg["fwdtransforms"]

        # 2. Stessa trasformazione applicata a ciascuna maschera dell'atlante.
        def _warp(path, name):
            mov = _sitk_to_ants(_orient(sitk.ReadImage(path)), tmp, name)
            out = ants.apply_transforms(fixed=fixed, moving=mov,
                                        transformlist=tl,
                                        interpolator="nearestNeighbor")
            return _ants_to_sitk(out, tmp, name + "_nat")

        ctx_sitk = _warp(cfg.voi_ctx_path, "voi_ctx")
        wc_sitk = _warp(cfg.voi_wc_path, "voi_wc")
        bm_sitk = None
        if want_brain_mask and cfg.brain_mask_path:
            bm_sitk = _warp(cfg.brain_mask_path, "brain_mask")

    info = {"type_of_transform": cfg.type_of_transform,
            "ctx_atlas_voxels": int((sitk.GetArrayFromImage(ctx_sitk) > 0).sum()),
            "wc_atlas_voxels": int((sitk.GetArrayFromImage(wc_sitk) > 0).sum()),
            "brain_mask_source": ("atlas_icbm" if bm_sitk is not None else None)}
    return bm_sitk, ctx_sitk, wc_sitk, info


def _resample_like(moving_sitk, reference_sitk):
    """Etichette di `moving_sitk` sulla griglia di `reference_sitk` (vicino
    piu' prossimo)."""
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(reference_sitk)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetDefaultPixelValue(0)
    return rs.Execute(moving_sitk)


def intersect_vois_atlas(ctx_atlas_sitk, wc_atlas_sitk, seg_sitk,
                         cfg: AtlasConfig, warp_info=None):
    """Interseca le VOI dell'atlante (gia' nello spazio nativo) con la
    segmentazione del metodo.

    Separata da `warp_atlas_to_native` perche' la registrazione, il passo
    lento, si fa una volta sola e serve anche a dare la maschera del cervello
    alla segmentazione.

    ctx_atlas_sitk, wc_atlas_sitk   VOI dell'atlante nello spazio nativo
    seg_sitk                        segmentazione del metodo (modulo 04)
    cfg                             etichette da usare e opzioni
    warp_info                       info della registrazione, copiate in `info`

    Restituisce (voi_ctx, voi_wc, info).
    """
    # 1. Segmentazione sulla griglia delle VOI, se non lo e' gia'.
    ctx_atlas, wc_atlas = ctx_atlas_sitk, wc_atlas_sitk
    seg = seg_sitk
    if seg.GetSize() != ctx_atlas.GetSize():
        seg = _resample_like(seg, ctx_atlas)

    seg_arr = sitk.GetArrayFromImage(seg)
    ctx_a = sitk.GetArrayFromImage(ctx_atlas) > 0
    wc_a = sitk.GetArrayFromImage(wc_atlas) > 0

    # 2. Quali voxel della segmentazione contano come corteccia e come
    #    tessuto di riferimento.
    if cfg.cortex_labels:
        # samseg_atlas: etichette FreeSurfer esplicite
        cortex_tissue = np.isin(seg_arr, list(cfg.cortex_labels))
        ref_tissue = (np.isin(seg_arr, list(cfg.reference_labels))
                      if cfg.reference_labels else cortex_tissue)
        method = ("atlas_native (Centiloid_Std_VOI ∩ SAMSEG-aseg "
                  "cortex%s)" % (str(tuple(cfg.cortex_labels))))
    else:
        # metodi classici: sostanza grigia (e bianca per il cervelletto)
        gm = seg_arr == cfg.gm_label
        wm = seg_arr == cfg.wm_label
        cortex_tissue = gm
        ref_tissue = gm if cfg.restrict_wc_to_gm else (gm | wm)
        method = "atlas_native (Centiloid_Std_VOI ∩ classical-GM)"

    # 3. Intersezione: la VOI dice dove, la segmentazione quale tessuto.
    ctx_mask = ctx_a & cortex_tissue
    wc_mask = wc_a & ref_tissue

    def _to_img(mask):
        m = sitk.GetImageFromArray(mask.astype(np.uint8))
        m.CopyInformation(ctx_atlas)
        return m

    info = {
        "method": method,
        **(warp_info or {}),
        "ctx_voxels": int(ctx_mask.sum()),
        "wc_voxels": int(wc_mask.sum()),
        "ctx_atlas_only_voxels": int(ctx_a.sum()),
        "wc_atlas_only_voxels": int(wc_a.sum()),
        "restrict_wc_to_gm": cfg.restrict_wc_to_gm,
        "cortex_labels": list(cfg.cortex_labels),
        "reference_labels": list(cfg.reference_labels),
    }
    if ctx_mask.sum() == 0 or wc_mask.sum() == 0:
        logger.warning("VOI vuota dopo l'intersezione (ctx=%d, wc=%d): "
                       "controllare la registrazione e le etichette.",
                       int(ctx_mask.sum()), int(wc_mask.sum()))
    return _to_img(ctx_mask), _to_img(wc_mask), info


def config_predefinita(per_samseg: bool = False,
                       type_of_transform: str = "SyN") -> AtlasConfig:
    """Configurazione con i file di `resources/atlante` (vedi shared/percorsi.py).

    per_samseg=False  metodi classici: intersezione con la sostanza grigia e
                      maschera del cervello dal template;
    per_samseg=True   samseg_atlas: intersezione con le etichette di SAMSEG,
                      nessuna maschera (SAMSEG segmenta il cervello da solo).
    """
    from shared import percorsi
    from .label_mappings import SAMSEG_CORTEX_LABELS, SAMSEG_CEREB_LABELS
    return AtlasConfig(
        template_path=str(percorsi.TEMPLATE_ICBM),
        voi_ctx_path=str(percorsi.VOI_CTX_STD),
        voi_wc_path=str(percorsi.VOI_WC_STD),
        brain_mask_path="" if per_samseg else str(percorsi.MASCHERA_ICBM),
        cortex_labels=SAMSEG_CORTEX_LABELS if per_samseg else (),
        reference_labels=SAMSEG_CEREB_LABELS if per_samseg else (),
        type_of_transform=type_of_transform)
