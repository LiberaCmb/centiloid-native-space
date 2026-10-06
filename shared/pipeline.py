"""La pipeline completa su un soggetto, dai file di ingresso al Centiloide.

E' l'unico punto in cui i moduli 00-06 vengono messi in sequenza: la usano sia
la riga di comando (`scripts/run_pipeline.py`, `scripts/batch_process.py`) sia
l'interfaccia (`ui/pipeline_service.py`), che quindi danno risultati identici.

    00 conversione DICOM -> NIfTI (facoltativa)
    01 caricamento di MRI e PET
    02 allineamento rigido PET -> MRI
    03 ricampionamento della PET sulla griglia MRI (+ figura di controllo)
    04 segmentazione della MRI
    05 regioni di interesse (dalle etichette o dall'atlante)
    06 SUVr e Centiloide con la formula pubblicata

Come si costruiscono le regioni dipende dal metodo:

    synthseg, fastsurfer, samseg   dalle etichette FreeSurfer;
    otsu, kmeans, fsl_fast         VOI standard portate nello spazio nativo
                                   ∩ sostanza grigia; la stessa registrazione
                                   del template fornisce la maschera del
                                   cervello alla segmentazione;
    samseg_atlas                   VOI standard ∩ corteccia/cervelletto di
                                   SAMSEG.

I moduli hanno nomi che iniziano con una cifra, quindi si importano con
importlib.
"""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import pandas as pd
import SimpleITK as sitk

from .context import ProjectContext
from .io_utils import load_image

#: I sette metodi confrontati nella tesi.
METODI = ["synthseg", "fastsurfer", "samseg", "samseg_atlas",
          "otsu", "kmeans", "fsl_fast"]
METODI_CLASSICI = ("otsu", "kmeans", "fsl_fast")

# Callback di avanzamento: (fase, frazione completata 0-1, messaggio). Stessa
# firma usata dall'interfaccia (ui/pipeline_service.py).
ProgressFn = Callable[[str, float, str], None]

_log = logging.getLogger("pipeline")


def _moduli():
    """Importa i moduli 00-06 per nome (iniziano con una cifra)."""
    return {nome: importlib.import_module(nome) for nome in (
        "00_DicomToNifti", "01_LoadImages", "02_Realignment", "03_VisualCheck",
        "04_Segmentation", "05_VOIExtraction", "06_CentiloidCalculation")}


@dataclass
class RisultatoSoggetto:
    """Tutto cio' che esce da un'esecuzione. `record` e' la riga di result.csv."""
    record: Dict
    info: Dict
    voi_ctx: sitk.Image
    voi_wc: sitk.Image
    messaggio_allineamento: Optional[str] = None
    figura_allineamento: Optional[str] = None
    avvisi: list = field(default_factory=list)


def metodo_usa_atlante(metodo: str) -> bool:
    """True se le VOI del metodo vengono dall'atlante ICBM152 deformato."""
    return metodo in METODI_CLASSICI or metodo == "samseg_atlas"


def esegui_soggetto(
    mri: str,
    pet: str,
    tracciante: str,
    metodo: str,
    cartella_output: str,
    *,
    registrazione: str = "ants",
    conversione: str = "off",
    co_acquisite: bool = False,
    riusa_allineamento: bool = False,
    dilatazione_ctx: int = 0,
    etichette_ctx: Optional[Sequence[int]] = None,
    etichette_wc: Optional[Sequence[int]] = None,
    seg_kwargs: Optional[Dict] = None,
    id_soggetto: Optional[str] = None,
    dataset: Optional[str] = None,
    trasformazione_atlante: str = "SyN",
    config_atlante=None,
    figura_qc: bool = False,
    progresso: Optional[ProgressFn] = None,
) -> RisultatoSoggetto:
    """Esegue la pipeline su un soggetto e scrive su disco segmentazione e VOI.

    Parametri principali
    --------------------
    mri, pet            file NIfTI o cartelle DICOM; con `riusa_allineamento`
                        sono gia' le immagini allineate del modulo 03.
    tracciante          'FBB', 'FMM' o 'FBP'.
    metodo              uno di METODI.
    registrazione       'ants' (usato nella tesi) o 'sitk'.
    conversione         'off' (gli ingressi si caricano cosi' come sono),
                        'auto' (dcm2niix se c'e', altrimenti SimpleITK),
                        'dcm2niix' o 'sitk' per forzare il motore.
    dilatazione_ctx     voxel di dilatazione del target (0 nella tesi).
    etichette_ctx/_wc   regioni a scelta (solo metodi FreeSurfer; None = standard).
    figura_qc           salva `qc_allineamento.png` (PET sopra la MRI).

    Il file `result.csv` non viene scritto qui ma da `scrivi_risultato`, cosi'
    chi chiama puo' aggiungere colonne prima (l'interfaccia lo fa).
    """
    # Controlli sugli argomenti prima di qualunque calcolo.
    M = _moduli()
    _seg, _voi, _cl = M["04_Segmentation"], M["05_VOIExtraction"], M["06_CentiloidCalculation"]
    progresso = progresso or (lambda fase, frazione, msg: _log.info("%s", msg))
    seg_kwargs = dict(seg_kwargs or {})
    tracciante = tracciante.upper()

    if tracciante not in _cl.CL_FORMULAS:
        raise ValueError(f"Tracciante non supportato: {tracciante}. "
                         f"Disponibili: {sorted(_cl.CL_FORMULAS)}")
    if metodo not in METODI:
        raise ValueError(f"Metodo sconosciuto: {metodo}. Disponibili: {METODI}")
    usa_atlante = metodo_usa_atlante(metodo)
    if usa_atlante and (etichette_ctx is not None or etichette_wc is not None):
        raise ValueError(f"Il metodo {metodo} usa le VOI dell'atlante: le "
                         "regioni a scelta valgono solo per i metodi FreeSurfer.")

    out = Path(cartella_output)
    out.mkdir(parents=True, exist_ok=True)

    ctx = ProjectContext()
    ctx.subject_id = id_soggetto or Path(mri).stem
    ctx.dataset_name = dataset
    ctx.is_co_acquired = co_acquisite
    ctx.output_dir = str(out)
    ctx.selected_tracer = tracciante
    ctx.segmenter_name = metodo
    risultato_reg, figura = None, None

    # Con `riusa_allineamento` i moduli 00-03 si saltano: MRI e PET sono gia'
    # le immagini allineate di un'esecuzione precedente.
    if riusa_allineamento:
        ctx.realigned_anat_path = str(mri)
        ctx.realigned_func_path = str(pet)
    else:
        mri_in, pet_in = str(mri), str(pet)
        # ── 00 conversione ───────────────────────────────────────────────
        if conversione != "off":
            progresso("convert", 0.02, "[00] Conversione DICOM -> NIfTI")
            conv = M["00_DicomToNifti"].DicomToNiftiConverter(
                prefer_dcm2niix=(conversione != "sitk"))
            rm = conv.convert(mri_in, str(out / "converted" / "MR.nii.gz"))
            rp = conv.convert(pet_in, str(out / "converted" / "PET.nii.gz"))
            mri_in, pet_in = rm["output"], rp["output"]

        # ── 01 caricamento ───────────────────────────────────────────────
        progresso("load", 0.08, "[01] Caricamento di MRI e PET")
        ok, msg = M["01_LoadImages"].update_context_with_file(ctx, "anat_1", mri_in)
        if not ok:
            raise RuntimeError(f"Caricamento della MRI non riuscito: {msg}")
        ok, msg = M["01_LoadImages"].update_context_with_file(ctx, "func_1", pet_in)
        if not ok:
            raise RuntimeError(f"Caricamento della PET non riuscito: {msg}")

        # ── 02 allineamento ──────────────────────────────────────────────
        progresso("realign", 0.18,
                  f"[02] Allineamento PET -> MRI ({registrazione.upper()})")
        allinea = M["02_Realignment"].get_registration(registrazione)
        transform, risultato_reg = allinea(ctx)
        if transform is None:
            raise RuntimeError(f"Allineamento non riuscito: {risultato_reg}")
        ctx.registration_transform = transform
        _log.info("    %s", risultato_reg)

        # ── 03 ricampionamento e controllo visivo ────────────────────────
        progresso("save_aligned", 0.30, "[03] Salvataggio delle immagini allineate")
        M["03_VisualCheck"].ImageFusionLogic.save_realigned_images_with_transform(
            ctx, str(out), transform)
        if figura_qc:
            try:
                figura = M["03_VisualCheck"].figura_allineamento(
                    ctx.realigned_anat_path, ctx.realigned_func_path,
                    out / "qc_allineamento.png", titolo=str(ctx.subject_id))
            except Exception as e:                      # la figura non e' critica
                _log.warning("Figura di controllo non generata: %s", e)

    # ── 04 segmentazione ─────────────────────────────────────────────────
    progresso("segment", 0.38, f"[04] Segmentazione con {metodo}")
    mri_sitk = load_image(ctx.realigned_anat_path)
    pet_sitk = load_image(ctx.realigned_func_path)
    # samseg_atlas usa la segmentazione di SAMSEG; cambia solo la costruzione
    # delle VOI (modulo 05).
    segmentatore = _seg.get_segmenter("samseg" if metodo == "samseg_atlas" else metodo)

    cfg = atlante_ctx = atlante_wc = info_warp = None
    if usa_atlante:
        # Una sola registrazione template -> soggetto, riusata per le VOI e,
        # nei metodi classici, per la maschera del cervello.
        per_samseg = metodo == "samseg_atlas"
        cfg = config_atlante or _voi.config_predefinita(
            per_samseg=per_samseg, type_of_transform=trasformazione_atlante)
        progresso("atlas", 0.42, f"[04a] Atlante ICBM152 -> spazio nativo "
                                 f"(ANTs {cfg.type_of_transform})")
        maschera, atlante_ctx, atlante_wc, info_warp = _voi.warp_atlas_to_native(
            mri_sitk, cfg, want_brain_mask=not per_samseg)
        if maschera is not None:
            seg_kwargs["external_brain_mask"] = maschera

    seg = segmentatore.segment(mri=mri_sitk, **seg_kwargs)
    seg_dir = out / "segmentation"
    seg_dir.mkdir(exist_ok=True)
    seg_path = str(seg_dir / f"{metodo}_seg.nii.gz")
    sitk.WriteImage(seg.label_map, seg_path)
    ctx.segmentation_path = seg_path
    ctx.segmentation_label_scheme = seg.label_scheme
    ctx.segmentation_runtime_s = seg.runtime_s

    # ── 05 regioni di interesse ──────────────────────────────────────────
    progresso("voi", 0.80, "[05] Regioni di interesse")
    if usa_atlante:
        voi_ctx, voi_wc, voi_info = _voi.intersect_vois_atlas(
            atlante_ctx, atlante_wc, seg.label_map, cfg, warp_info=info_warp)
        # Dilatazione facoltativa del target, limitata al cervello della
        # segmentazione; il conteggio prima della dilatazione resta in info.
        if dilatazione_ctx and dilatazione_ctx > 0:
            ctx_arr = sitk.GetArrayFromImage(voi_ctx) > 0
            brain = sitk.GetArrayFromImage(seg.label_map)
            dil = _voi.dilata_target(ctx_arr, brain, dilatazione_ctx)
            nuovo = sitk.GetImageFromArray(dil)
            nuovo.CopyInformation(voi_ctx)
            voi_info["ctx_voxels_before_dilation"] = voi_info["ctx_voxels"]
            voi_info["ctx_voxels"] = int(dil.sum())
            voi_info["method"] += f"+dil{dilatazione_ctx}"
            voi_ctx = nuovo
    else:
        voi_ctx, voi_wc, voi_info = _voi.extract_vois(
            seg.label_map, seg.label_scheme,
            ctx_dilation_iters=int(dilatazione_ctx),
            ctx_labels=list(etichette_ctx) if etichette_ctx is not None else None,
            wc_labels=list(etichette_wc) if etichette_wc is not None else None)
    ctx.voi_ctx_path, ctx.voi_wc_path = _voi.save_vois(
        voi_ctx, voi_wc, str(out / "voi"), prefix=f"{metodo}_")
    _log.info("    voxel target=%d  cervelletto=%d  (%s)",
              voi_info["ctx_voxels"], voi_info["wc_voxels"], voi_info["method"])

    # ── 06 SUVr e Centiloide ─────────────────────────────────────────────
    progresso("centiloid", 0.92, "[06] SUVr e Centiloide")
    ctx_mean, wc_mean, suvr, cl, cl_info = _cl.compute_centiloid(
        pet_sitk, voi_ctx, voi_wc, tracciante)
    ctx.ctx_mean, ctx.wc_mean = ctx_mean, wc_mean
    ctx.suvr_value, ctx.centiloid_value = suvr, cl

    # Riga di result.csv e dettagli per info.json; gli avvisi del modulo 06
    # (VOI vuota, SUVr fuori intervallo) risalgono al chiamante.
    record = ctx.to_record()
    record.update({"voi_extraction_method": voi_info["method"],
                   "ctx_voxels": voi_info["ctx_voxels"],
                   "wc_voxels": voi_info["wc_voxels"]})
    info = {"voi": voi_info, "centiloid": cl_info,
            "seg_runtime_s": seg.runtime_s, "seg_extras": seg.extras,
            "registration": risultato_reg}
    avvisi = [str(cl_info["warning"])] if cl_info.get("warning") else []
    return RisultatoSoggetto(record=record, info=info, voi_ctx=voi_ctx,
                             voi_wc=voi_wc, messaggio_allineamento=risultato_reg,
                             figura_allineamento=figura, avvisi=avvisi)


def scrivi_risultato(cartella_output, record: Dict, info: Dict) -> Path:
    """Scrive `result.csv` (una riga) e `info.json` nella cartella del soggetto."""
    out = Path(cartella_output)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record]).to_csv(out / "result.csv", index=False)
    with open(out / "info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    return out / "result.csv"
