"""Il servizio che l'interfaccia chiama per elaborare un paziente.

`run_single(...)` esegue la stessa pipeline della riga di comando
(`shared/pipeline.py`) e aggiunge cio' che serve all'interfaccia:

  * la barra di avanzamento (callback `progress(fase, frazione, messaggio)`);
  * le regioni a scelta per target e riferimento;
  * il Centiloide RICALIBRATO, quando le regioni sono quelle standard;
  * le anteprime delle VOI sopra MRI e PET;
  * un risultato strutturato, con l'errore leggibile se qualcosa va storto.

Gli ingressi passano sempre dal modulo 00 (cartella DICOM convertita, NIfTI
riorientato in RAS), cosi' l'interfaccia accetta entrambi i formati.
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import percorsi                                     # noqa: E402

# Firma della callback di avanzamento: (fase, frazione in [0, 1], messaggio).
ProgressFn = Callable[[str, float, str], None]

# L'interfaccia usa solo SynthSeg: il metodo con l'errore piu' basso su FBB e
# FMM, il piu' stabile e fra i piu' rapidi. Gli altri restano nella pipeline.
SUPPORTED_SEGMENTERS = ["synthseg"]
SUPPORTED_TRACERS = ["FMM", "FBB", "FBP"]

# ═══════════════════════════════════════════════════════════════════════════
# Coefficienti di ricalibrazione
# ═══════════════════════════════════════════════════════════════════════════
# Il SUVr nativo e' sistematicamente piu' basso di quello in spazio standard su
# cui e' tarata la formula pubblicata, quindi l'interfaccia usa CL = a*SUVr + b
# con a e b stimati per regressione sui dati GAAIN. L'unica fonte e'
#     data/comparison_all_methods/recalibration_final.csv
# scritto da scripts/analisi/build_final_comparison.py: se lo si rigenera,
# l'interfaccia legge i nuovi valori da sola.
RECAL_COEFFS_FILE = percorsi.COEFFICIENTI
RECAL_METHOD = "synthseg"

# Valori di riserva, usati SOLO se il CSV manca o e' illeggibile (l'interfaccia
# lo dichiara). Sono quelli del file finale; non vanno aggiornati a mano.
RECAL_COEFFS_FALLBACK = {
    "FBB": (162.6199, -158.4498),
    "FMM": (131.6780, -124.2663),
    "FBP": (181.9503, -175.9127),
}


def load_recal_coeffs(method: str = RECAL_METHOD,
                      path: Optional[Path] = None) -> tuple[Dict, str]:
    """({tracciante: (a, b)}, descrizione della fonte) dal CSV dei coefficienti."""
    p = Path(path) if path is not None else RECAL_COEFFS_FILE
    try:
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        coeffs = {str(r["tracciante"]).strip().upper(): (float(r["a"]), float(r["b"]))
                  for r in rows
                  if str(r.get("metodo", "")).strip() == method
                  and str(r.get("calibrazione", "regressione")).strip() == "regressione"}
        if not coeffs:
            raise ValueError(f"nessuna riga per metodo='{method}'")
        return coeffs, f"{p.name} (metodo {method}, regressione)"
    except Exception as e:
        return dict(RECAL_COEFFS_FALLBACK), (
            f"valori di riserva scritti nel codice — {p.name} non leggibile ({e})")


# Letti una volta all'import: la fonte (CSV o valori di riserva) viene poi
# riportata nei dettagli del risultato.
RECAL_COEFFS_SYNTHSEG, RECAL_COEFFS_SOURCE = load_recal_coeffs()


# ═══════════════════════════════════════════════════════════════════════════
# Quando il Centiloide ha senso
# ═══════════════════════════════════════════════════════════════════════════
# Il SUVr e' un rapporto fra due medie e resta definito per qualunque coppia
# di regioni. Il Centiloide no: la retta CL = a*SUVr + b e' stimata su UNA
# coppia precisa di VOI (il target della specifica Centiloid e il cervelletto
# intero, senza dilatazione). Su regioni diverse il numero avrebbe l'aspetto di
# un Centiloide senza esserlo: in quel caso non lo si mostra affatto.

@lru_cache(maxsize=1)
def _label_standard() -> tuple:
    """Le due liste canoniche, lette da 05_VOIExtraction/label_mappings.py.

    Il file si carica per percorso, senza importare il pacchetto (che
    porterebbe con se' SimpleITK): servono solo due insiemi di interi.
    """
    import importlib.util as _u
    spec = _u.spec_from_file_location(
        "_label_mappings", ROOT / "05_VOIExtraction" / "label_mappings.py")
    lm = _u.module_from_spec(spec)
    spec.loader.exec_module(lm)
    return frozenset(lm.FREESURFER_CTX_LABELS), frozenset(lm.FREESURFER_WC_LABELS)


def centiloide_applicabile(ctx_labels: Optional[Sequence[int]],
                           wc_labels: Optional[Sequence[int]],
                           ctx_dilation_iters: int = 0) -> tuple[bool, str]:
    """(applicabile, motivo). None significa "regioni standard".

    Una lista esplicita va bene se coincide con quella canonica: chi
    ricostruisce a mano la selezione standard ottiene comunque il Centiloide.
    """
    try:
        std_ctx, std_wc = _label_standard()
    except Exception:
        return False, ("impossibile leggere le etichette standard da "
                       "05_VOIExtraction/label_mappings.py")
    problemi = []
    if ctx_labels is not None and set(ctx_labels) != std_ctx:
        problemi.append("il target corticale non è quello della specifica Centiloid")
    if wc_labels is not None and set(wc_labels) != std_wc:
        problemi.append("la regione di riferimento non è il cervelletto intero")
    if int(ctx_dilation_iters or 0) > 0:
        problemi.append("il target è dilatato, mentre i coefficienti sono "
                        "stimati su VOI non dilatate")
    return (not problemi), "; ".join(problemi)


@dataclass
class PipelineResult:
    """Esito di run_single, letto dall'interfaccia per disegnare i risultati.

    Con ok=False sono valorizzati solo `error` (messaggio breve) ed
    `error_detail` (traceback); i campi numerici restano None.
    """
    ok: bool
    subject_id: str
    tracer: str
    segmenter: str
    suvr: Optional[float] = None
    centiloid: Optional[float] = None            # ricalibrato
    centiloid_standard: Optional[float] = None   # con la formula pubblicata
    centiloid_ok: bool = True
    centiloid_reason: str = ""
    recal_source: Optional[str] = None
    ctx_mean: Optional[float] = None
    wc_mean: Optional[float] = None
    ctx_voxels: Optional[int] = None
    wc_voxels: Optional[int] = None
    seg_runtime_s: Optional[float] = None
    output_dir: Optional[str] = None
    overlay_mri_path: Optional[str] = None
    overlay_pet_path: Optional[str] = None
    registration_message: Optional[str] = None
    voi_method: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    error_detail: Optional[str] = None
    info: Dict = field(default_factory=dict)


def _normalize_path(p: Optional[str]) -> Optional[str]:
    """Accetta percorsi Windows anche quando l'app gira in WSL.

    Toglie virgolette e spazi e, su Linux, trasforma 'C:\\Users\\...' in
    '/mnt/c/Users/...'.
    """
    if not p:
        return p
    p = p.strip().strip('"').strip("'")
    if os.name == "posix":
        m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
        if m:
            return f"/mnt/{m.group(1).lower()}/{m.group(2).replace(chr(92), '/')}"
    return p


def run_single(mri_path: str, pet_path: str, tracer: str, segmenter: str,
               output_dir: str, *, registration: str = "ants",
               co_acquired: bool = False, ctx_dilation_iters: int = 0,
               ctx_labels: Optional[Sequence[int]] = None,
               wc_labels: Optional[Sequence[int]] = None,
               seg_kwargs: Optional[Dict] = None,
               subject_id: Optional[str] = None, dataset: Optional[str] = None,
               progress: Optional[ProgressFn] = None) -> PipelineResult:
    """Conversione -> allineamento -> segmentazione -> VOI -> Centiloide.

    mri_path, pet_path   NIfTI o cartella DICOM (anche percorso Windows)
    tracer               'FMM', 'FBB' o 'FBP'
    segmenter            metodo di segmentazione (l'interfaccia usa 'synthseg')
    output_dir           cartella dove scrivere immagini, VOI e result.csv
    registration         backend di allineamento, 'ants' o 'sitk'
    co_acquired          True se PET e MRI sono gia' co-registrate
    ctx_dilation_iters   iterazioni di dilatazione del target (0 = nessuna)
    ctx_labels, wc_labels  etichette FreeSurfer di target e riferimento;
                         None = regioni standard del Centiloide
    seg_kwargs           opzioni passate al segmentatore
    subject_id, dataset  identificativi scritti nel risultato
    progress             callback di avanzamento (vedi ProgressFn)

    Restituisce un PipelineResult; non solleva eccezioni: un errore qualunque
    finisce in `error`/`error_detail` con ok=False.
    """
    progress = progress or (lambda fase, frazione, msg: None)
    tracer = tracer.upper()
    mri_path = _normalize_path(mri_path)
    pet_path = _normalize_path(pet_path)
    result = PipelineResult(ok=False, subject_id=subject_id or Path(mri_path).stem,
                            tracer=tracer, segmenter=segmenter)
    try:
        from shared.pipeline import esegui_soggetto, scrivi_risultato
        import importlib
        _visual = importlib.import_module("03_VisualCheck")

        if tracer not in SUPPORTED_TRACERS:
            raise ValueError(f"Tracciante non supportato: {tracer}. "
                             f"Scegli fra {SUPPORTED_TRACERS}.")

        ris = esegui_soggetto(
            mri_path, pet_path, tracer, segmenter, output_dir,
            registrazione=registration, conversione="auto",
            co_acquisite=co_acquired, dilatazione_ctx=int(ctx_dilation_iters),
            etichette_ctx=ctx_labels, etichette_wc=wc_labels,
            seg_kwargs=seg_kwargs, id_soggetto=result.subject_id,
            dataset=dataset, progresso=progress)
        # 1. Pipeline comune (moduli 00-06), identica alla riga di comando.
        rec, info = ris.record, ris.info
        result.registration_message = ris.messaggio_allineamento
        result.warnings.extend(ris.avvisi)
        voi = info["voi"]
        # Una VOI vuota non blocca il run (il SUVr esce NaN) ma va segnalata.
        if voi.get("ctx_voxels", 0) == 0:
            result.warnings.append("Il target corticale è vuoto: le etichette "
                                   "scelte non compaiono nella segmentazione.")
        if voi.get("wc_voxels", 0) == 0:
            result.warnings.append("La regione di riferimento è vuota: controllare "
                                   "le etichette o il campo di vista della PET.")

        # ── 2. Ricalibrazione, solo se le regioni sono quelle standard ───
        suvr, cl_standard = rec["suvr"], rec["centiloid"]
        cl_ok, motivo = centiloide_applicabile(ctx_labels, wc_labels,
                                               ctx_dilation_iters)
        cl = None
        if not cl_ok:
            cl_standard = None           # nemmeno il valore grezzo, per non usarlo per sbaglio
            info["centiloid"]["recalibration"] = None
            info["centiloid"]["centiloide_non_calcolato"] = motivo
        elif tracer in RECAL_COEFFS_SYNTHSEG and suvr is not None and not math.isnan(suvr):
            a, b = RECAL_COEFFS_SYNTHSEG[tracer]
            cl = a * suvr + b
            info["centiloid"]["recalibration"] = {
                "method": f"{RECAL_METHOD}_regression", "a": a, "b": b,
                "source": RECAL_COEFFS_SOURCE, "cl_standard": cl_standard}

        # ── 3. Anteprime delle VOI sopra MRI e PET ───────────────────────
        # Un errore nelle figure non invalida il calcolo: diventa un avviso.
        progress("preview", 0.96, "Anteprima delle VOI su MRI e PET…")
        out = Path(output_dir)
        try:
            result.overlay_mri_path = _visual.figura_voi(
                rec["anat_path"], rec["voi_ctx_path"], rec["voi_wc_path"],
                out / "voi_overlay_mri.png",
                titolo="VOI su MRI — corteccia (rosso) · cervelletto (azzurro)")
            result.overlay_pet_path = _visual.figura_voi(
                rec["pet_path"], rec["voi_ctx_path"], rec["voi_wc_path"],
                out / "voi_overlay_pet.png", ricampiona_su=rec["anat_path"],
                titolo="VOI su PET — corteccia (rosso) · cervelletto (azzurro)")
        except Exception as e:
            result.warnings.append(f"Anteprima non generata: {e}")

        # ── 4. result.csv: il Centiloide salvato e' quello ricalibrato ───
        rec.update({"centiloid": cl,
                    "ctx_dilation_iters": int(ctx_dilation_iters),
                    "centiloide_applicabile": bool(cl_ok),
                    "centiloide_motivo": motivo})
        scrivi_risultato(out, rec, info)

        # 5. Risultato strutturato per l'interfaccia.
        result.suvr, result.centiloid = suvr, cl
        result.centiloid_standard = cl_standard
        result.centiloid_ok, result.centiloid_reason = cl_ok, motivo
        result.recal_source = RECAL_COEFFS_SOURCE
        result.ctx_mean, result.wc_mean = rec["ctx_mean"], rec["wc_mean"]
        result.ctx_voxels, result.wc_voxels = voi.get("ctx_voxels"), voi.get("wc_voxels")
        result.voi_method = voi.get("method")
        result.seg_runtime_s = rec["seg_runtime_s"]
        result.output_dir = str(out)
        result.info = {"voi": voi, "centiloid": info["centiloid"]}
        result.ok = True
        progress("done", 1.0, "Completato.")
        return result

    except Exception as e:                       # qualunque errore arriva all'interfaccia
        result.ok = False
        result.error = str(e)
        result.error_detail = traceback.format_exc()
        progress("error", 1.0, f"Errore: {e}")
        return result
