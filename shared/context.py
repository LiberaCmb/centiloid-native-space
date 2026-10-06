"""Contesto di un'esecuzione della pipeline e contenitore delle immagini.

I campi descrivono, nell'ordine, le immagini caricate (modulo 01), la
trasformazione dell'allineamento (moduli 02-03) e i risultati della
segmentazione in spazio nativo, delle regioni e del centiloide (moduli 04-06).
"""

from __future__ import annotations
import dataclasses
from typing import Optional, Any, Dict
import numpy as np


@dataclasses.dataclass
class ImageVolume:
    """Un volume caricato da disco, con la sua geometria e i metadati."""

    pixel_data: np.ndarray
    metadata: Any                       # dataset pydicom, oppure dict per i NIfTI
    filepath: str
    modality: str                       # 'MR', 'PT', ...
    affine: Any = None
    spacing: tuple = (1.0, 1.0, 1.0)
    origin: tuple = (0.0, 0.0, 0.0)
    direction: tuple = (1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0)


class ProjectContext:
    """Stato di un'esecuzione su un singolo soggetto."""

    def __init__(self):
        # ── Modulo 01: caricamento ────────────────────────────────────────
        self.anatomical_1: Optional[ImageVolume] = None    # MRI T1
        self.functional_1: Optional[ImageVolume] = None    # PET amiloide
        self.selected_tracer: Optional[str] = None         # 'FBB', 'FMM', 'FBP'

        # ── Modulo 02: allineamento ───────────────────────────────────────
        # True se PET e MRI sono gia' nello stesso spazio: il
        # modulo 02 restituisce allora l'identita'.
        self.is_co_acquired: bool = False
        self.registration_transform: Optional[Any] = None

        # ── Modulo 03: immagini allineate salvate su disco ────────────────
        self.realigned_anat_path: Optional[str] = None
        self.realigned_func_path: Optional[str] = None

        # ── Modulo 04: segmentazione ──────────────────────────────────────
        self.segmenter_name: Optional[str] = None
        self.segmentation_path: Optional[str] = None
        self.segmentation_label_scheme: Optional[str] = None   # 'freesurfer' o 'tissue3'
        self.segmentation_runtime_s: Optional[float] = None

        # ── Modulo 05: regioni di interesse ───────────────────────────────
        self.voi_ctx_path: Optional[str] = None            # target corticale
        self.voi_wc_path: Optional[str] = None             # cervelletto intero

        # ── Modulo 06: SUVr e Centiloide ──────────────────────────────────
        self.ctx_mean: Optional[float] = None
        self.wc_mean: Optional[float] = None
        self.suvr_value: Optional[float] = None
        self.centiloid_value: Optional[float] = None

        # ── Identificazione ───────────────────────────────────────────────
        self.subject_id: Optional[str] = None
        self.dataset_name: Optional[str] = None
        self.output_dir: Optional[str] = None

    def to_record(self) -> Dict[str, Any]:
        """Riga di `result.csv`: i campi che servono alle analisi successive.

        `centiloid` e' quello con la formula pubblicata; i percorsi sono quelli
        dei file intermedi, riletti dagli script di analisi e delle figure.
        """
        return {
            "subject_id":      self.subject_id,
            "dataset":         self.dataset_name,
            "tracer":          self.selected_tracer,
            "segmenter":       self.segmenter_name,
            "ctx_mean":        self.ctx_mean,
            "wc_mean":         self.wc_mean,
            "suvr":            self.suvr_value,
            "centiloid":       self.centiloid_value,
            "seg_runtime_s":   self.segmentation_runtime_s,
            "anat_path":       self.realigned_anat_path,
            "pet_path":        self.realigned_func_path,
            "seg_path":        self.segmentation_path,
            "voi_ctx_path":    self.voi_ctx_path,
            "voi_wc_path":     self.voi_wc_path,
        }
