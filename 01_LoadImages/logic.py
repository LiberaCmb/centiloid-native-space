"""Caricamento di volumi DICOM e NIfTI.

Modulo 01 della pipeline: riceve il percorso della MRI o della PET (cartella
DICOM, un file della serie, oppure un NIfTI/MHA) e produce un `ImageVolume`
riorientato in RAS, che `update_context_with_file` mette nel `ProjectContext`
per l'allineamento del modulo 02.
"""

from __future__ import annotations
import os
import logging
import pydicom
import numpy as np
import SimpleITK as sitk

from shared.context import ImageVolume, ProjectContext

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# 1. Lettura da disco
# ════════════════════════════════════════════════════════════════════════════
class ImageLoader:
    """Lettura di volumi da disco in `ImageVolume`."""

    @staticmethod
    def load_dicom_series(path: str) -> ImageVolume:
        """Carica una serie DICOM (cartella o uno dei suoi file) oppure un
        singolo file NIfTI/MetaImage, e lo riorienta in RAS."""
        # Un file puo' essere un NIfTI oppure uno dei file della serie DICOM:
        # in quel caso serve la sua cartella, e il nome indica quale serie.
        if os.path.isfile(path):
            directory = os.path.dirname(path)
            filename = os.path.basename(path)
        else:
            directory = path
            filename = None

        # ── File singolo NIfTI / MHA ──────────────────────────────────────
        if filename and filename.lower().endswith(('.nii', '.nii.gz', '.mha')):
            try:
                sitk_image = sitk.ReadImage(path)
                orient = sitk.DICOMOrientImageFilter()
                orient.SetDesiredCoordinateOrientation("RAS")
                sitk_image = orient.Execute(sitk_image)

                arr = sitk.GetArrayFromImage(sitk_image)
                spacing = sitk_image.GetSpacing()
                origin = sitk_image.GetOrigin()
                direction = sitk_image.GetDirection()

                # Il NIfTI non porta la modalita': la si deduce dal nome.
                ds = {'Modality': 'Unknown', 'SeriesDescription': filename}
                modality = "Unknown"
                lname = filename.lower()
                if "ct" in lname:
                    modality = "CT"
                elif "mr" in lname or "t1" in lname:
                    modality = "MR"
                elif "pt" in lname or "pet" in lname:
                    modality = "PT"
                elif "nm" in lname or "spect" in lname:
                    modality = "NM"

                # `affine` resta l'identita': la geometria usata dalla
                # pipeline e' in spacing, origin e direction.
                return ImageVolume(
                    pixel_data=arr,
                    metadata=ds,
                    filepath=path,
                    modality=modality,
                    affine=np.eye(4),
                    spacing=spacing,
                    origin=origin,
                    direction=direction,
                )
            except Exception as e:
                raise Exception(f"Lettura del file non DICOM fallita: {e}")

        # ── Serie DICOM ───────────────────────────────────────────────────
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(directory)
        if not series_ids:
            raise Exception("Nessuna serie DICOM trovata nella cartella.")

        # Piu' serie nella stessa cartella: si prende la prima, oppure quella
        # che contiene il file indicato dal chiamante.
        selected_series_id = series_ids[0]
        if filename:
            for sid in series_ids:
                fnames = reader.GetGDCMSeriesFileNames(directory, sid)
                if any(os.path.basename(f) == filename for f in fnames):
                    selected_series_id = sid
                    break

        dicom_names = reader.GetGDCMSeriesFileNames(directory, selected_series_id)
        reader.SetFileNames(dicom_names)

        # Gli avvisi di ITK sono spenti durante la lettura e riaccesi dopo,
        # anche nel ramo di ripiego.
        try:
            sitk.ProcessObject.SetGlobalWarningDisplay(False)
            sitk_image = reader.Execute()
            orient = sitk.DICOMOrientImageFilter()
            orient.SetDesiredCoordinateOrientation("RAS")
            sitk_image = orient.Execute(sitk_image)
            sitk.ProcessObject.SetGlobalWarningDisplay(True)

            arr = sitk.GetArrayFromImage(sitk_image)
            spacing = sitk_image.GetSpacing()
            origin = sitk_image.GetOrigin()
            direction = sitk_image.GetDirection()
        except RuntimeError as e:
            # Ripiego: si impilano le fette una per una.
            sitk.ProcessObject.SetGlobalWarningDisplay(True)
            logger.warning("ImageSeriesReader non riuscito (%s): impilo le "
                           "fette a mano.", e)

            slices = []
            for fname in dicom_names:
                slices.append(sitk.GetArrayFromImage(sitk.ReadImage(fname)))
            if not slices:
                raise Exception("Nessuna fetta leggibile.")
            arr = np.concatenate(slices, axis=0)

            # Geometria dalla prima fetta; il passo fra le fette e' la
            # distanza fra le origini delle prime due, perche' la lettura
            # di una fetta singola non lo conosce.
            first = sitk.ReadImage(dicom_names[0])
            spacing = first.GetSpacing()
            origin = first.GetOrigin()
            direction = first.GetDirection()
            if len(dicom_names) > 1:
                second = sitk.ReadImage(dicom_names[1])
                d = float(np.linalg.norm(np.array(second.GetOrigin()) -
                                         np.array(first.GetOrigin())))
                if d > 0:
                    spacing = (spacing[0], spacing[1], d)

        # Metadati (modalita', tracciante) dall'intestazione del primo file,
        # senza leggere i pixel.
        first_file = dicom_names[0]
        ds = pydicom.dcmread(first_file, stop_before_pixels=True, force=True)
        modality = ds.get("Modality", "Unknown")

        return ImageVolume(
            pixel_data=arr,
            metadata=ds,
            filepath=first_file,
            modality=modality,
            affine=np.eye(4),
            spacing=spacing,
            origin=origin,
            direction=direction,
        )


# ════════════════════════════════════════════════════════════════════════════
# 2. Riconoscimento del tracciante dai metadati DICOM
# ════════════════════════════════════════════════════════════════════════════
# Parole chiave cercate nei campi descrittivi DICOM: nome del farmaco, sigla,
# nome commerciale e codice di sviluppo di ciascun tracciante.
TRACER_KEYWORDS = {
    'FBB': ['florbetaben', 'fbb', 'neuraceq', 'bay94'],
    'FMM': ['flutemetamol', 'fmm', 'vizamyl', 'flute'],
    'FBP': ['florbetapir', 'fbp', 'av-45', 'av45', 'amyvid'],
}


def detect_tracer(metadata) -> str:
    """Restituisce 'FBB', 'FMM', 'FBP' oppure 'Unknown'.

    `metadata` puo' essere un dataset pydicom (campi come attributi) o il
    dict prodotto per i NIfTI (campi come chiavi). Vince il primo tracciante
    di TRACER_KEYWORDS le cui parole compaiono nei campi letti.
    """
    fields = []
    for tag in ('Radiopharmaceutical', 'SeriesDescription', 'StudyDescription'):
        try:
            v = getattr(metadata, tag, None)
            if v is None and isinstance(metadata, dict):
                v = metadata.get(tag)
            if v:
                fields.append(str(v))
        except Exception:
            pass
    combined = ' '.join(fields).lower()
    for code, kws in TRACER_KEYWORDS.items():
        if any(k in combined for k in kws):
            return code
    return 'Unknown'


# ════════════════════════════════════════════════════════════════════════════
# 3. Aggiornamento del contesto
# ════════════════════════════════════════════════════════════════════════════
def update_context_with_file(context: ProjectContext,
                             slot_name: str,
                             file_path: str):
    """Carica `file_path` nel contesto come MRI ('anat_1') o PET ('func_1').

    Il tracciante viene dedotto dai metadati solo se il chiamante non l'ha gia'
    indicato: sui NIfTI la deduzione restituisce 'Unknown' e non deve
    sovrascrivere la scelta esplicita.

    Restituisce (ok, messaggio): gli errori non vengono sollevati ma
    riportati nel messaggio, e chi chiama decide cosa farne.
    """
    try:
        volume = ImageLoader.load_dicom_series(file_path)
        if slot_name == "anat_1":
            context.anatomical_1 = volume
        elif slot_name == "func_1":
            context.functional_1 = volume

        if (slot_name == "func_1" and volume.modality in ("PT", "NM")
                and not context.selected_tracer):
            context.selected_tracer = detect_tracer(volume.metadata)

        return True, f"Caricato {volume.modality} da {os.path.basename(file_path)}"
    except Exception as e:
        return False, str(e)
