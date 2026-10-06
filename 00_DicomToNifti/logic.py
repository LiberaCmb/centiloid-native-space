"""Conversione DICOM -> NIfTI: dcm2niix se disponibile, altrimenti SimpleITK.

Punto d'ingresso:
    convert_to_nifti(src, dst, ...) -> dict
dove `src` e' una cartella DICOM oppure un file .nii/.nii.gz gia' esistente e
`dst` e' il .nii.gz da scrivere. Restituisce un piccolo resoconto:
    {"output", "engine", "sidecar", "shape", "spacing", "note"}

Entrambi i motori riorientano il volume in RAS con lo stesso filtro di
SimpleITK usato dal modulo 01, cosi' la geometria su disco coincide con quella
che la pipeline ricostruisce in memoria.

Il modulo e' facoltativo: `shared/pipeline.py` lo chiama solo con
`conversione` diversa da "off", e scrive i NIfTI in
`<cartella_output>/converted/MR.nii.gz` e `PET.nii.gz`.
"""
from __future__ import annotations
import importlib
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)

# Estensioni riconosciute come NIfTI (confronto in minuscolo).
_NIFTI_SUFFIXES = (".nii", ".nii.gz")


# ────────────────────────────────────────────────────────────────────────────
# Ricerca dell'eseguibile dcm2niix
# ────────────────────────────────────────────────────────────────────────────
def find_dcm2niix(explicit: str | None = None) -> str | None:
    """Trova dcm2niix: percorso esplicito, poi PATH, poi il pacchetto pip."""
    if explicit and os.path.exists(explicit):
        return explicit
    on_path = shutil.which("dcm2niix")
    if on_path:
        return on_path
    # il pacchetto pip `dcm2niix` contiene l'eseguibile nella sua cartella
    try:
        pkg = importlib.import_module("dcm2niix")
        cand = os.path.join(os.path.dirname(pkg.__file__), "dcm2niix")
        if os.path.exists(cand):
            return cand
        if os.path.exists(cand + ".exe"):          # Windows
            return cand + ".exe"
    except Exception:
        pass
    return None


# ────────────────────────────────────────────────────────────────────────────
# Funzioni di appoggio
# ────────────────────────────────────────────────────────────────────────────
def _is_nifti(path: str) -> bool:
    """True se `path` e' gia' un file NIfTI (da non convertire)."""
    return path.lower().endswith(_NIFTI_SUFFIXES)


def _orient_ras_and_write(sitk_img, dst: str):
    """Riorienta in RAS (come il modulo 01) e scrive il .nii.gz."""
    orient = sitk.DICOMOrientImageFilter()
    orient.SetDesiredCoordinateOrientation("RAS")
    sitk_img = orient.Execute(sitk_img)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    sitk.WriteImage(sitk_img, dst)
    return sitk_img


def _report(sitk_img, dst, engine, sidecar, note=""):
    """Resoconto della conversione restituito da `convert_to_nifti`."""
    return {
        "output": dst,
        "engine": engine,
        "sidecar": sidecar,
        "shape": tuple(int(x) for x in sitk_img.GetSize()),
        "spacing": tuple(round(float(x), 4) for x in sitk_img.GetSpacing()),
        "note": note,
    }


# ────────────────────────────────────────────────────────────────────────────
# Motori
# ────────────────────────────────────────────────────────────────────────────
def _passthrough_nifti(src: str, dst: str):
    """Ingresso gia' NIfTI: solo riorientamento in RAS e riscrittura."""
    img = sitk.ReadImage(src)
    img = _orient_ras_and_write(img, dst)
    return _report(img, dst, "passthrough-nifti", None,
                   note="l'ingresso era gia' NIfTI")


def _convert_with_dcm2niix(src_dir: str, dst: str, dcm2niix_bin: str):
    """Esegue dcm2niix su una cartella DICOM e porta il risultato in `dst`.

    Se la cartella contiene piu' serie, dcm2niix scrive piu' file: si tiene
    quello con piu' voxel (il volume principale) con il suo JSON.
    """
    # 1. dcm2niix scrive in una cartella temporanea, con nome fisso "conv".
    dst = str(dst)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [dcm2niix_bin,
               "-z", "y",            # uscita compressa
               "-b", "y",            # file JSON con i metadati (formato BIDS)
               "-o", tmp,
               "-f", "conv",
               src_dir]
        res = subprocess.run(cmd, capture_output=True, text=True)
        # 2. Un codice d'uscita non nullo e' fatale solo se non e' stato
        #    prodotto alcun file: se un NIfTI c'e', lo si usa comunque.
        #    Dello stderr si riportano solo le ultime righe.
        produced = sorted(Path(tmp).glob("*.nii*"))
        if res.returncode != 0 and not produced:
            raise RuntimeError(
                f"dcm2niix non riuscito (codice {res.returncode}):\n"
                f"{res.stderr[-1500:]}")
        if not produced:
            raise RuntimeError("dcm2niix non ha prodotto alcun NIfTI.")

        # 3. Fra i NIfTI prodotti si sceglie quello con piu' voxel.
        def _nvox(p):
            """Numero di voxel del file `p` (0 se illeggibile)."""
            try:
                return int(np.prod(sitk.ReadImage(str(p)).GetSize()))
            except Exception:
                return 0
        best = max(produced, key=_nvox)

        img = _orient_ras_and_write(sitk.ReadImage(str(best)), dst)

        # 4. Il JSON accompagna il NIfTI con lo stesso nome; in mancanza si
        #    prende il primo JSON prodotto. Viene copiato accanto a `dst`,
        #    con estensione .json al posto di .nii.gz.
        sidecar = None
        json_match = best.with_suffix("").with_suffix(".json")
        if not json_match.exists():
            jsons = list(Path(tmp).glob("*.json"))
            json_match = jsons[0] if jsons else None
        if json_match and Path(json_match).exists():
            sidecar = os.path.splitext(dst)[0]
            sidecar = (sidecar[:-4] if sidecar.endswith(".nii") else sidecar) + ".json"
            shutil.copyfile(json_match, sidecar)

    return _report(img, dst, "dcm2niix", sidecar)


def _convert_with_sitk(src_dir: str, dst: str):
    """Ripiego: il lettore del modulo 01, poi scrittura in NIfTI.

    Usare lo stesso lettore garantisce che il NIfTI su disco coincida con il
    volume che la pipeline ricostruirebbe in memoria dal DICOM.
    """
    _load = importlib.import_module("01_LoadImages")
    vol = _load.ImageLoader.load_dicom_series(src_dir)
    arr = vol.pixel_data
    if arr.ndim == 4:                      # dinamica: primo frame
        arr = arr[0]
    img = sitk.GetImageFromArray(arr)
    sp, og, dr = list(vol.spacing), list(vol.origin), list(vol.direction)
    img.SetSpacing(tuple(sp[:3]))
    img.SetOrigin(tuple(og[:3]))
    # Direzione 4x4 (immagine 4D): se ne tiene il blocco 3x3 spaziale.
    if len(dr) == 16:
        dr = [dr[i] for i in (0, 1, 2, 4, 5, 6, 8, 9, 10)]
    if len(dr) == 9:
        img.SetDirection(tuple(dr))
    img = _orient_ras_and_write(sitk.Cast(img, sitk.sitkFloat32), dst)
    return _report(img, dst, "simpleitk-fallback", None)


# ────────────────────────────────────────────────────────────────────────────
# Interfaccia pubblica
# ────────────────────────────────────────────────────────────────────────────
class DicomToNiftiConverter:
    """Converte un DICOM (o fa passare un NIfTI) in un .nii.gz uniforme."""

    def __init__(self, dcm2niix_path: str | None = None,
                 prefer_dcm2niix: bool = True):
        """
        dcm2niix_path     percorso esplicito dell'eseguibile (facoltativo)
        prefer_dcm2niix   se False si usa sempre SimpleITK
        """
        self.prefer_dcm2niix = prefer_dcm2niix
        self.dcm2niix_bin = find_dcm2niix(dcm2niix_path) if prefer_dcm2niix else None
        if prefer_dcm2niix and not self.dcm2niix_bin:
            logger.warning("dcm2niix non trovato: uso SimpleITK. "
                           "(Per installarlo: pip install dcm2niix)")

    def convert(self, src: str, dst: str) -> dict:
        """Converte `src` in `dst` (.nii.gz) e restituisce il resoconto.

        Ordine dei tentativi: NIfTI gia' pronto -> dcm2niix -> SimpleITK. Un
        errore di dcm2niix non interrompe la conversione: si passa al ripiego.
        """
        if _is_nifti(src):
            return _passthrough_nifti(src, str(dst))
        # cartella DICOM, oppure un singolo .dcm della serie
        src_dir = src if os.path.isdir(src) else os.path.dirname(src)
        if self.dcm2niix_bin:
            try:
                return _convert_with_dcm2niix(src_dir, str(dst), self.dcm2niix_bin)
            except Exception as e:
                logger.warning("dcm2niix non riuscito (%s): uso SimpleITK.", e)
        return _convert_with_sitk(src_dir, str(dst))


def convert_to_nifti(src: str, dst: str, dcm2niix_path: str | None = None,
                     prefer_dcm2niix: bool = True) -> dict:
    """Scorciatoia per DicomToNiftiConverter(...).convert(src, dst)."""
    return DicomToNiftiConverter(dcm2niix_path, prefer_dcm2niix).convert(src, dst)
