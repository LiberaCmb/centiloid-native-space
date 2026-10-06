"""SynthSeg (Billot et al. 2023), eseguito tramite FreeSurfer (`mri_synthseg`).

Rete convoluzionale addestrata su immagini sintetiche di contrasto e
risoluzione casuali, quindi utilizzabile su qualunque MRI senza
riaddestramento. Si lancia con la parcellazione corticale attiva (`--parc`):
senza, restituirebbe solo l'atlante sottocorticale e la corteccia resterebbe
indivisa, come in SAMSEG, senza un target Centiloid. `--robust` attiva la
variante pensata per acquisizioni cliniche eterogenee.

Requisiti: FreeSurfer >= 7.3 con `mri_synthseg` nel PATH (in Windows dentro
WSL2). Con la GPU impiega circa un minuto a soggetto.
"""

from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
import logging
import SimpleITK as sitk

from ..base import BaseSegmenter, register_segmenter, resample_labels_like

logger = logging.getLogger(__name__)


@register_segmenter("synthseg")
class SynthSegSegmenter(BaseSegmenter):
    """SynthSeg con parcellazione corticale: etichette FreeSurfer aparc+aseg."""

    label_scheme = "freesurfer"
    requires_gpu = True          # funziona anche su CPU, piu' lentamente
    requires_external_tool = "freesurfer (mri_synthseg)"

    def _segment(self, mri, robust: bool = True, parc: bool = True,
                 fast: bool = False, device: str = "cuda", **_):
        """Lancia `mri_synthseg` e riporta le etichette sulla griglia della MRI.

        mri       MRI T1 (sitk.Image)
        robust    variante `--robust` per acquisizioni cliniche
        parc      parcellazione corticale (`--parc`), necessaria per il target
        fast      variante `--fast` (meno post-elaborazione)
        device    "cuda" oppure "cpu" (`--cpu`)

        Restituisce (mappa di etichette, extras).
        """
        if not shutil.which("mri_synthseg"):
            raise RuntimeError(
                "SynthSeg non disponibile: serve FreeSurfer >= 7.3 con "
                "`mri_synthseg` nel PATH (caricare SetUpFreeSurfer.sh).")

        # La MRI passa a SynthSeg attraverso un NIfTI in una cartella
        # temporanea, cancellata all'uscita dal blocco.
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "in.nii.gz")
            out_path = os.path.join(tmp, "seg.nii.gz")
            sitk.WriteImage(mri, in_path)

            cmd = ["mri_synthseg", "--i", in_path, "--o", out_path]
            if robust:
                cmd += ["--robust"]
            if parc:
                cmd += ["--parc"]
            if fast:
                cmd += ["--fast"]
            if device == "cpu":
                cmd += ["--cpu"]
            logger.info("mri_synthseg: %s", " ".join(cmd))
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"mri_synthseg non riuscito:\n{res.stderr}")

            seg = sitk.ReadImage(out_path, sitk.sitkInt16)

        # SynthSeg lavora a 1 mm isotropo: si riporta sulla griglia della MRI
        return resample_labels_like(seg, mri), {"backend": "freesurfer",
                                                "parcellation": parc}
