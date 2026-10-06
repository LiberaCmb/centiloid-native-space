"""SAMSEG (Puonti et al. 2016), segmentazione bayesiana di FreeSurfer (`run_samseg`).

Modello generativo: un atlante probabilistico deformabile fa da distribuzione a
priori delle etichette e una mistura di gaussiane per ogni struttura modella
le intensita'; la disomogeneita' di campo e' stimata insieme al resto. Non
richiede addestramento ed e' indipendente dal contrasto.

Nella configurazione predefinita SAMSEG restituisce le strutture dell'atlante
`aseg`, con la corteccia di ciascun emisfero come un'unica etichetta (3 e 42):
manca la parcellazione in lobi del target Centiloid. Per questo la pipeline la
usa in due modi (vedi modulo 05):
  samseg         target = corteccia intera dei due emisferi;
  samseg_atlas   target = VOI Centiloid standard ∩ corteccia di SAMSEG.

Requisiti: FreeSurfer con `run_samseg` nel PATH, oppure il pacchetto `samseg`
installato a parte e il percorso passato con
`--seg-kwargs '{"samseg_binary": "/percorso/run_samseg"}'`.
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


@register_segmenter("samseg")
class SamsegSegmenter(BaseSegmenter):
    """SAMSEG: etichette FreeSurfer aseg, corteccia non parcellizzata (3 e 42)."""

    label_scheme = "freesurfer"
    requires_gpu = False
    requires_external_tool = "freesurfer (run_samseg)"

    def _segment(self, mri, samseg_binary: str | None = None,
                 threads: int = 8, **_):
        """Lancia `run_samseg` e riporta `seg.mgz` sulla griglia della MRI.

        mri             MRI T1 (sitk.Image)
        samseg_binary   percorso o nome dell'eseguibile; se None, `run_samseg`
                        dal PATH
        threads         thread di calcolo (`--threads`)

        Restituisce (mappa di etichette, extras).
        """
        # L'eseguibile puo' essere un percorso esistente o un nome nel PATH.
        run_cmd = samseg_binary or shutil.which("run_samseg")
        if not run_cmd or (not os.path.exists(run_cmd) and not shutil.which(run_cmd)):
            raise RuntimeError(
                "`run_samseg` non trovato. Caricare FreeSurfer, oppure "
                "installare samseg a parte e passare il percorso con "
                "--seg-kwargs '{\"samseg_binary\": \"/percorso/run_samseg\"}'.")

        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "in.nii.gz")
            out_dir = os.path.join(tmp, "samseg_out")
            os.makedirs(out_dir, exist_ok=True)
            sitk.WriteImage(mri, in_path)

            cmd = [run_cmd, "-i", in_path, "-o", out_dir, "--threads", str(threads)]
            logger.info("run_samseg: %s", " ".join(cmd))
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                # stderr troncato per non riempire il registro
                raise RuntimeError(f"run_samseg non riuscito:\n{res.stderr[:2000]}")

            seg_path = os.path.join(out_dir, "seg.mgz")
            # SimpleITK non sempre legge il formato .mgz di FreeSurfer: in quel
            # caso si passa da nibabel. SAMSEG lavora sulla griglia della MRI
            # d'ingresso, quindi se ne puo' copiare la geometria.
            try:
                seg = sitk.ReadImage(seg_path, sitk.sitkInt16)
            except RuntimeError:
                import nibabel as nib
                arr = nib.load(seg_path).get_fdata().astype("int16")
                seg = sitk.GetImageFromArray(arr.transpose(2, 1, 0))  # (x,y,z) -> (z,y,x)
                seg = sitk.Cast(seg, sitk.sitkInt16)
                seg.SetSpacing(mri.GetSpacing())
                seg.SetOrigin(mri.GetOrigin())
                seg.SetDirection(mri.GetDirection())

        return resample_labels_like(seg, mri), {"backend": "freesurfer:samseg"}
