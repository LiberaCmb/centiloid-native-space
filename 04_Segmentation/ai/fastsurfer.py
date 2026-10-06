"""FastSurfer (Henschel et al. 2020), parcellazione DKT con reti convoluzionali.

Si esegue solo la fase di segmentazione (`--seg_only`), senza la ricostruzione
delle superfici corticali, che allungherebbe di ore l'elaborazione senza
servire a nulla di quanto si misura qui.

La rete principale non etichetta il cervelletto: lo fa CerebNet, attivato con
`--cereb_segfile`. Le sue etichette (anche quelle fini dei lobuli) vengono
riportate alla convenzione FreeSurfer (8 = corteccia cerebellare sinistra,
47 = destra), cosi' la regione di riferimento e' la stessa per tutti i metodi.

Requisiti: il repository FastSurfer (variabile FASTSURFER_HOME) e un ambiente
conda dedicato (predefinito "fastsurfer"), perche' le sue dipendenze non sono
compatibili con quelle della pipeline. Opzioni passabili con --seg-kwargs:
    fastsurfer_home, conda_env, device ("cuda" | "cpu"), seg_only.
"""

from __future__ import annotations
import glob
import os
import shutil
import subprocess
import tempfile
import logging
import numpy as np
import SimpleITK as sitk

from ..base import BaseSegmenter, register_segmenter, resample_labels_like

logger = logging.getLogger(__name__)

# Nomi possibili dell'uscita, in ordine di preferenza (cambiano fra versioni).
_NOMI_USCITA = ("aparc.DKTatlas+aseg.deep.mgz",
                "aparc+aseg.deep.mgz",
                "aparc.DKTatlas+aseg.mgz",
                "aparc.DKTatlas+aseg.deep.withCC.mgz",
                "aparc.mapped+aseg.mgz")


@register_segmenter("fastsurfer")
class FastSurferSegmenter(BaseSegmenter):
    """FastSurfer (DKT) con il cervelletto di CerebNet: etichette FreeSurfer."""

    label_scheme = "freesurfer"
    requires_gpu = True
    requires_external_tool = "fastsurfer (ambiente conda dedicato)"

    def _segment(self, mri, fastsurfer_home: str | None = None,
                 conda_env: str = "fastsurfer", device: str = "cuda",
                 seg_only: bool = True, **_):
        """Lancia `run_fastsurfer.sh` e restituisce aparc+aseg con il cervelletto.

        mri               MRI T1 (sitk.Image)
        fastsurfer_home   cartella del repository; se None, FASTSURFER_HOME
        conda_env         ambiente conda in cui eseguire FastSurfer (se conda
                          non c'e' si lancia lo script direttamente)
        device            "cuda" oppure "cpu"
        seg_only          solo segmentazione, senza superfici

        Restituisce (mappa di etichette, extras).
        """
        # 1. Verifica dell'installazione.
        fs_home = fastsurfer_home or os.environ.get("FASTSURFER_HOME")
        if not fs_home or not os.path.isdir(fs_home):
            raise RuntimeError(
                f"FASTSURFER_HOME non impostata o non valida: {fs_home!r}. "
                "Clonare https://github.com/Deep-MI/FastSurfer.git ed "
                "esportare la variabile.")
        run_sh = os.path.join(fs_home, "run_fastsurfer.sh")
        if not os.path.exists(run_sh):
            raise RuntimeError(f"Manca {run_sh}")

        # 2. Esecuzione in una cartella temporanea, con la struttura di
        #    FreeSurfer <sd>/<sid>/mri. Il nome del soggetto e' fittizio.
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "in.nii.gz")
            sid = "subject1"
            sd = os.path.join(tmp, "fs_out")
            os.makedirs(sd, exist_ok=True)
            sitk.WriteImage(mri, in_path)

            mri_dir = os.path.join(sd, sid, "mri")
            cereb_segfile = os.path.join(mri_dir, "cerebellum.CerebNet.nii.gz")
            cmd = ["bash", run_sh, "--t1", in_path, "--sd", sd, "--sid", sid,
                   "--device", device, "--cereb_segfile", cereb_segfile]
            if seg_only:
                cmd += ["--seg_only"]
            # `conda run` esegue FastSurfer nel suo ambiente senza attivarlo
            # nella shell corrente.
            conda_exe = shutil.which("conda")
            if conda_exe and conda_env:
                cmd = [conda_exe, "run", "-n", conda_env, "--no-capture-output"] + cmd

            logger.info("FastSurfer: %s", " ".join(cmd))
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                # solo la coda dello stderr: il registro completo e' molto lungo
                tail = res.stderr[-2000:] if res.stderr else ""
                raise RuntimeError(f"FastSurfer non riuscito (codice "
                                   f"{res.returncode}):\n{tail}")

            # 3. Ricerca dell'uscita: prima i nomi noti, poi qualunque file
            #    *aparc*aseg*.mgz.
            seg_path = next((os.path.join(mri_dir, n) for n in _NOMI_USCITA
                             if os.path.exists(os.path.join(mri_dir, n))), None)
            if seg_path is None:
                fuzzy = glob.glob(os.path.join(mri_dir, "*aparc*aseg*.mgz"))
                if not fuzzy:
                    disponibili = (sorted(os.listdir(mri_dir))
                                   if os.path.isdir(mri_dir) else [])
                    raise RuntimeError(f"Uscita di FastSurfer non trovata in "
                                       f"{mri_dir}. File presenti: {disponibili}")
                seg_path = fuzzy[0]
            logger.info("Uscita di FastSurfer: %s", os.path.basename(seg_path))

            # 4. Lettura. L'uscita di FastSurfer vive nello spazio "conformato" di
            # FreeSurfer (256^3, 1 mm), non sulla griglia della MRI: va letta
            # con la sua matrice affine vera (nibabel -> NIfTI) e solo dopo
            # ricampionata. Copiare la geometria della MRI sarebbe sbagliato.
            try:
                seg = sitk.ReadImage(seg_path, sitk.sitkInt16)
            except RuntimeError:
                import nibabel as nib
                nb_img = nib.load(seg_path)
                nii_path = seg_path.replace(".mgz", ".nii.gz")
                if nii_path == seg_path:
                    nii_path = seg_path + ".nii.gz"
                nib.save(nib.Nifti1Image(nb_img.get_fdata().astype("int16"),
                                         nb_img.affine, nb_img.header), nii_path)
                seg = sitk.ReadImage(nii_path, sitk.sitkInt16)

            # 5. Unione con il cervelletto di CerebNet (NIfTI o, in alcune
            #    versioni, .mgz).
            cereb_file = next((p for p in (cereb_segfile,
                                           os.path.join(mri_dir, "cerebellum.CerebNet.mgz"))
                               if os.path.exists(p)), None)
            if cereb_file is not None:
                seg = _unisci_cerebnet(seg, cereb_file)
            else:
                logger.warning("Uscita di CerebNet non trovata: mancheranno "
                               "le etichette del cervelletto.")

        # 6. Dallo spazio conformato alla griglia della MRI.
        return resample_labels_like(seg, mri), {
            "backend": "native+conda_env", "conda_env": conda_env,
            "seg_only": seg_only, "cereb_merged": cereb_file is not None}


def _unisci_cerebnet(seg: sitk.Image, cereb_file: str) -> sitk.Image:
    """Sovrascrive con le etichette di CerebNet i voxel che esso assegna al
    cervelletto, riportandole a 8 (sinistra) e 47 (destra).

    CerebNet puo' emettere sia le etichette standard sia quelle dei lobuli
    (601-630): per il Centiloide basta il cervelletto intero, quindi ogni voxel
    non nullo diventa corteccia cerebellare, sinistra o destra a seconda della
    posizione rispetto al piano mediano. Si sovrascrive anche dove la rete
    principale aveva gia' un'etichetta (di solito sostanza bianca cerebrale
    "sbavata" nel cervelletto), perche' CerebNet e' piu' affidabile li'.
    """
    try:
        import nibabel as nib
        cereb_arr = nib.load(cereb_file).get_fdata().astype("int16")
        seg_arr = sitk.GetArrayFromImage(seg)
        # nibabel legge in ordine (x, y, z), SimpleITK in (z, y, x)
        if cereb_arr.shape == seg_arr.shape[::-1]:
            cereb_arr = cereb_arr.transpose(2, 1, 0)
        if cereb_arr.shape != seg_arr.shape:
            logger.warning("CerebNet %s e aparc %s hanno forme diverse: "
                           "nessuna unione.", cereb_arr.shape, seg_arr.shape)
            return seg
        cereb_nz = cereb_arr != 0
        nz_idx = np.argwhere(cereb_nz)
        if nz_idx.size == 0:
            logger.warning("Uscita di CerebNet vuota: nessuna unione.")
            return seg
        # Piano mediano stimato come mediana dell'indice x (ultimo asse
        # dell'array) dei voxel del cervelletto, che e' circa simmetrico; i
        # voxel con indice minore ricevono l'etichetta 8. Per il riferimento
        # conta solo l'unione 8 + 47, non il lato.
        x_mid = float(np.median(nz_idx[:, 2]))
        left = cereb_nz & (np.arange(cereb_arr.shape[2])[None, None, :] < x_mid)
        mapped = np.zeros_like(cereb_arr)
        mapped[left] = 8                     # Left-Cerebellum-Cortex
        mapped[cereb_nz & ~left] = 47        # Right-Cerebellum-Cortex
        sovrascritti = int((seg_arr != 0)[cereb_nz].sum())
        seg_arr[cereb_nz] = mapped[cereb_nz]
        logger.info("CerebNet: %d voxel assegnati a 8/47 (%d sovrascritti).",
                    int(cereb_nz.sum()), sovrascritti)
        merged = sitk.GetImageFromArray(seg_arr.astype("int16"))
        merged.CopyInformation(seg)
        return merged
    except Exception as e:
        logger.warning("Unione con CerebNet non riuscita (%s): mancheranno "
                       "le etichette del cervelletto.", e)
        return seg
