"""FSL FAST: modello a mistura di gaussiane con campo di Markov (Zhang et al. 2001).

FAST classifica i voxel in tre tessuti stimando, per ciascuno, una gaussiana
dell'intensita' e regolarizzando la classificazione con un campo di Markov che
favorisce etichette uguali fra voxel vicini; corregge internamente anche la
disomogeneita' di campo.

Configurazione usata nella tesi:
  * tre classi (`-n 3`) su immagine T1 (`-t 1`);
  * niente estrazione del cervello con BET: FAST riceve la MRI gia'
    mascherata con la maschera comune ai metodi classici (template ICBM152
    portato nello spazio del soggetto);
  * come uscita si prende `_pveseg`, la classificazione che tiene conto del
    volume parziale, e solo se manca o e' degenere `_seg`.

Requisiti: FSL installato con l'eseguibile `fast` nel PATH (in Windows dentro
WSL2), oppure la variabile FSLDIR impostata.
"""

from __future__ import annotations
import os
import glob
import shutil
import subprocess
import tempfile
import logging
import numpy as np
import SimpleITK as sitk

from ..base import BaseSegmenter, register_segmenter

logger = logging.getLogger(__name__)


def _resolve_fsl_binary(name: str, explicit: str | None = None) -> str | None:
    """Trova un eseguibile FSL nel PATH o in $FSLDIR/share/fsl/bin."""
    if explicit and os.path.exists(explicit):
        return explicit
    found = shutil.which(name)
    if found:
        return found
    fsldir = os.environ.get("FSLDIR")
    if fsldir:
        cand = os.path.join(fsldir, "share", "fsl", "bin", name)
        if os.path.exists(cand):
            return cand
    return None


def _label_diversity(arr: np.ndarray) -> int:
    """Numero di etichette non nulle distinte (per riconoscere un'uscita degenere)."""
    return int(np.unique(arr[arr > 0]).size)


@register_segmenter("fsl_fast")
class FslFastSegmenter(BaseSegmenter):
    """FSL FAST a tre classi: 1 = liquor, 2 = sostanza grigia, 3 = sostanza bianca."""

    label_scheme = "tissue3"
    requires_external_tool = "fsl"

    def _segment(self, mri, external_brain_mask=None,
                 number_classes: int = 3,
                 fast_binary: str | None = None,
                 prefer_pveseg: bool = True,
                 work_dir: str | None = None,
                 keep_intermediate: bool = False, **_):
        """Lancia `fast` sulla MRI mascherata e ne rilegge la classificazione.

        mri                   MRI T1 (sitk.Image)
        external_brain_mask   maschera comune dei metodi classici; se None,
                              ripiego di `_brain_mask`
        number_classes        classi di tessuto chieste a FAST (`-n`)
        fast_binary           percorso esplicito dell'eseguibile `fast`
        prefer_pveseg         preferisce `_pveseg` a `_seg` come uscita
        work_dir              cartella di lavoro; se None, temporanea
        keep_intermediate     non cancella la cartella temporanea

        Restituisce (mappa di etichette, extras con l'uscita scelta).
        """
        # 1. Eseguibile e cartella di lavoro.
        fast =_resolve_fsl_binary("fast", fast_binary)
        if not fast:
            raise RuntimeError(
                "FSL FAST non trovato nel PATH. Installare FSL (in Windows "
                "dentro WSL2) oppure passare `fast_binary=...`.")

        if work_dir is not None:
            os.makedirs(work_dir, exist_ok=True)
            tmp_ctx, tmp = None, work_dir
        else:
            tmp_ctx = tempfile.TemporaryDirectory()
            tmp = tmp_ctx.name

        try:
            # 2. MRI mascherata: fuori dal cervello l'intensita' e' azzerata
            mask_arr = self._brain_mask(mri, use_external_mask=external_brain_mask)
            mri_arr = sitk.GetArrayFromImage(mri).astype(np.float32)
            mri_arr[mask_arr <= 0] = 0
            masked = sitk.GetImageFromArray(mri_arr)
            masked.CopyInformation(mri)
            fast_input = os.path.join(tmp, "in_brain.nii.gz")
            sitk.WriteImage(masked, fast_input)

            # 3. Esecuzione di FAST: `-t 1` immagine T1, `-n` numero di classi,
            #    `-o` prefisso dei file di uscita.
            cmd = [fast, "-t", "1", "-n", str(number_classes),
                   "-o", os.path.join(tmp, "out"), fast_input]
            logger.info("FAST: %s", " ".join(cmd))
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"FAST non riuscito:\nstderr={res.stderr}")

            produced = sorted(glob.glob(os.path.join(tmp, "out*")))
            logger.info("FAST ha prodotto %d file.", len(produced))

            # 4. Scelta dell'uscita. Si prende la prima utilizzabile (almeno
            #    due etichette) nell'ordine di preferenza; se nessuna lo e',
            #    quella con piu' etichette.
            primary = "out_pveseg.nii.gz" if prefer_pveseg else "out_seg.nii.gz"
            fallback = "out_seg.nii.gz" if prefer_pveseg else "out_pveseg.nii.gz"
            candidates = [primary, fallback]
            chosen_arr, chosen_path, chosen_div = None, None, 0
            for c in candidates:
                p = os.path.join(tmp, c)
                if not os.path.exists(p):
                    continue
                a = sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(np.int16)
                d = _label_diversity(a)
                if d >= 2:
                    chosen_arr, chosen_path, chosen_div = a, c, d
                    break
            if chosen_arr is None:
                for c in candidates:
                    p = os.path.join(tmp, c)
                    if not os.path.exists(p):
                        continue
                    a = sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(np.int16)
                    d = _label_diversity(a)
                    if d > chosen_div:
                        chosen_arr, chosen_path, chosen_div = a, c, d

            if chosen_arr is None:
                raise RuntimeError(
                    f"FAST non ha prodotto una segmentazione utilizzabile in "
                    f"{tmp}. File: {produced}")
            if chosen_div < 2:
                logger.warning("La segmentazione di FAST e' degenere (%d "
                               "etichetta): le VOI saranno probabilmente vuote.",
                               chosen_div)

            # 5. Le etichette di FAST (1, 2, 3 per T1) seguono gia' lo schema
            #    'tissue3'; si riporta la geometria della MRI originale.
            logger.info("Uso %s (%d etichette)", chosen_path, chosen_div)
            out = self._np_to_sitk_like(chosen_arr, mri)
            info = {"backend": "fsl_fast",
                    "n_classes": number_classes,
                    "chosen_output": chosen_path,
                    "label_diversity": chosen_div,
                    "external_brain_mask_used": external_brain_mask is not None,
                    "prefer_pveseg": bool(prefer_pveseg),
                    "work_dir": tmp if (keep_intermediate or work_dir) else None}
            return out, info
        finally:
            # la cartella temporanea si cancella anche in caso di errore
            if tmp_ctx is not None and not keep_intermediate:
                tmp_ctx.cleanup()
