"""Ricampionamento e salvataggio della coppia PET/MRI allineata.

La PET viene portata sulla griglia della MRI con la trasformazione del
modulo 02; la MRI resta com'e'. Dopo questo modulo esistono:
    Realigned_Anatomical/Realigned_Anatomical.nii.gz   (MRI, invariata)
    Realigned_Functional/Realigned_Functional.nii.gz   (PET sulla griglia MRI)

Modulo 03: riceve il contesto con le immagini del modulo 01 e la
trasformazione del modulo 02; i due file scritti sono l'ingresso dei moduli
04-06.
"""

from __future__ import annotations
import os
import SimpleITK as sitk


class ImageFusionLogic:
    """Applicazione della trasformazione e salvataggio della coppia allineata."""

    @staticmethod
    def save_realigned_images_with_transform(context, output_dir, final_transform):
        """Scrive la MRI cosi' com'e' e la PET ricampionata sulla sua griglia.

        La MRI e' l'immagine fissa dell'allineamento: non viene mai spostata, e
        le regioni di interesse calcolate su di essa non dipendono
        dall'allineamento. Solo la lettura del segnale PET ne dipende.

        context          ProjectContext con MRI e PET caricate
        output_dir       cartella del soggetto
        final_transform  sitk.Transform del modulo 02 (PET -> MRI)

        Restituisce (percorso MRI, percorso PET) e li registra nel contesto.
        """
        if not context.anatomical_1 or not context.functional_1:
            raise Exception("Mancano le immagini da salvare.")

        anat_dir = os.path.join(output_dir, "Realigned_Anatomical")
        func_dir = os.path.join(output_dir, "Realigned_Functional")
        os.makedirs(anat_dir, exist_ok=True)
        os.makedirs(func_dir, exist_ok=True)

        def to_sitk(volume):
            """ImageVolume -> sitk.Image con la sua geometria."""
            img = sitk.GetImageFromArray(volume.pixel_data)
            img.SetSpacing(volume.spacing)
            img.SetOrigin(volume.origin)
            img.SetDirection(volume.direction)
            return img

        # MRI: riscritta senza modifiche
        anat_img = to_sitk(context.anatomical_1)
        anat_path = os.path.join(anat_dir, "Realigned_Anatomical.nii.gz")
        sitk.WriteImage(anat_img, anat_path)

        # PET: ricampionata sulla griglia della MRI (interpolazione lineare).
        # Fuori dal campo di vista della PET il valore e' 0.
        rs = sitk.ResampleImageFilter()
        rs.SetReferenceImage(anat_img)
        rs.SetTransform(final_transform)
        rs.SetInterpolator(sitk.sitkLinear)
        rs.SetDefaultPixelValue(0)
        rs.SetOutputPixelType(sitk.sitkFloat32)
        out = rs.Execute(to_sitk(context.functional_1))

        func_path = os.path.join(func_dir, "Realigned_Functional.nii.gz")
        sitk.WriteImage(out, func_path)

        context.realigned_anat_path = anat_path
        context.realigned_func_path = func_path
        return anat_path, func_path
