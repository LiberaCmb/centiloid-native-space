"""Modulo 00 — Conversione DICOM -> NIfTI (prima del modulo 01).

Gli scanner producono DICOM; dal modulo 01 in poi la pipeline lavora su un
unico formato, il NIfTI compresso (.nii.gz), qualunque fosse il formato di
partenza dell'archivio (DICOM per FBB, gia' NIfTI per FMM e DLBS).

Motore di conversione
---------------------
Il motore principale e' **dcm2niix**, lo strumento di riferimento in
neuroimaging: ordina le fette, applica i fattori di scala (pendenza e
intercetta) e scrive accanto al NIfTI un file JSON con i metadati della PET.
Si installa con `pip install dcm2niix`.

Se dcm2niix non c'e', il convertitore usa il lettore DICOM di SimpleITK (lo
stesso del modulo 01), quindi la pipeline gira anche senza installazioni
aggiuntive. Gli ingressi gia' in NIfTI vengono solo riorientati in RAS e
riscritti, cosi' l'albero di uscita e' uniforme.
"""

from .logic import convert_to_nifti, DicomToNiftiConverter

__all__ = ["convert_to_nifti", "DicomToNiftiConverter"]
