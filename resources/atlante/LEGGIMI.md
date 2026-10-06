# Atlante

File usati dai metodi con atlante (Otsu, k-means, FSL FAST, samseg_atlas) per
portare le regioni standard nello spazio del soggetto. Sono copie compresse
(.nii.gz, stessi voxel e stessa geometria) dei file originali.

| file | che cos'è | origine |
|---|---|---|
| `icbm_avg_152_t1_tal_lin.nii.gz` | template T1 ICBM152 lineare, 181×217×181, 1 mm | MNI, pacchetto `mni_icbm152_lin_nifti` |
| `icbm_avg_152_t1_tal_lin_mask.nii.gz` | maschera del cervello del template | idem |
| `voi_ctx_1mm.nii.gz` | VOI corticale standard del Centiloid Project | GAAIN, `Centiloid_Std_VOI/nifti/1mm` |
| `voi_WhlCbl_1mm.nii.gz` | VOI del cervelletto intero | idem |

Le VOI GAAIN sono definite nello spazio ICBM152 *lineare*: il template deve
essere questo, non l'MNI152 di FSL (griglia 182×218×182). La licenza del
template è in `COPYING_ICBM152.txt`.

Per usare file diversi: variabile d'ambiente `CENTILOID_ATLANTE` con la cartella
che li contiene (stessi nomi).
