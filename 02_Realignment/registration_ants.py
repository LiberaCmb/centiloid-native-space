"""Allineamento rigido PET (mobile) -> MRI (fissa) con ANTs (ANTsPy).

E' il motore usato per tutti i risultati della tesi. Stessa interfaccia della
versione SimpleITK:

    run_registration_ants(context)  ->  (sitk.Transform, messaggio)

"Rigid" vuol dire sei gradi di liberta' (tre rotazioni e tre traslazioni),
corretto per due immagini dello stesso soggetto; la metrica e' l'informazione
mutua di Mattes ("mattes"), adatta a due modalita' con scale di intensita'
non confrontabili.

RIPETIBILITA'
-------------
Con le impostazioni predefinite ANTs NON e' deterministico. Misura sul soggetto
2002 (FBB), cinque allineamenti sugli stessi NIfTI:

    configurazione                    dev.std SUVR  escurs. CL   tempo
    predefinita (multithread)           0.002496       1.42       4.9 s
    ITK a un solo thread                0.001097       0.40      ~27 s
    seme fissato, multithread           0.000021       0.008      4.9 s
    seme fissato, un solo thread        0.000000       0.000     26.9 s

La fonte dominante e' il campionamento casuale della metrica: fissando il seme
la dispersione scende sotto il millesimo di CL senza perdere velocita'. Il
multi-restart della versione SimpleITK qui non servirebbe, perche' le soluzioni
hanno tutte la stessa informazione mutua (fra -0.5348 e -0.5331) e la metrica
non saprebbe sceglierne una. Misura ripetibile con
`python -m scripts.analisi.diag_ants_determinismo`.

La modalita' con il seme fissato e' spenta di default, perche' accenderla
cambierebbe di qualche millesimo tutti i risultati gia' calcolati. Si attiva
senza toccare il codice:

    export CENTILOID_ANTS_DETERMINISTIC=1
    export CENTILOID_ANTS_SEED=123          # facoltativo
    export CENTILOID_ANTS_SINGLE_THREAD=1   # solo se serve l'identita' al bit

oppure per una singola chiamata con `run_registration_ants(ctx, deterministic=True)`.

COME LA TRASFORMAZIONE ARRIVA AL RESTO DELLA PIPELINE
-----------------------------------------------------
ANTs scrive la trasformazione rigida come file ITK `.mat`, che SimpleITK legge
con `sitk.ReadTransform()`. Entrambi lavorano in coordinate LPS e con la stessa
convenzione di ricampionamento, quindi la trasformazione si passa cosi' com'e'
al modulo 03.

`ants` viene importato dentro la funzione: importare il modulo 02 non richiede
ANTsPy, serve solo quando si usa questo motore.
"""

from __future__ import annotations
import logging
import os
import tempfile

import SimpleITK as sitk

from shared.context import ProjectContext
# Stessa costruzione delle immagini del motore SimpleITK: MRI e PET vivono
# nello stesso spazio fisico nei due motori, che restano confrontabili.
from .registration import _to_sitk_3d

logger = logging.getLogger(__name__)

# Trasformazione e metrica passate ad ants.registration (vedi sopra).
TYPE_OF_TRANSFORM = "Rigid"
AFF_METRIC        = "mattes"

# Ripetibilita' (vedi sopra). Predefinito: spenta.
DETERMINISTIC_DEFAULT = os.environ.get("CENTILOID_ANTS_DETERMINISTIC", "0") == "1"
RANDOM_SEED_DEFAULT   = int(os.environ.get("CENTILOID_ANTS_SEED", "123"))
SINGLE_THREAD_DEFAULT = os.environ.get("CENTILOID_ANTS_SINGLE_THREAD", "0") == "1"


def _apply_deterministic(seed: int, single_thread: bool = True) -> bool:
    """Attiva il seme fisso di ANTsPy. Restituisce True se ha funzionato."""
    try:
        import ants
        ants.config.set_ants_deterministic(True, seed)
        if not single_thread:
            # set_ants_deterministic impone anche un solo thread ITK:
            # lo si toglie per tenere il multithreading.
            os.environ.pop("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", None)
        return True
    except Exception as e:      # ANTsPy troppo vecchio
        logger.warning("Modalita' deterministica non disponibile in questa "
                       "versione di ANTsPy: %s", e)
        return False


def _sitk_to_ants(img_sitk, tmpdir, name):
    """Passa un'immagine da SimpleITK ad ANTs attraverso un NIfTI temporaneo,
    cosi' ANTs legge esattamente la geometria scritta da SimpleITK."""
    import ants
    path = os.path.join(tmpdir, f"{name}.nii.gz")
    sitk.WriteImage(img_sitk, path)
    return ants.image_read(path)


def run_registration_ants(context: ProjectContext, progress_callback=None,
                          type_of_transform: str = TYPE_OF_TRANSFORM,
                          deterministic: bool | None = None,
                          random_seed: int | None = None,
                          single_thread: bool | None = None):
    """Allineamento rigido PET->MRI con ANTs.

    Restituisce (Transform, messaggio): la trasformazione e' un
    `sitk.Transform` letto dal `.mat` di ANTs, il messaggio riporta
    l'informazione mutua finale e i parametri rigidi.

    Se le immagini sono co-registrate restituisce l'identita'.

    context            ProjectContext con MRI (anatomical_1) e PET (functional_1)
    progress_callback  chiamata con 1 all'inizio e 100 alla fine
    type_of_transform  trasformazione di ANTs (predefinita "Rigid")
    deterministic      seme fisso; None = CENTILOID_ANTS_DETERMINISTIC
    random_seed        seme; None = CENTILOID_ANTS_SEED
    single_thread      un solo thread ITK; None = CENTILOID_ANTS_SINGLE_THREAD
    """
    # 1. Impostazioni di ripetibilita': gli argomenti espliciti prevalgono
    #    sulle variabili d'ambiente. Vanno applicate prima di usare ANTs.
    if deterministic is None:
        deterministic = DETERMINISTIC_DEFAULT
    seed = RANDOM_SEED_DEFAULT if random_seed is None else int(random_seed)
    one_thread = SINGLE_THREAD_DEFAULT if single_thread is None else single_thread
    det_attivo = (_apply_deterministic(seed, one_thread)
                  if deterministic else False)
    try:
        import ants
    except ImportError as e:
        return None, ("ANTsPy non disponibile (pip install antspyx). "
                      f"Dettaglio: {e}")

    # 2. Immagini in SimpleITK, come nel motore SimpleITK.
    if not context.anatomical_1:
        return None, "Nessuna MRI caricata."
    if not context.functional_1:
        return None, "Nessuna PET caricata."

    fixed = _to_sitk_3d(context.anatomical_1.pixel_data,
                        context.anatomical_1.spacing,
                        context.anatomical_1.origin,
                        context.anatomical_1.direction)
    moving = _to_sitk_3d(context.functional_1.pixel_data,
                         context.functional_1.spacing,
                         context.functional_1.origin,
                         context.functional_1.direction)

    if context.is_co_acquired:
        return sitk.Transform(3, sitk.sitkIdentity), \
               "Immagini co-registrate: applicata l'identita' (ANTs)."

    if fixed.GetDimension() != 3 or moving.GetDimension() != 3:
        return None, (f"Dimensioni incompatibili: fissa {fixed.GetDimension()}D, "
                      f"mobile {moving.GetDimension()}D")

    if progress_callback:
        progress_callback(1)

    # 3. Allineamento in ANTs; i file intermedi vivono in una cartella
    #    temporanea, quindi il .mat va letto prima di uscirne.
    with tempfile.TemporaryDirectory() as tmp:
        ants_fixed  = _sitk_to_ants(fixed,  tmp, "fixed_mr")
        ants_moving = _sitk_to_ants(moving, tmp, "moving_pet")

        # fissa = MRI, mobile = PET: fwdtransforms porta la PET sulla MRI
        try:
            reg = ants.registration(
                fixed=ants_fixed,
                moving=ants_moving,
                type_of_transform=type_of_transform,
                aff_metric=AFF_METRIC,
            )
        except Exception as e:
            logger.error("Allineamento ANTs non riuscito: %s", e)
            return None, f"Allineamento ANTs non riuscito: {e}"

        fwd = reg.get("fwdtransforms", [])
        if not fwd:
            return None, "ANTs non ha restituito alcuna trasformazione."

        # Con "Rigid" la lista contiene un solo .mat; per sicurezza si cerca
        # il primo file .mat, altrimenti si prende il primo elemento.
        mat_path = next((p for p in fwd if p.endswith(".mat")), fwd[0])
        if not os.path.exists(mat_path):
            return None, f"File della trasformazione non trovato: {mat_path}"

        try:
            sitk_tx = sitk.ReadTransform(mat_path)
        except Exception as e:
            return None, f"Impossibile leggere il .mat di ANTs con SimpleITK: {e}"

        # Controllo: informazione mutua fra MRI e PET allineata
        # (piu' negativa = allineamento migliore, come in SimpleITK).
        try:
            warped = ants.apply_transforms(
                fixed=ants_fixed, moving=ants_moving, transformlist=fwd)
            mi = float(ants.image_mutual_information(ants_fixed, warped))
            mi_str = f"{mi:.4f}"
        except Exception as e:                       # solo informativo
            logger.warning("Informazione mutua di controllo non calcolata: %s", e)
            mi_str = "n/a"

    if progress_callback:
        progress_callback(100)

    # 4. Messaggio: informazione mutua, parametri rigidi e modalita'.
    try:
        params_str = ", ".join(f"{p:.4f}" for p in sitk_tx.GetParameters())
    except Exception:
        params_str = "n/a"

    modo = (f"deterministico, seme={seed}"
            f"{'' if one_thread else ', multithread'}" if det_attivo
            else "esecuzione singola, non deterministica")
    # Il formato "MI(MR, warped-PET)=" e' letto da diag_ants_determinismo.py.
    msg = (f"ANTs {type_of_transform} ({AFF_METRIC} MI): MI(MR, warped-PET)="
           f"{mi_str}  params=[{params_str}]  ({modo})")
    return sitk_tx, msg
