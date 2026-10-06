"""Allineamento rigido PET (mobile) -> MRI (fissa) con SimpleITK.

Metrica: informazione mutua di Mattes, valutata su un campione casuale del 10%
dei voxel. Il campione dipende dal seme del generatore casuale, e con un solo
tentativo il risultato cambiava molto da un seme all'altro: su un soggetto di
prova, a parita' di tutto il resto, il Centiloide variava fra 27 e 93 (valore
di riferimento 102, deviazione standard su dieci tentativi 17 CL). La metrica
finale, pero', distingue bene le soluzioni buone da quelle cattive.

Per questo l'allineamento viene ripetuto N_RESTARTS volte con semi fissati
(0, 1, ..., N-1) e si tiene la trasformazione con l'informazione mutua migliore
(il "multi-restart" del capitolo 3). I parametri di ciascun tentativo sono
rimasti quelli di partenza: aumentare iterazioni o campionamento aumentava la
varianza fra i tentativi invece di ridurla.

Il messaggio restituito riporta l'informazione mutua di tutti i tentativi:
`scripts/analisi/measure_registration_stability.py` lo legge per misurare la
stabilita' dell'allineamento.
"""

from __future__ import annotations
import logging
import multiprocessing
import numpy as np
import SimpleITK as sitk

from shared.context import ProjectContext

logger = logging.getLogger(__name__)

# ── Parametri ────────────────────────────────────────────────────────────
# Parametri di ciascun tentativo; N_RESTARTS e' il numero di tentativi.
# Piramide a tre livelli: immagini ridotte di 4, 2 e 1 volte, lisciate con
# sigma di 2, 1 e 0 mm, dal grossolano al fine.
N_RESTARTS       = 10
SAMPLING_PCT     = 0.10
N_ITERATIONS     = 100
HISTOGRAM_BINS   = 50
LEARNING_RATE    = 1.0
SHRINK_LEVELS    = [4, 2, 1]
SMOOTH_SIGMAS_MM = [2, 1, 0]


def _to_sitk_3d(arr, spacing, origin, direction):
    """Costruisce un'immagine SimpleITK 3D float a partire da array e geometria."""
    # Volume 4D (PET dinamica): si usa il primo frame, e della geometria si
    # tengono solo le tre componenti spaziali (direzione 4x4 -> 3x3).
    vol = arr[0] if arr.ndim == 4 else arr
    img = sitk.GetImageFromArray(vol)
    sp = list(spacing)
    og = list(origin)
    dr = list(direction)
    if len(sp) > 3:
        sp = sp[:3]
    if len(og) > 3:
        og = og[:3]
    if len(dr) == 16:
        dr = [dr[i] for i in [0, 1, 2, 4, 5, 6, 8, 9, 10]]
    img.SetSpacing(sp)
    img.SetOrigin(og)
    if len(dr) == 9:
        img.SetDirection(dr)
    # Immagini vettoriali (es. RGB): si tiene la prima componente.
    if img.GetNumberOfComponentsPerPixel() > 1:
        sel = sitk.VectorIndexSelectionCastImageFilter()
        sel.SetIndex(0)
        img = sel.Execute(img)
    return sitk.Cast(img, sitk.sitkFloat32)


def _build_method(initial, seed: int):
    """Un'ottimizzazione completa con il seme indicato."""
    m = sitk.ImageRegistrationMethod()
    m.SetMetricAsMattesMutualInformation(numberOfHistogramBins=HISTOGRAM_BINS)
    m.SetMetricSamplingStrategy(m.RANDOM)
    try:
        m.SetMetricSamplingPercentage(SAMPLING_PCT, seed=seed)
    except TypeError:                     # SimpleITK molto vecchi: niente seme
        m.SetMetricSamplingPercentage(SAMPLING_PCT)
    m.SetInterpolator(sitk.sitkLinear)
    # Discesa del gradiente: si ferma prima delle N_ITERATIONS se la metrica
    # varia meno di 1e-6 su una finestra di 10 iterazioni.
    m.SetOptimizerAsGradientDescent(
        learningRate=LEARNING_RATE, numberOfIterations=N_ITERATIONS,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    # Rende confrontabili i passi su rotazioni (radianti) e traslazioni (mm).
    m.SetOptimizerScalesFromPhysicalShift()
    m.SetShrinkFactorsPerLevel(SHRINK_LEVELS)
    m.SetSmoothingSigmasPerLevel(SMOOTH_SIGMAS_MM)
    m.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    m.SetInitialTransform(initial, inPlace=False)
    return m


def run_registration(context: ProjectContext, progress_callback=None,
                     n_restarts: int = N_RESTARTS):
    """Allineamento rigido PET->MRI con multi-restart.

    Esegue `n_restarts` ottimizzazioni indipendenti e restituisce quella con
    la metrica piu' bassa (Mattes la restituisce con il segno meno: piu'
    negativa = allineamento migliore).

    Restituisce (Transform, messaggio). Se le immagini sono gia' co-registrate
    (`context.is_co_acquired`) restituisce l'identita' senza ottimizzare.

    context            ProjectContext con MRI (anatomical_1) e PET (functional_1)
    progress_callback  funzione chiamata con l'iterazione corrente (0-100)
    n_restarts         numero di tentativi, con semi 0..n_restarts-1
    """
    # 1. Tutti i core disponibili per ITK, poi le immagini in SimpleITK.
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(multiprocessing.cpu_count())

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
               "Immagini co-registrate: applicata l'identita'."

    if fixed.GetDimension() != 3 or moving.GetDimension() != 3:
        return None, (f"Dimensioni incompatibili: fissa {fixed.GetDimension()}D, "
                      f"mobile {moving.GetDimension()}D")

    # Posizione iniziale: centri di massa allineati (deterministica e uguale
    # per tutti i tentativi; la variabilita' viene dal campionamento).
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS)

    # 2. Multi-restart: stessa posizione iniziale, semi diversi; si tiene la
    #    metrica piu' bassa. Un tentativo fallito non ferma gli altri.
    best_tx = None
    best_mi = float("inf")
    mi_log: list[float] = []
    failures: list[str] = []

    for seed in range(int(n_restarts)):
        method = _build_method(initial, seed=seed)
        # la barra di avanzamento segue solo il primo tentativo
        if progress_callback and seed == 0:
            method.AddCommand(
                sitk.sitkIterationEvent,
                lambda mref=method: progress_callback(
                    min(100, mref.GetOptimizerIteration())))
        try:
            tx = method.Execute(fixed, moving)
            mi = float(method.GetMetricValue())
            mi_log.append(mi)
            if mi < best_mi:
                best_mi = mi
                best_tx = tx
        except Exception as e:
            failures.append(f"restart {seed}: {e}")
            logger.warning("Tentativo %d non riuscito: %s", seed, e)

    # 3. Se nessun tentativo e' riuscito resta l'allineamento per momenti.
    if best_tx is None:
        msg = ("Tutti i %d tentativi sono falliti (%s): uso l'allineamento "
               "iniziale per momenti." % (n_restarts, "; ".join(failures)))
        logger.error(msg)
        return initial, msg

    # 4. Messaggio con le statistiche di tutti i tentativi.
    mi_mean = float(np.mean(mi_log)) if mi_log else float("nan")
    mi_std  = float(np.std(mi_log))  if mi_log else float("nan")
    # Il formato di questa stringa e' letto da measure_registration_stability.py:
    # se lo si cambia, va cambiata anche l'espressione regolare la'.
    msg = (f"Multi-restart Mattes MI ({len(mi_log)}/{n_restarts} ok): "
           f"best={best_mi:.4f}  mean={mi_mean:.4f}  std={mi_std:.4f}  "
           f"trials={['{:.3f}'.format(x) for x in mi_log]}")
    if failures:
        msg += f"  failures={failures}"
    return best_tx, msg
