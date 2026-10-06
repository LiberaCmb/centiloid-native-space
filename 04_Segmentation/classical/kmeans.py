"""K-means sull'intensita' della MRI T1 (tre classi di tessuto).

I voxel nella maschera del cervello, dopo la correzione N4 e una leggera
levigatura gaussiana, vengono raggruppati in tre cluster (scikit-learn,
inizializzazione k-means++ con 20 ripartenze e seme fisso). I cluster vengono
poi assegnati ai tessuti in ordine di centroide: il piu' scuro e' il liquor, il
medio la sostanza grigia, il piu' chiaro la sostanza bianca.

Controllo di plausibilita': se il cluster assegnato alla sostanza grigia copre
meno del 15% o piu' del 70% del cervello, l'ordine dei centroidi non e'
affidabile e l'assegnazione si rifa' dividendo l'intervallo fra il centroide
minimo e quello massimo in tre fasce di intensita'.
"""

from __future__ import annotations
import logging
import numpy as np
import SimpleITK as sitk
from sklearn.cluster import KMeans

from ..base import BaseSegmenter, register_segmenter, n4_bias_correct

logger = logging.getLogger(__name__)


@register_segmenter("kmeans")
class KMeansSegmenter(BaseSegmenter):
    """K-means a tre classi: 1 = liquor, 2 = sostanza grigia, 3 = sostanza bianca."""

    label_scheme = "tissue3"

    def _segment(self, mri, external_brain_mask=None,
                 n_clusters: int = 3,
                 smooth_sigma_mm: float = 0.5,
                 n4_correct: bool = True,
                 n4_shrink: int = 4,
                 random_state: int = 42, **_):
        """Mappa a tre tessuti della MRI per raggruppamento delle intensita'.

        mri                   MRI T1 (sitk.Image)
        external_brain_mask   maschera comune dei metodi classici; se None,
                              ripiego di `_brain_mask`
        n_clusters            numero di cluster (3 = liquor, grigia, bianca)
        smooth_sigma_mm       sigma della levigatura gaussiana, in mm
                              (0 = nessuna)
        n4_correct            applica la correzione N4 prima del k-means
        n4_shrink             fattore di riduzione per la stima N4
        random_state          seme di k-means, fisso per la riproducibilita'

        Restituisce (mappa di etichette, extras con centroidi e frazioni).
        """
        # 1. Maschera del cervello e correzione N4 (se fallisce, intensita'
        #    originali).
        mask = self._brain_mask(mri, use_external_mask=external_brain_mask)
        if mask.sum() == 0:
            raise RuntimeError("KMeans: la maschera del cervello e' vuota.")

        if n4_correct:
            try:
                mri_corr = n4_bias_correct(mri, mask=mask, shrink=n4_shrink)
                logger.info("KMeans: correzione N4 applicata (shrink=%d).", n4_shrink)
            except Exception as e:
                logger.warning("KMeans: N4 non riuscita (%s), uso le intensita' "
                               "originali.", e)
                mri_corr = sitk.Cast(mri, sitk.sitkFloat32)
        else:
            mri_corr = sitk.Cast(mri, sitk.sitkFloat32)

        # 2. Leggera levigatura: evita etichette "sale e pepe"
        if smooth_sigma_mm > 0:
            sm = sitk.SmoothingRecursiveGaussianImageFilter()
            sm.SetSigma(smooth_sigma_mm)
            mri_s = sm.Execute(mri_corr)
        else:
            mri_s = mri_corr

        # 3. K-means sulla sola intensita' (una caratteristica per voxel) dei
        #    voxel del cervello. Le 20 ripartenze riducono il rischio di un
        #    minimo locale; il seme fisso rende il risultato ripetibile.
        arr = sitk.GetArrayFromImage(mri_s).astype(np.float32)
        feats = arr[mask > 0].reshape(-1, 1)

        km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=20,
                    random_state=random_state)
        cluster_ids = km.fit_predict(feats)
        centroids = km.cluster_centers_.ravel()

        counts = np.bincount(cluster_ids, minlength=n_clusters).astype(np.float64)
        fracs = counts / max(counts.sum(), 1.0)

        # 4. Assegnazione cluster -> tessuto in ordine di centroide.
        order = np.argsort(centroids)                 # dal piu' scuro
        tissue_for_sorted = np.full(n_clusters, 3, dtype=np.int16)
        tissue_for_sorted[0] = 1                      # liquor
        if n_clusters >= 2:
            tissue_for_sorted[1] = 2                  # sostanza grigia
        # gli altri restano sostanza bianca

        # 5. Controllo di plausibilita' sulla frazione di sostanza grigia
        #    (15-70% del cervello); fuori intervallo si riassegna dividendo
        #    [centroide minimo, massimo] in tre fasce di circa un terzo.
        sorted_centroids = centroids[order]
        gm_frac = fracs[order][1] if n_clusters >= 2 else 0.0
        if not (0.15 < gm_frac < 0.70):
            logger.warning("KMeans: il cluster della sostanza grigia copre il "
                           "%.1f%% del cervello (atteso 15-70%%): riassegno per "
                           "fasce di intensita'.", 100 * gm_frac)
            lo, hi = float(sorted_centroids.min()), float(sorted_centroids.max())
            mid_gm_wm = lo + 0.66 * (hi - lo)
            mid_csf_gm = lo + 0.33 * (hi - lo)
            for rank, c in enumerate(sorted_centroids):
                if c < mid_csf_gm:
                    tissue_for_sorted[rank] = 1
                elif c < mid_gm_wm:
                    tissue_for_sorted[rank] = 2
                else:
                    tissue_for_sorted[rank] = 3

        # 6. Dal rango del centroide all'identificativo del cluster, poi
        #    scrittura delle etichette nei voxel del cervello.
        cluster_to_tissue = np.zeros(n_clusters, dtype=np.int16)
        for rank, cid in enumerate(order):
            cluster_to_tissue[cid] = tissue_for_sorted[rank]

        out = np.zeros_like(arr, dtype=np.int16)
        out[mask > 0] = cluster_to_tissue[cluster_ids]

        # 7. Avviso se un tessuto copre meno del 2% del cervello.
        unique, ucounts = np.unique(out[mask > 0], return_counts=True)
        for u, c in zip(unique, ucounts):
            if u in (1, 2, 3) and c / max(feats.size, 1) < 0.02:
                logger.warning("KMeans: il tessuto %d copre solo il %.1f%%: "
                               "controllare N4 e maschera.", int(u),
                               100 * c / max(feats.size, 1))

        seg = self._np_to_sitk_like(out, mri)
        return seg, {
            "centroids": centroids.tolist(),
            "cluster_fracs": fracs.tolist(),
            "cluster_to_tissue": cluster_to_tissue.tolist(),
            "inertia": float(km.inertia_),
            "n4_correct": bool(n4_correct),
        }
