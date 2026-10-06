"""Prove rapide su dati sintetici: non servono dati reali, FreeSurfer, FSL o ANTs.

    python -m pytest tests -q
"""

from __future__ import annotations
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_seg = importlib.import_module("04_Segmentation")
_voi = importlib.import_module("05_VOIExtraction")
_cl = importlib.import_module("06_CentiloidCalculation")
_cmp = importlib.import_module("07_Comparison")


def _immagine(arr):
    """sitk.Image da un array numpy, con voxel di 1 mm e orientamento identita'."""
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return img


def _cervello_sintetico(size=(64, 64, 64), seed=0):
    """T1 finta con tre ellissoidi concentrici (liquor, grigia, bianca)."""
    rng = np.random.default_rng(seed)
    z, y, x = np.indices(size)
    c = (np.array(size) - 1) / 2

    def ellissoide(r):
        return ((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2) <= r ** 2

    cervello = ellissoide(28)
    grigia = ellissoide(24) & ~ellissoide(18)
    bianca = ellissoide(18)
    liquor = cervello & ~ellissoide(26)
    t1 = np.zeros(size, np.float32)
    t1[liquor], t1[grigia], t1[bianca] = 30, 90, 150
    t1 += rng.normal(0, 3, size).astype(np.float32)
    t1[~cervello] = rng.uniform(0, 5, size).astype(np.float32)[~cervello]
    return _immagine(t1), cervello, grigia, bianca


@pytest.mark.parametrize("nome", ["otsu", "kmeans"])
def test_segmentatori_classici(nome):
    """Otsu e k-means riconoscono grigia e bianca sul cervello sintetico."""
    mri, cervello, grigia, bianca = _cervello_sintetico()
    mask = _immagine(cervello.astype(np.uint8))
    res = _seg.get_segmenter(nome).segment(mri=mri, external_brain_mask=mask)
    lab = sitk.GetArrayFromImage(res.label_map)
    assert res.label_map.GetSize() == mri.GetSize()
    assert res.label_scheme == "tissue3"
    # la maggior parte della sostanza grigia e bianca va riconosciuta
    assert (lab[grigia] == 2).mean() > 0.8
    assert (lab[bianca] == 3).mean() > 0.8


def test_voi_freesurfer_e_centiloide():
    """Etichette FreeSurfer sintetiche: target e cervelletto con SUVr noto."""
    seg = np.zeros((40, 40, 40), np.int16)
    seg[25:35, 10:30, 10:30] = 1028          # frontale superiore sinistro
    seg[5:12, 10:30, 10:30] = 8              # corteccia cerebellare sinistra
    pet = np.zeros_like(seg, np.float32)
    pet[seg == 1028] = 2.6
    pet[seg == 8] = 2.0
    voi_ctx, voi_wc, info = _voi.extract_vois(_immagine(seg), "freesurfer")
    assert info["ctx_voxels"] == 10 * 20 * 20
    assert info["wc_voxels"] == 7 * 20 * 20
    _, _, suvr, cl, _ = _cl.compute_centiloid(_immagine(pet), voi_ctx, voi_wc, "FBB")
    assert suvr == pytest.approx(1.3)
    assert cl == pytest.approx(153.4 * 1.3 - 154.9)


def test_voi_samseg_senza_parcellazione():
    """Senza etichette aparc il target diventa la corteccia intera (3, 42)."""
    seg = np.zeros((20, 20, 20), np.int16)
    seg[10:15] = 3
    seg[2:5] = 47
    _, _, info = _voi.extract_vois(_immagine(seg), "freesurfer")
    assert info["method"] == "freesurfer_label_lookup_aseg"
    assert info["ctx_voxels"] == 5 * 400


def test_dilatazione_resta_nel_cervello():
    """La dilatazione del target aggiunge esattamente un voxel per faccia."""
    seg = np.zeros((20, 20, 20), np.int16)
    seg[5:15, 5:15, 5:15] = 2
    seg[9:11, 9:11, 9:11] = 1028
    _, _, info = _voi.extract_vois(_immagine(seg), "freesurfer", ctx_dilation_iters=1)
    assert info["ctx_voxels_before_dilation"] == 8
    assert info["ctx_voxels"] == 8 + 6 * 4     # dilatazione a croce di un voxel


def test_schema_classico_senza_atlante_rifiutato():
    """Una segmentazione a tre tessuti senza atlante non basta per le VOI."""
    with pytest.raises(ValueError):
        _voi.extract_vois(_immagine(np.ones((5, 5, 5), np.int16)), "tissue3")


def test_preset_interfaccia_coincide_con_label_mappings():
    """I preset standard dell'interfaccia coincidono con quelli del modulo 05."""
    from ui import freesurfer_labels as fsl
    assert fsl.CENTILOID_STANDARD_CTX == _voi.FREESURFER_CTX_LABELS
    assert fsl.CENTILOID_STANDARD_WC == _voi.FREESURFER_WC_LABELS
    assert len(_voi.FREESURFER_CTX_LABELS) == 38


def test_calibrazioni():
    """Regressione e two-point ritrovano una retta nota; two-point senza
    abbastanza soggetti per gruppo restituisce None."""
    suvr = np.array([1.0, 1.1, 1.2, 1.8, 2.0, 2.2])
    vero = 150 * suvr - 150
    a, b = _cmp.calibrazione_regressione(suvr, vero)
    assert (a, b) == pytest.approx((150, -150))
    gruppi = ["YC", "YC", "YC", "AD", "AD", "AD"]
    assert _cmp.calibrazione_two_point(suvr, vero, gruppi) == pytest.approx((150, -150))
    assert _cmp.calibrazione_two_point(suvr, vero, ["YC"] + ["AD"] * 5) is None


def test_gruppi_dagli_identificativi():
    """Gruppo YC/AD dedotto dalle convenzioni degli identificativi GAAIN."""
    attesi = {"Y1001": "YC", "1008": "AD", "YHV_001": "YC", "AD_100": "AD", "x": ""}
    for sid, g in attesi.items():
        assert _cmp.gruppo_soggetto(sid) == g


def test_filtro_qualita():
    """Un soggetto con SUVr di riferimento sotto soglia e' escluso per intero."""
    df = pd.DataFrame({"subject_id": ["a", "a", "b", "b"],
                       "segmenter": ["synthseg", "otsu"] * 2,
                       "suvr": [1.1, 1.0, 0.7, 1.2]})
    out = _cmp.filtro_qualita(df, 0.85, "prova")
    assert set(out.subject_id) == {"a"}


def test_risolvi_percorsi(monkeypatch):
    """I percorsi 'data/...' seguono la cartella dati, gli altri restano fissi."""
    from shared import percorsi
    monkeypatch.setattr(percorsi, "DATI", Path("/altrove/dati"))
    assert percorsi.risolvi("data/aligned_ants/1008/x.nii.gz") == \
        Path("/altrove/dati/aligned_ants/1008/x.nii.gz")
    assert percorsi.risolvi("/assoluto/x") == Path("/assoluto/x")
    assert percorsi.risolvi("resources/atlante") == percorsi.ROOT / "resources/atlante"


def test_metodi_disponibili():
    """Registro dei segmentatori ed elenco dei metodi coerenti fra i moduli."""
    assert set(_seg.list_segmenters()) == {"otsu", "kmeans", "fsl_fast",
                                           "synthseg", "fastsurfer", "samseg"}
    from shared.pipeline import METODI
    assert METODI == _cmp.METODI
