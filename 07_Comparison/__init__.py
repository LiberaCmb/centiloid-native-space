"""Modulo 07 — Confronto fra metodi e con i valori di riferimento."""

from .logic import (
    METODI,
    CATEGORIA,
    CARTELLE_RISULTATI,
    GROUND_TRUTH,
    SUVR_MIN_FBP,
    DLBS_QC_ESCLUSI,
    formule_pubblicate,
    gruppo_soggetto,
    gruppi_da_manifest,
    raccogli_risultati,
    leggi_ground_truth,
    filtro_qualita,
    metriche,
    calibrazione_regressione,
    calibrazione_two_point,
    leggi_coefficienti,
    tabelle_confronto,
)

__all__ = [
    "METODI", "CATEGORIA", "CARTELLE_RISULTATI", "GROUND_TRUTH",
    "SUVR_MIN_FBP", "DLBS_QC_ESCLUSI", "formule_pubblicate",
    "gruppo_soggetto", "gruppi_da_manifest", "raccogli_risultati",
    "leggi_ground_truth", "filtro_qualita", "metriche",
    "calibrazione_regressione", "calibrazione_two_point",
    "leggi_coefficienti", "tabelle_confronto",
]
