"""Modulo 02 — Allineamento rigido della PET sulla MRI dello stesso soggetto.

Due implementazioni intercambiabili, entrambe restituiscono
(sitk.Transform, messaggio):

  * run_registration_ants — ANTs (ANTsPy), rigida + informazione mutua di
    Mattes. E' il motore usato per tutti i risultati della tesi.
  * run_registration      — SimpleITK, rigida + Mattes con multi-restart
    (dieci ottimizzazioni con semi fissati, si tiene la migliore).

`get_registration(nome)` restituisce la funzione per "ants" o "sitk", cosi'
chi la chiama non dipende dal motore scelto.
"""

from .registration import run_registration
from .registration_ants import run_registration_ants

#: Motori di allineamento disponibili.
REGISTRATION_METHODS = {
    "ants": run_registration_ants,
    "sitk": run_registration,
}


def get_registration(method: str = "ants"):
    """Funzione di allineamento per `method` ("ants" | "sitk")."""
    key = (method or "ants").lower()
    if key not in REGISTRATION_METHODS:
        raise ValueError(
            f"Motore di allineamento sconosciuto {method!r}. "
            f"Disponibili: {sorted(REGISTRATION_METHODS)}")
    return REGISTRATION_METHODS[key]


__all__ = [
    "run_registration",
    "run_registration_ants",
    "get_registration",
    "REGISTRATION_METHODS",
]
