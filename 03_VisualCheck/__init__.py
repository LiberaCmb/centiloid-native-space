"""Modulo 03 — Ricampionamento delle immagini allineate e controllo visivo.

* `ImageFusionLogic.save_realigned_images_with_transform` ricampiona la PET
  sulla griglia della MRI e salva le due immagini allineate, che sono
  l'ingresso dei moduli successivi.
* `figura_allineamento` sovrappone la PET allineata alla MRI: e' l'immagine su
  cui si verifica a occhio l'allineamento (sezione "Quality control" del
  capitolo 3).
* `figura_voi` mostra le due regioni di interesse sopra la MRI o la PET.

Le funzioni grafiche importano matplotlib solo quando vengono chiamate.
"""

from .logic import ImageFusionLogic
from .figure_qc import figura_allineamento, figura_voi

__all__ = ["ImageFusionLogic", "figura_allineamento", "figura_voi"]
