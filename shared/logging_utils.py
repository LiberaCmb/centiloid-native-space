"""Configurazione del logger: usa `rich` se installato, altrimenti testo semplice."""

from __future__ import annotations
import logging
import sys


def setup_logger(name: str = "centiloide",
                 level: int = logging.INFO) -> logging.Logger:
    """Restituisce il logger `name`, configurandolo solo alla prima chiamata.

    name    nome del logger
    level   livello minimo dei messaggi

    Se il logger ha gia' un gestore non lo si tocca, cosi' chiamate ripetute
    non duplicano le righe. `propagate = False` evita che il logger radice le
    stampi una seconda volta.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    try:
        from rich.logging import RichHandler
        handler = RichHandler(rich_tracebacks=True, show_time=True,
                              show_level=True, show_path=False)
        handler.setFormatter(logging.Formatter("%(message)s"))
    except ImportError:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S"))

    logger.addHandler(handler)
    logger.propagate = False
    return logger
