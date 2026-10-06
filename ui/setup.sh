#!/usr/bin/env bash
# ============================================================
#  Setup UNA-TANTUM dell'ambiente Python per l'interfaccia
#  Centiloide, dentro WSL2 / Linux.
#  Uso:   bash ui/setup.sh
# ============================================================
set -e

# Radice del progetto (una sopra a /ui)
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "============================================================"
echo " Setup ambiente interfaccia Centiloide (WSL2/Linux)"
echo " Progetto: $ROOT"
echo "============================================================"

# --- 1. Python 3 + venv ---
if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERRORE] python3 non trovato."
    echo "Installa con:  sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
    exit 1
fi

# python3-venv potrebbe mancare su Ubuntu appena installato
if ! python3 -c "import venv" >/dev/null 2>&1; then
    echo "[INFO] Il modulo venv non e' disponibile. Installalo con:"
    echo "       sudo apt update && sudo apt install -y python3-venv python3-pip"
    exit 1
fi

# Usa un venv DEDICATO a Linux/WSL2 (.venv-wsl), separato dall'eventuale
# .venv creato su Windows (che ha Scripts/ invece di bin/ e non e' attivabile
# qui). Cosi' i due ambienti non si pestano i piedi.
VENV=".venv-wsl"
if [ -f "$VENV/bin/activate" ]; then
    echo "Virtual environment $VENV gia' presente e valido: lo riuso."
else
    if [ -d "$VENV" ]; then
        echo "Trovato $VENV incompleto: lo ricreo."
        rm -rf "$VENV"
    fi
    echo "Creo il virtual environment $VENV ..."
    python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# --- 2. Dipendenze ---
echo "Aggiorno pip ..."
python -m pip install --upgrade pip
echo "Installo le dipendenze minime dell'app (SynthSeg + interfaccia)..."
pip install -r ui/requirements-app.txt

echo
echo "============================================================"
echo " Setup Python completato (ambiente: $VENV)."
echo " Avvia l'app con:   bash ui/run_ui.sh"
echo "============================================================"
echo
echo "RICORDA: SynthSeg richiede FreeSurfer (mri_synthseg) installato in WSL2"
echo "(vedi ui/SETUP_WSL.md)."
