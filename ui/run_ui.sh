#!/usr/bin/env bash
# ============================================================
#  Avvio dell'interfaccia Centiloide (Streamlit) dentro WSL2/Linux.
#
#  Uso:   bash ui/run_ui.sh
#  Poi apri http://localhost:8501 nel browser di Windows.
#
#  AMBIENTE PYTHON
#  ---------------
#  Lo script usa l'ambiente in cui lo lanci, con questa precedenza:
#
#    1. la variabile CENTILOID_PYTHON, se la imposti tu;
#    2. l'ambiente gia' attivo (conda oppure virtualenv);
#    3. .venv-wsl, se esiste;
#    4. il 'python' che trova nel PATH.
#
#  Per forzare un interprete preciso:
#    CENTILOID_PYTHON=$HOME/miniconda3/envs/centiloid/bin/python bash ui/run_ui.sh
# ============================================================
set -e

cd "$(dirname "$0")/.."

# --- 1. Interprete Python -------------------------------------------------
if [ -n "${CENTILOID_PYTHON:-}" ]; then
    PY="$CENTILOID_PYTHON"
    ORIGINE="variabile CENTILOID_PYTHON"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PY="$CONDA_PREFIX/bin/python"
    ORIGINE="ambiente conda attivo ($(basename "$CONDA_PREFIX"))"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PY="$VIRTUAL_ENV/bin/python"
    ORIGINE="virtualenv attivo ($(basename "$VIRTUAL_ENV"))"
elif [ -x ".venv-wsl/bin/python" ]; then
    PY=".venv-wsl/bin/python"
    ORIGINE=".venv-wsl del progetto"
elif command -v python >/dev/null 2>&1; then
    PY="$(command -v python)"
    ORIGINE="python del PATH"
else
    echo "[ERRORE] Nessun interprete Python trovato."
    echo "         Attiva l'ambiente della pipeline, oppure indica l'interprete:"
    echo "           CENTILOID_PYTHON=/percorso/a/python bash ui/run_ui.sh"
    exit 1
fi

echo "[Python] $ORIGINE"
echo "         $("$PY" -c 'import sys; print(sys.executable, "-", sys.version.split()[0])')"

if ! "$PY" -c "import streamlit" 2>/dev/null; then
    echo
    echo "[ERRORE] Streamlit non e' installato in questo ambiente."
    echo "         Installalo con:   $PY -m pip install streamlit"
    exit 1
fi

# --- 2. FreeSurfer (per SynthSeg) ----------------------------------------
# Se il tuo .bashrc lo carica gia', non tocchiamo niente: sovrascrivere
# FREESURFER_HOME con un percorso sbagliato e' un classico modo per far
# sparire mri_synthseg.
if [ -z "${FREESURFER_HOME:-}" ]; then
    for CAND in /usr/local/freesurfer/*/SetUpFreeSurfer.sh \
                /usr/local/freesurfer/SetUpFreeSurfer.sh; do
        if [ -f "$CAND" ]; then
            export FREESURFER_HOME="$(dirname "$CAND")"
            # shellcheck disable=SC1090
            source "$CAND" >/dev/null 2>&1 || true
            break
        fi
    done
fi

if command -v mri_synthseg >/dev/null 2>&1; then
    echo "[OK]     SynthSeg disponibile: $(command -v mri_synthseg)"
else
    echo "[NOTA]   SynthSeg non e' sul PATH: la segmentazione dara' errore."
    echo "         Carica FreeSurfer prima di lanciare questo script."
fi

# --- 3. Avvio ------------------------------------------------------------
echo
echo "Avvio dell'interfaccia... apri  http://localhost:8501  nel browser."
echo "Per chiudere: premi CTRL+C in questo terminale."
echo
exec "$PY" -m streamlit run ui/app.py
