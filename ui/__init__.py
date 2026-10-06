"""Interfaccia grafica (Streamlit) per il calcolo del Centiloide su un paziente.

  freesurfer_labels.py   catalogo delle regioni FreeSurfer selezionabili e
                         preset del Centiloide standard;
  pipeline_service.py    esegue la pipeline (la stessa della riga di comando,
                         shared/pipeline.py) e applica la ricalibrazione;
  app.py                 la pagina Streamlit (avvio: bash ui/run_ui.sh).

L'interfaccia non contiene calcoli propri: orchestra i moduli 00-06.
"""
