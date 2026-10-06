# Interfaccia Centiloide (Streamlit)

Interfaccia grafica per elaborare un paziente alla volta: data una MRI e una PET
amiloide, esegue **conversione → allineamento ANTs → segmentazione SynthSeg →
regioni di interesse → SUVr** e, quando le regioni sono quelle standard,
**Centiloide ricalibrato**, tutto nello spazio nativo del soggetto.

L'interfaccia usa la stessa pipeline della riga di comando
(`shared/pipeline.py`): non contiene calcoli propri, quindi i numeri coincidono.
Il metodo è SynthSeg, quello con l'errore più basso su FBB e FMM e fra i più
rapidi.

## Avvio (in WSL2)

SynthSeg gira dentro FreeSurfer, che è un programma Linux: l'app si avvia in
WSL2 (guida completa in `SETUP_WSL.md`).

```bash
cd /mnt/c/Users/<utente>/<cartella-del-progetto>
bash ui/setup.sh      # solo la prima volta: ambiente Python dell'app
bash ui/run_ui.sh     # ogni volta; poi aprire http://localhost:8501
```

`run_ui.sh` usa l'ambiente Python già attivo (conda o virtualenv) e carica
FreeSurfer da solo se è installato in `/usr/local/freesurfer`.

## Come si usa

1. **Dati del paziente**: MRI T1 e PET, come percorso su disco (file NIfTI o
   cartella DICOM, anche in stile Windows `C:\...`) oppure come file NIfTI
   caricato. Si sceglie il tracciante (FBB, FMM o FBP).
2. **Regioni su cui misurare**, in due modalità:
   * *Centiloide · regioni standard*: target corticale della specifica
     Centiloid e cervelletto intero. Restituisce SUVr e Centiloide.
   * *Solo SUVr · regioni a scelta*: target e riferimento composti fra tutte le
     regioni prodotte da SynthSeg. Restituisce il solo SUVr. Le regioni ★ sono
     quelle dello standard; ogni casella comprende i due emisferi. Le selezioni
     si possono salvare come *gruppi* (file `ui/label_groups.json`).
3. **Esegui**: alla fine compaiono SUVr, medie del target e del riferimento, il
   Centiloide quando è applicabile, l'anteprima delle regioni sopra MRI e PET
   (target in rosso, riferimento in azzurro) e la cartella dei risultati.

## Perché a volte il Centiloide non compare

La retta `CL = a·SUVr + b` è stimata su una coppia precisa di regioni (il target
della specifica Centiloid e il cervelletto intero, senza dilatazione). Su
regioni diverse il SUVr resta valido, ma applicargli quei coefficienti darebbe
un numero dall'aspetto di un Centiloide senza esserlo. L'interfaccia quindi lo
mostra solo con le regioni standard, e lo dice prima di eseguire. La regola sta
in `pipeline_service.centiloide_applicabile` e viene registrata in `result.csv`
(colonne `centiloide_applicabile` e `centiloide_motivo`).

I coefficienti `a` e `b` (SynthSeg, regressione) si leggono da
`data/comparison_all_methods/recalibration_final.csv` se esiste, altrimenti
dalla copia della tesi in `resources/coefficienti/`; se mancano entrambi l'app
usa i valori di riserva scritti nel codice e lo segnala.

## Che cosa finisce nella cartella dei risultati

`result.csv` (con il Centiloide ricalibrato), `info.json`, le immagini allineate,
la segmentazione, le VOI in NIfTI e le due anteprime `voi_overlay_mri.png` e
`voi_overlay_pet.png`.
