# Centiloide in spazio nativo: segmentazione diretta della MRI

Codice della tesi magistrale in Matematica di Libera Ciambriello.

La pipeline standard del Centiloid Project (Klunk et al. 2015) deforma ogni
soggetto su un cervello di riferimento (MNI152) e vi applica le regioni di
interesse standard. Qui si fa il contrario: si resta nello **spazio nativo** del
soggetto, si **segmenta direttamente la sua MRI** e da quella segmentazione si
ricavano le regioni su cui misurare il segnale PET. Si confrontano sette metodi
di segmentazione (classici, appresi, bayesiani) su tre traccianti
(florbetaben, flutemetamol, florbetapir) del Centiloid Project e si valida il
risultato sul Dallas Lifespan Brain Study (DLBS).

I risultati usati nella tesi sono riproducibili con gli script elencati nella
sezione [Riprodurre i risultati](#riprodurre-i-risultati-della-tesi).

---

## La pipeline

```
 MRI T1 ─┐
         ├─ 00 conversione ─ 01 caricamento ─ 02 allineamento PET→MRI ─ 03 ricampionamento + controllo
 PET  ───┘                                                                         │
                                                                                   ▼
            07 confronto ◄─ 06 SUVr e Centiloide ◄─ 05 regioni di interesse ◄─ 04 segmentazione
```

| modulo | cartella | che cosa fa |
|---|---|---|
| 00 | `00_DicomToNifti/` | DICOM → NIfTI con dcm2niix (o SimpleITK se manca), riorientamento in RAS |
| 01 | `01_LoadImages/` | lettura di MRI e PET e della loro geometria |
| 02 | `02_Realignment/` | allineamento rigido PET→MRI con informazione mutua: **ANTs** (usato nella tesi) o SimpleITK con multi-restart |
| 03 | `03_VisualCheck/` | PET ricampionata sulla griglia MRI; figure di controllo dell'allineamento e delle VOI |
| 04 | `04_Segmentation/` | i segmentatori: `otsu`, `kmeans`, `fsl_fast`, `synthseg`, `fastsurfer`, `samseg` |
| 05 | `05_VOIExtraction/` | target corticale e cervelletto, dalle etichette FreeSurfer o dalle VOI standard dell'atlante |
| 06 | `06_CentiloidCalculation/` | SUVr = media target / media cervelletto, CL = a·SUVr + b |
| 07 | `07_Comparison/` | raccolta dei risultati, controllo qualità, metriche, ricalibrazione (regressione e two-point) |

La sequenza dei moduli sta in un unico punto, `shared/pipeline.py`, usato sia
dalla riga di comando sia dall'interfaccia: i due danno lo stesso risultato.
La MRI è l'immagine fissa dell'allineamento e non viene mai spostata, quindi
le regioni di interesse non dipendono dall'allineamento; solo la lettura del
segnale PET ne dipende.

### I sette metodi

| metodo | famiglia | come si ottengono le regioni | strumento esterno |
|---|---|---|---|
| `synthseg` | rete convoluzionale | etichette DKT (parcellazione corticale) | FreeSurfer ≥ 7.3 (`mri_synthseg`) |
| `fastsurfer` | rete convoluzionale | etichette DKT (+ CerebNet per il cervelletto) | FastSurfer (ambiente conda) |
| `samseg` | bayesiano | corteccia intera (etichette 3, 42) | FreeSurfer (`run_samseg`) |
| `samseg_atlas` | bayesiano + atlante | VOI standard ∩ corteccia di SAMSEG | FreeSurfer + ANTs |
| `otsu` | classico | VOI standard ∩ sostanza grigia | — |
| `kmeans` | classico | VOI standard ∩ sostanza grigia | — |
| `fsl_fast` | classico | VOI standard ∩ sostanza grigia | FSL (`fast`) |

I metodi classici distinguono solo liquor, sostanza grigia e bianca: l'anatomia
viene dalle VOI standard del Centiloid Project, portate sul soggetto
registrandovi il template ICBM152 (ANTs, affine + SyN). Il paziente non viene
deformato; si muove il template. La stessa registrazione fornisce la maschera
del cervello comune ai tre metodi classici.

Il `centiloid` scritto in ogni `result.csv` usa la formula pubblicata del
tracciante. I risultati della tesi usano invece coefficienti **ricalibrati per
regressione**, uno per coppia metodo–tracciante, che si trovano in
`data/comparison_all_methods/recalibration_final.csv`.

---

## Struttura del progetto

```
Thesis-libera-consegna/
├── 00_DicomToNifti/ … 07_Comparison/   i moduli della pipeline
├── shared/
│   ├── pipeline.py        la pipeline completa su un soggetto
│   ├── percorsi.py        dove si trovano dati, atlante e archivi originali
│   ├── context.py         stato di un'esecuzione
│   ├── io_utils.py, logging_utils.py
├── resources/atlante/     template ICBM152 lineare, sua maschera, VOI standard GAAIN
├── scripts/
│   ├── run_pipeline.py    un soggetto
│   ├── batch_process.py   tutti i soggetti di un manifest
│   ├── preparazione/      manifest, conversione in NIfTI, valori di riferimento
│   ├── esecuzione/        samseg_atlas, riferimento FreeSurfer del DLBS, dilatazione
│   ├── analisi/           confronti, validazione, stabilità, tempi
│   └── figure/            figure della tesi e figure di controllo
├── ui/                    interfaccia grafica (Streamlit)
├── tests/                 prove automatiche su dati sintetici
├── main.py                equivale a `python -m scripts.run_pipeline`
└── requirements.txt
```

---

## Installazione

La guida completa, da un computer senza nulla fino al primo avvio
dell'interfaccia, è in **`INSTALLAZIONE.md`**. In breve:

```bash
# dentro Linux o WSL2 (Ubuntu), nella cartella del progetto
conda create -n centiloid python=3.11 -y && conda activate centiloid
pip install -r requirements.txt
python -m pytest tests -q                               # prova rapida, senza dati
```

Strumenti esterni, solo per i metodi che li richiedono:

* **FreeSurfer ≥ 7.3** per SynthSeg e SAMSEG, con la licenza gratuita
  (https://surfer.nmr.mgh.harvard.edu/registration.html). `FREESURFER_HOME`
  deve puntare alla cartella della versione (per esempio
  `/usr/local/freesurfer/8.2.0`).
* **FastSurfer**: `git clone https://github.com/Deep-MI/FastSurfer`, ambiente
  conda dedicato chiamato `fastsurfer`, variabile `FASTSURFER_HOME`.
* **FSL** per FSL FAST, con `fast` nel PATH.

Otsu e k-means non richiedono nulla oltre ai pacchetti Python.

---

## Dove stanno i dati

La pipeline cerca tutto a partire dalla cartella del progetto; nessun percorso
è scritto nel codice.

| che cosa | dove (predefinito) | variabile per spostarlo |
|---|---|---|
| risultati, manifest, valori di riferimento | `data/` | `CENTILOID_DATI` |
| atlante (incluso nel progetto) | `resources/atlante/` | `CENTILOID_ATLANTE` |
| coefficienti di ricalibrazione della tesi (inclusi) | `resources/coefficienti/` | si usa la copia in `data/` se esiste |
| archivi GAAIN scaricati | `../datasets/` | `CENTILOID_DATASETS` |
| DLBS (BIDS) | `../DLBS/` | `CENTILOID_DLBS` |
| elaborazioni FreeSurfer del DLBS | `../freesurfer/` | opzione `--fs-root` |

I file di riferimento attesi in `data/ground_truth/` sono `gaain_fbb.csv`,
`gaain_fmm.csv`, `gaain_fbp.csv` (prodotti da
`scripts/preparazione/build_groundtruth.py`) e `QAmyloid_results_DLBS.xlsx`
(centiloidi QAmyloid del DLBS, fornito con il dataset di validazione).

La cartella `data/` è esclusa dal controllo di versione: contiene immagini di
pazienti, non redistribuibili.

---

## Uso

**Un soggetto**, dalle cartelle DICOM o dai NIfTI:

```bash
python -m scripts.run_pipeline \
    --mri "../datasets/<archivio>/2002_MR" --pet "../datasets/<archivio>/2002_PET_FBB" \
    --tracer FBB --segmenter synthseg --convert auto \
    --subject-id 2002 --output data/prove/2002_synthseg
```

Nella cartella di uscita: immagini allineate, `qc_allineamento.png` (da guardare
prima di fidarsi del numero), segmentazione, VOI, `result.csv` e `info.json`
(con i parametri effettivi del metodo e del modulo 05).

**Tutti i soggetti di un archivio**, con uno o più metodi:

```bash
python -m scripts.batch_process --manifest data/manifest_fbb_nifti.csv \
    --segmenters synthseg fastsurfer samseg \
    --cache-dir data/aligned_ants --output data/results_fbb_ants --skip-done
```

Ogni soggetto viene allineato una volta sola e tutti i metodi leggono la stessa
PET allineata.

**Interfaccia grafica** (SynthSeg, un paziente alla volta, regioni a scelta):

```bash
bash ui/setup.sh      # la prima volta
bash ui/run_ui.sh     # poi http://localhost:8501
```

Dettagli in `ui/README.md`.

---

## Riprodurre i risultati della tesi

Tutti i comandi si lanciano dalla cartella del progetto. Le tabelle finiscono in
`data/comparison_all_methods/`, le figure in `data/figure_tesi/`.

### 1. Preparazione dei dati

| passo | comando | uscita |
|---|---|---|
| manifest FBB e FMM | `python -m scripts.preparazione.build_manifest_fbb` (e `_fmm`) | `data/manifest_fbb.csv`, `data/manifest_fmm.csv` |
| conversione in NIfTI (modulo 00) | `python -m scripts.preparazione.convert_dicom_to_nifti --manifest data/manifest_fbb.csv` (e FMM) | `data/nifti/…`, `data/manifest_*_nifti.csv` |
| florbetapir: DICOM, media dei due frame | `python -m scripts.preparazione.prepare_fbp_nifti` | `data/nifti_fbp/…`, `data/manifest_fbp_nifti.csv` |
| manifest DLBS | `python -m scripts.preparazione.build_manifest_dlbs` | `data/manifest_dlbs.csv` |
| valori di riferimento GAAIN | `python -m scripts.preparazione.build_groundtruth` | `data/ground_truth/gaain_{fbb,fmm,fbp}.csv` |

### 2. Esecuzione dei metodi

Per ogni archivio (FBB mostrato; FMM e FBP analoghi, DLBS in `data/results_dlbs`
con cache `data/aligned_dlbs_ants`):

```bash
python -m scripts.batch_process --manifest data/manifest_fbb_nifti.csv \
    --segmenters synthseg fastsurfer samseg \
    --cache-dir data/aligned_ants --output data/results_fbb_ants --skip-done
python -m scripts.batch_process --manifest data/manifest_fbb_nifti.csv \
    --segmenters otsu kmeans fsl_fast \
    --cache-dir data/aligned_ants --output data/results_fbb_atlas --skip-done
python -m scripts.esecuzione.recompute_samseg_atlas --results-dir data/results_fbb_ants
```

Per il DLBS, il riferimento anatomico FreeSurfer:
`python -m scripts.esecuzione.riferimento_freesurfer_dlbs`.

### 3. Analisi, nell'ordine della tesi

| capitolo / argomento | script (`python -m …`) | uscita |
|---|---|---|
| Cap. 3 — variabilità di SimpleITK al cambiare del seme (79 soggetti, archivio PiB 50-70 min) | `scripts.analisi.measure_registration_stability --manifest … --out data/registration_stability.csv` | `registration_stability.csv` |
| Cap. 3 — ANTs contro SimpleITK: NMI e correlazione dei gradienti | `scripts.analisi.compare_alignment_quality` | `alignment_quality.csv` |
| Cap. 3 — ANTs contro SimpleITK: errore sul centiloide | `scripts.analisi.confronto_motori` | `confronto_motori.csv/.xlsx` |
| Cap. 3 — ripetibilità di ANTs, quattro configurazioni | `scripts.analisi.diag_ants_determinismo --modo default` (poi `thread1`, `seme`, `deterministico`) | `data/diag_ants_determinismo/` |
| Cap. 3 — incertezza dell'allineamento su 97 soggetti | `scripts.analisi.diag_ants_stabilita --dataset tutti` | `data/stabilita_ants/` |
| Cap. 3-4 — confronto dei sette metodi, ricalibrazioni, coefficienti | `scripts.analisi.build_final_comparison` | `confronto_metodi_centiloide_ricalibrato_ANTs_finale.xlsx`, `recalibration_final.csv` |
| Cap. 4 — sovrapposizione delle VOI con FreeSurfer (DLBS) | `scripts.analisi.dice_vs_freesurfer` | `dice_*.csv`, `dice_vs_freesurfer.xlsx` |
| Cap. 4 — tempi di calcolo | `scripts.analisi.report_runtimes_final` | `tempi_*.csv`, `tempi_calcolo.xlsx` |
| Cap. 4 — validazione sul DLBS con i coefficienti GAAIN | `scripts.analisi.validate_dlbs` | `validazione_dlbs.csv/.xlsx` |
| Cap. 4 — peso degli outlier | `scripts.analisi.robustezza_outlier` | `robustezza_regressione.csv/.xlsx` |
| Cap. 4 — dilatazione di un voxel del target | `scripts.esecuzione.dilation_recompute --dataset FBB` (e FMM, FBP, DLBS), poi `scripts.analisi.analisi_dilatazione` | `data/comparison_all_methods/dilation_*.csv`, `confronto_dilatazione_centiloide_sintesi.xlsx` |

### 4. Figure

| figura | script (`python -m …`) |
|---|---|
| risultati del cap. 4 (`fig_res_*.pdf`) | `scripts.figure.figure_risultati` |
| incertezza dell'allineamento contro il SUVr (`stabilita_vs_suvr_en.pdf`) | `scripts.figure.figura_stabilita --lingua en` |
| schema della pipeline (`fig_pipeline_en.pdf`) | `scripts.figure.figure_pipeline` |
| PET positiva/negativa, controllo dell'allineamento, VOI di due metodi, soggetto DLBS escluso | `scripts.figure.figure_capitoli` |
| sette metodi sullo stesso cervello, VOI dei metodi con etichette, SAMSEG e samseg_atlas, VOI dall'atlante al soggetto, maschera dilatata, FreeSurfer/SynthSeg/FastSurfer su un soggetto DLBS, semi di SimpleITK, esclusione FBP, calibrazione SynthSeg, dilatazione | `scripts.figure.figure_aggiuntive` (la prima volta la registrazione SyN richiede qualche minuto) |
| ANTs contro SimpleITK sullo stesso soggetto (`fig_ants_vs_sitk.pdf`) | `scripts.figure.figura_ants_vs_sitk` (rifà l'allineamento SimpleITK, qualche minuto la prima volta) |
| controllo visivo dell'allineamento, un PNG per soggetto | `scripts.figure.qc_overlay --all` |
| controllo visivo delle VOI, un PNG per soggetto e metodo | `scripts.figure.qc_voi_overlay --results-dir … --out …` |

Le figure didattiche del capitolo 2 (Otsu, Dice, Bland-Altman, retta SUVr→CL)
non sono prodotte da questo codice, tranne quella del multi-Otsu su una MRI
reale (`fig_multi_otsu.pdf`, `scripts.figure.figura_multi_otsu`) e di quelle
sugli altri segmentatori, sempre sul soggetto 1018 (`fig_kmeans.pdf`,
`fig_fast.pdf`, `fig_samseg.pdf`, `fig_dl_parcellation.pdf`,
`scripts.figure.figure_segmentatori`).

### Verifica sul codice consegnato

`scripts.analisi.verifica_riproducibilita` riesegue la pipeline su alcuni
soggetti e confronta SUVr e centiloide ricalibrato con i `result.csv` della
tesi, scrivendo i nuovi risultati in `data/verifica/` senza toccare gli altri.
Con `--modo riusa` (predefinito) parte dalle immagini già allineate, con
`--modo completo` rifà anche l'allineamento ANTs.

```bash
python -m scripts.analisi.verifica_riproducibilita --metodi otsu kmeans synthseg
```

### Ripetibilità

L'allineamento ANTs con le impostazioni predefinite non è deterministico: su un
soggetto, cinque ripetizioni differiscono fino a 1.4 CL (mediana sui 97
soggetti: 0.07–0.10 CL). Per risultati identici al decimale:

```bash
export CENTILOID_ANTS_DETERMINISTIC=1     # seme fisso, multithread attivo
```

La variabile vale anche per la registrazione del template ICBM152 dei metodi con
atlante, che usa ANTs allo stesso modo. Le segmentazioni classiche sono
deterministiche (k-means usa un seme fisso).

---

## Prove

```bash
python -m pytest tests -q
```

Le prove usano volumi sintetici e controllano segmentatori classici, regioni
FreeSurfer, dilatazione, calcolo del Centiloide, ricalibrazioni, filtro di
qualità e coerenza fra le etichette dell'interfaccia e quelle della pipeline.
