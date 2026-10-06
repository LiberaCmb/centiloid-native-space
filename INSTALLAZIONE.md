# Installazione e primo avvio

Questa guida porta da un computer "vuoto" alla pipeline funzionante. Le parti
1–4 bastano per usare l'**interfaccia** (un paziente alla volta, con SynthSeg) e
per lanciare la pipeline da terminale con SynthSeg, Otsu e k-means. La parte 6
serve solo per rifare tutti i risultati della tesi.

Tutti i comandi sono stati provati su Windows 11 con WSL2 (Ubuntu 22.04),
Python 3.11 e FreeSurfer 8.2.0.

---

## Che cosa serve

| componente | a che cosa serve | obbligatorio? |
|---|---|---|
| Linux, oppure Windows con **WSL2** | FreeSurfer, FastSurfer e FSL esistono solo per Linux e macOS | sì |
| **Python 3.10–3.12** con i pacchetti di `requirements.txt` | tutta la pipeline e l'interfaccia | sì |
| **FreeSurfer ≥ 7.3** e la sua licenza gratuita | SynthSeg (anche nell'interfaccia) e SAMSEG | per SynthSeg e SAMSEG |
| FastSurfer | il metodo `fastsurfer` | solo per quel metodo |
| FSL | il metodo `fsl_fast` | solo per quel metodo |
| GPU NVIDIA | rende più veloci SynthSeg e FastSurfer | no |

Otsu e k-means non richiedono nulla oltre ai pacchetti Python. L'atlante ICBM152
e le VOI standard del Centiloid Project sono già nel progetto
(`resources/atlante/`), così come i coefficienti di ricalibrazione della tesi
(`resources/coefficienti/`).

**Non sono inclusi** i dati (immagini dei pazienti, non redistribuibili) e la
licenza di FreeSurfer, che è personale.

---

## 1. Linux dentro Windows (WSL2)

Su Linux o macOS si salta questo passo.

1. Aprire **PowerShell come amministratore** e lanciare:
   ```powershell
   wsl --install
   ```
   Riavviare il PC quando richiesto. Al primo avvio Ubuntu chiede un nome
   utente e una password.
2. Da quel momento, per entrare in Ubuntu basta scrivere `wsl` in PowerShell (o
   aprire "Ubuntu" dal menu Start). I file di Windows si trovano sotto
   `/mnt/c/...`: per esempio `C:\Users\Mario\Documenti` diventa
   `/mnt/c/Users/Mario/Documenti`.

Tutti i comandi che seguono vanno dati **dentro Ubuntu**.

## 2. Il codice

Con git (se il progetto è stato consegnato come repository):
```bash
sudo apt update && sudo apt install -y git
cd /mnt/c/Users/<utente>/Documenti          # o dove si preferisce
git clone <indirizzo-del-repository> centiloide
cd centiloide
```
Oppure si copia la cartella del progetto e ci si entra con `cd`.

## 3. Python e pacchetti

Si consiglia Miniconda (un ambiente separato, senza toccare il Python di
sistema):
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh       # accettare le opzioni predefinite
source ~/.bashrc

conda create -n centiloid python=3.11 -y
conda activate centiloid
pip install -r requirements.txt
```

Prova rapida, senza dati e senza strumenti esterni:
```bash
python -m pytest tests -q          # deve finire con "passed"
```

## 4. FreeSurfer (per SynthSeg e SAMSEG)

1. Registrarsi gratuitamente e scaricare la licenza (`license.txt`):
   https://surfer.nmr.mgh.harvard.edu/registration.html
2. Scaricare il pacchetto per Ubuntu dalla pagina ufficiale
   https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
   (per Ubuntu è un file `.deb`) e installarlo:
   ```bash
   sudo apt install ./freesurfer_<versione>_amd64.deb
   ```
   FreeSurfer finisce in `/usr/local/freesurfer/<versione>/`.
3. Copiare la licenza nella cartella di FreeSurfer:
   ```bash
   sudo cp /mnt/c/<percorso>/license.txt /usr/local/freesurfer/<versione>/license.txt
   ```
4. Fare in modo che FreeSurfer si carichi da solo a ogni apertura del
   terminale (sostituire `<versione>`, per esempio `8.2.0`):
   ```bash
   echo 'export FREESURFER_HOME=/usr/local/freesurfer/<versione>' >> ~/.bashrc
   echo 'source $FREESURFER_HOME/SetUpFreeSurfer.sh' >> ~/.bashrc
   source ~/.bashrc
   ```
5. Verifica:
   ```bash
   which mri_synthseg       # deve stampare un percorso
   ```

Attenzione: `FREESURFER_HOME` deve puntare alla cartella con la versione
(`.../freesurfer/8.2.0`), non a `.../freesurfer`. Con il percorso sbagliato
SynthSeg si ferma con l'errore `fspython: No such file or directory`.

---

## 5. Uso

### Interfaccia grafica (il modo più semplice)

```bash
cd <cartella-del-progetto>
conda activate centiloid
bash ui/run_ui.sh
```
Poi si apre **http://localhost:8501** nel browser (anche quello di Windows). Si
indicano la MRI T1 e la PET amiloide (file NIfTI o cartelle DICOM, anche con
percorsi in stile Windows `C:\...`), il tracciante, e si preme *Esegui*. Dopo
qualche minuto compaiono SUVr, centiloide ricalibrato e le immagini di
controllo. Per chiudere: `Ctrl+C` nel terminale. Dettagli in `ui/README.md`.

Da PowerShell, tutto in una riga:
```powershell
wsl -e bash -ic "cd /mnt/c/<cartella-del-progetto> && conda activate centiloid && bash ui/run_ui.sh"
```

### Un soggetto da terminale, con qualunque metodo

```bash
python -m scripts.run_pipeline \
    --mri <MRI: file NIfTI o cartella DICOM> --pet <PET> \
    --tracer FBB --segmenter synthseg --convert auto \
    --subject-id prova --output data/prove/prova_synthseg
```
`--tracer` è `FBB`, `FMM` o `FBP`; `--segmenter` uno fra `synthseg`,
`fastsurfer`, `samseg`, `samseg_atlas`, `otsu`, `kmeans`, `fsl_fast`. Alla fine
il terminale stampa SUVr e centiloide ricalibrato; nella cartella di uscita ci
sono le immagini allineate, `qc_allineamento.png` (da guardare prima di fidarsi
del numero), la segmentazione, le regioni e `result.csv`.

---

## 6. Rifare i risultati della tesi (facoltativo)

Servono, oltre ai passi precedenti:

* i dati: gli archivi del Centiloid Project scaricati da GAAIN
  (https://www.gaain.org/centiloid-project) in `../datasets/` e il Dallas
  Lifespan Brain Study in `../DLBS/` (i percorsi si cambiano con le variabili
  descritte in `README.md`, sezione *Dove stanno i dati*);
* FastSurfer, per il metodo `fastsurfer`: repository
  https://github.com/Deep-MI/FastSurfer, ambiente conda dedicato `fastsurfer`
  e variabile `FASTSURFER_HOME` che punta alla cartella clonata;
* FSL, per il metodo `fsl_fast`: https://fsl.fmrib.ox.ac.uk/fsl/docs/install/
  (deve essere disponibile il comando `fast`).

L'ordine degli script, dalla preparazione dei dati alle figure, è in
`README.md`, sezione *Riprodurre i risultati della tesi*. Elaborare tutti i
soggetti con tutti i metodi richiede molte ore.

---

## Problemi frequenti

| messaggio | causa e rimedio |
|---|---|
| `mri_synthseg: command not found` | FreeSurfer non caricato: ripetere il passo 4.4 o aprire un terminale nuovo |
| `fspython: No such file or directory` | `FREESURFER_HOME` senza la versione: deve essere `/usr/local/freesurfer/<versione>` |
| `Streamlit non è installato in questo ambiente` | ambiente sbagliato: `conda activate centiloid` |
| `run_samseg non trovato` | come per SynthSeg: FreeSurfer non caricato |
| `FASTSURFER_HOME` non impostata | serve solo per il metodo `fastsurfer` (parte 6) |
| la pagina `localhost:8501` non si apre | controllare che il terminale con `run_ui.sh` sia ancora aperto |
| lentezza su `/mnt/c` | l'accesso ai file di Windows da WSL è più lento: per elaborazioni lunghe si può copiare il progetto nella home di Ubuntu |
