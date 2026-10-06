# Guida: eseguire l'interfaccia in WSL2 (Windows)

SynthSeg (via FreeSurfer) è uno strumento **Linux**. Su Windows si esegue dentro
**WSL2** (il sottosistema Linux integrato in Windows), che supporta la GPU
NVIDIA. Faremo girare *tutto* dentro Ubuntu (WSL2): FreeSurfer, l'ambiente
Python e l'app Streamlit. L'app si apre poi nel browser di Windows. I dati
restano in locale sul tuo computer.

Ordine consigliato: **A → B → C → D → E**.

---

## A. Installare WSL2 + Ubuntu

1. Apri **PowerShell come amministratore** e lancia:
   ```powershell
   wsl --install
   ```
   Installa WSL2 e Ubuntu. **Riavvia** il PC quando richiesto.
2. Al primo avvio Ubuntu chiede uno username e una password (Linux).
3. Verifica di essere su WSL **2**:
   ```powershell
   wsl -l -v      # la colonna VERSION deve dire 2
   ```

## B. GPU in WSL2

Non serve installare driver NVIDIA *dentro* Ubuntu: basta avere l'ultimo
**driver NVIDIA per Windows** (quello normale della scheda, con supporto WSL).
Verifica dentro Ubuntu:
```bash
nvidia-smi        # deve mostrare la scheda NVIDIA
```
La GPU velocizza SynthSeg ma non è obbligatoria (SynthSeg va bene anche su CPU).

## C. Preparare l'ambiente Python dell'app

Il progetto sta su Windows in `C:\Users\<utente>\<cartella-del-progetto>`, che da
Ubuntu si raggiunge in `/mnt/c/Users/<utente>/<cartella-del-progetto>`.

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip git

cd /mnt/c/Users/<utente>/<cartella-del-progetto>
bash ui/setup.sh        # crea .venv-wsl e installa le dipendenze (antspyx, streamlit, ...)
```

> Suggerimento: puoi aprire la cartella in VS Code con l'estensione **WSL**
> (comando `code .` da Ubuntu), così editi in Windows ma esegui in Linux.

## D. SynthSeg (via FreeSurfer ≥ 7.3)

1. Registrati (gratis) per la **licenza** FreeSurfer:
   https://surfer.nmr.mgh.harvard.edu/registration.html → ricevi un `license.txt`.
2. Installa FreeSurfer per **linux/Ubuntu** dalla pagina ufficiale:
   https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
3. Metti la licenza dove FreeSurfer se la aspetta (di norma
   `$FREESURFER_HOME/license.txt` o il percorso indicato da `FS_LICENSE`).
4. Test:
   ```bash
   which mri_synthseg      # deve stampare un percorso
   ```
   Quando `mri_synthseg` è sul PATH, il metodo **SynthSeg** dell'app funziona.
   (`ui/run_ui.sh` carica FreeSurfer da solo se è in `/usr/local/freesurfer/<versione>`.)

## E. Avviare l'app

```bash
cd /mnt/c/Users/<utente>/<cartella-del-progetto>
bash ui/run_ui.sh
```
Apri **http://localhost:8501** nel browser di Windows. WSL2 inoltra `localhost`
automaticamente.

---

## Riepilogo rapido (dopo la prima installazione)

```bash
cd /mnt/c/Users/<utente>/<cartella-del-progetto>
bash ui/run_ui.sh        # -> http://localhost:8501
```

## Problemi comuni
- **`nvidia-smi` non trovato** → aggiorna il driver NVIDIA di *Windows*; non
  installare driver dentro Ubuntu.
- **`mri_synthseg: command not found`** → FreeSurfer non caricato: verifica che
  sia installato e sul PATH.
- **Lentezza su `/mnt/c`** → l'I/O sui file Windows da WSL è più lento; per
  esperimenti pesanti puoi copiare i dati nella home di Ubuntu (`~/`).
