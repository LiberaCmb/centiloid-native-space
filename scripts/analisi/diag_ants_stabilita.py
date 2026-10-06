"""Stabilita' della registrazione ANTs su PIU' soggetti — l'analogo di
`data/registration_stability.csv`, che misura la stessa cosa per SimpleITK.

PERCHE'
-------
Il 9 agosto 2026 e' stato accertato che la registrazione ANTs non e'
riproducibile con le impostazioni predefinite (vedi
02_Realignment/registration_ants.py e diag_ants_determinismo.py). La misura pero'
riguardava un solo soggetto con cinque ripetizioni: il FATTO e' certo, la
MISURA no. Due serie da cinque sullo stesso soggetto avevano gia' medie
diverse fra loro, il che dice quanto sia instabile una stima con n cosi'
piccolo.

Questo script produce il numero da citare in tesi: ripete la registrazione su
N soggetti e riassume l'incertezza del Centiloide, per soggetto e complessiva.

COME
----
Per ogni soggetto ripete solo la registrazione PET->MR e ricalcola SUVR e
Centiloide riusando le VOI gia' esistenti. E' lecito perche' le VOI vivono
sulla griglia della MRI, che e' l'immagine FISSA della registrazione: non
cambiano mai. Cosi' non serve rieseguire SynthSeg e un giro costa ~5 secondi
invece di ~3 minuti.

Costo indicativo: soggetti x ripetizioni x 5 s. Con i valori predefiniti
(10 soggetti x 5 ripetizioni) circa 4-5 minuti.

USO
---
    python -m scripts.analisi.diag_ants_stabilita                    # 10 sogg., 5 giri
    python -m scripts.analisi.diag_ants_stabilita --soggetti 20 --ripetizioni 10
    python -m scripts.analisi.diag_ants_stabilita --dataset fmm
    python -m scripts.analisi.diag_ants_stabilita --dataset tutti --soggetti 15
    python -m scripts.analisi.diag_ants_stabilita --soggetti-elenco 1008 1009 2002

Produce, in data/stabilita_ants/ (un gruppo di file per dataset):
    stabilita_FBB_per_giro.csv       una riga per (soggetto, giro)
    stabilita_FBB_per_soggetto.csv   media, dev.std, escursione per soggetto
    stabilita_FBB_riepilogo.txt      il paragrafo pronto da leggere
e, con --dataset tutti, anche:
    stabilita_confronto_traccianti.csv   le tre mediane a confronto
oltre a ambiente.txt (interprete e versioni delle librerie usate).

Confronto onesto: il backend SimpleITK, sulla stessa prova, oscillava di
decine di CL (data/registration_stability.csv). ANTs e' molto piu' stabile —
il punto non e' che sia inaffidabile, ma che l'incertezza non era quantificata.

CORREZIONI DEL 10 AGOSTO 2026
-----------------------------
Le prime esecuzioni su FMM e FBP hanno mostrato tre difetti, tutti corretti qui:

1. La soglia sul SUVR era applicata a ogni dataset, mentre riguarda solo
   l'FBP: su FMM buttava fuori AD_144 (0.841) e YHV_004 (0.816), che nel file
   finale ci sono. Ora vive dentro DATASET, uno per tracciante.
2. Un soggetto senza SUVR calcolabile (181120, valore assente e non basso)
   superava la soglia e produceva nan, che inquinava media ed escursione
   dell'intero dataset. Ora i valori assenti e i nan sono esclusi a monte, e
   un nan che compaia comunque durante le ripetizioni fa saltare il soggetto.
3. I soggetti erano i primi in ordine alfabetico: su FMM i 30 scelti erano
   tutti AD_1xx, senza un solo controllo giovane, perche' gli YHV_ stanno in
   fondo all'elenco. Poiche' la dispersione cresce con il carico di amiloide,
   il numero risultava gonfiato e non confrontabile con l'FBB. La scelta e'
   ora stratificata per gruppo (vedi scegli_soggetti), e la composizione del
   campione viene stampata e salvata insieme ai risultati.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared.percorsi import risolvi  # noqa: E402

from shared import ProjectContext, load_image  # noqa: E402

_load = importlib.import_module("01_LoadImages")
_realign = importlib.import_module("02_Realignment")
_visual = importlib.import_module("03_VisualCheck")
_cl = importlib.import_module("06_CentiloidCalculation")

# Pendenza della ricalibrazione per regressione di SynthSeg (da
# recalibration_final.csv), per esprimere l'incertezza in centiloidi
# ricalibrati, quelli dei risultati. L'intercetta non conta: una deviazione
# standard cambia solo con la pendenza.
SLOPE = {"FBB": 162.6199, "FMM": 131.6780, "FBP": 181.9503}

# `suvr_min` vale SOLO per l'FBP: e' li' che 14 soggetti su 46 hanno PET con
# scala di intensita' anomala, e sono gli stessi che build_final_comparison.py
# tiene fuori dal file finale. Applicare la stessa soglia a FBB e FMM
# escluderebbe soggetti che nei risultati ci sono (su FMM, AD_144 e YHV_004).
#
# `manifest` serve solo a etichettare i gruppi: nell'FBP gli identificativi
# sono numerici e non dicono chi sia giovane e chi anziano, mentre FBB e FMM
# lo scrivono nel nome del soggetto.
DATASET = {
    "fbb": dict(tracer="FBB", nifti="data/nifti/fbb_project",
                results="data/results_fbb_ants", suvr_min=None, manifest=None),
    "fmm": dict(tracer="FMM", nifti="data/nifti/fmm_project",
                results="data/results_fmm_ants", suvr_min=None, manifest=None),
    "fbp": dict(tracer="FBP", nifti="data/nifti_fbp",
                results="data/results_fbp_ants", suvr_min=0.85,
                manifest="data/manifest_fbp_nifti.csv"),
}

# Soglia convenzionale di positivita' all'amiloide sul SUVR, usata sia per
# stratificare il campione sia per scomporre il risultato.
SOGLIA_POSITIVI = 1.2


def parse_args():
    """Opzioni da riga di comando (vedi la docstring del modulo)."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(DATASET) + ["tutti"],
                   default="fbb",
                   help="'tutti' esegue FBB, FMM e FBP di seguito e aggiunge "
                        "una tabella di confronto fra i tre traccianti.")
    p.add_argument("--soggetti", type=int, default=10,
                   help="Quanti soggetti, scelti a copertura della coorte "
                        "(stratificati per gruppo, equispaziati dentro ogni "
                        "gruppo) fra quelli che hanno sia NIfTI sia VOI. "
                        "Default 10.")
    p.add_argument("--soggetti-elenco", nargs="*", default=None,
                   help="Elenco esplicito di ID, al posto di --soggetti.")
    p.add_argument("--ripetizioni", type=int, default=5)
    p.add_argument("--suvr-min", type=float, default=None,
                   help="Soglia di plausibilita' sul SUVR gia' calcolato. Se "
                        "non la imposti vale quella del dataset: 0.85 per FBP "
                        "(gli stessi 14 soggetti che build_final_comparison.py "
                        "tiene fuori), nessuna per FBB e FMM.")
    p.add_argument("--metodo", default="synthseg",
                   help="Da quale metodo prendere le VOI gia' calcolate.")
    p.add_argument("--out", default="data/stabilita_ants")
    p.add_argument("--tmp", default="data/stabilita_ants/_lavoro")
    p.add_argument("--conserva-lavoro", action="store_true",
                   help="Non cancellare le immagini riallineate di appoggio. "
                        "Servono solo per ispezione visiva e pesano parecchio: "
                        "~200 MB per soggetto con 5 ripetizioni.")
    return p.parse_args()


def _suvr_gia_calcolato(res_dir, s, metodo):
    """SUVR del run gia' presente, per filtrare i soggetti non utilizzabili."""
    f = res_dir / s / metodo / "result.csv"
    if not f.is_file():
        return None
    try:
        # l'ultima riga e' quella del run piu' recente
        with open(f, newline="", encoding="utf-8") as fh:
            riga = list(csv.DictReader(fh))[-1]
        return float(riga["suvr"])
    except Exception:
        return None


def gruppi_da_manifest(cfg) -> dict:
    """Mappa soggetto -> 'YC'/'AD' letta dal manifest, dove esiste."""
    rel = cfg.get("manifest")
    if not rel:
        return {}
    f = risolvi(rel)
    if not f.is_file():
        return {}
    mappa = {}
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            g = str(r.get("group", "")).strip().upper()
            mappa[str(r.get("subject_id", "")).strip()] = "YC" if g == "YC" else "AD"
    return mappa


def gruppo(sid: str, mappa: dict = None) -> str:
    """'YC' (controllo giovane) o 'AD'.

    FBB e FMM lo scrivono nell'identificativo ('Y1017', 'YHV_004', 'AD_144');
    l'FBP no — li' i soggetti sono numeri e il gruppo sta nel manifest, che
    arriva qui come `mappa`.
    """
    if mappa:
        g = mappa.get(str(sid).strip())
        if g:
            return g
    x = str(sid).strip().upper()
    if x.startswith("YHV") or x.startswith("YC"):
        return "YC"
    if x.startswith("AD"):
        return "AD"
    if x and x[0] == "Y" and len(x) > 1 and x[1].isdigit():
        return "YC"
    return "AD"


def strato(suvr: float) -> str:
    """Il criterio con cui il campione viene stratificato.

    Non e' il gruppo clinico ma lo stato amiloide misurato, per due motivi.
    E' la variabile da cui la dispersione dipende davvero — la correlazione
    con la deviazione standard e' rho ~ 0.66 — quindi e' quella che il
    campione deve rappresentare. Ed e' disponibile per tutti e tre gli
    archivi, mentre l'etichetta clinica sull'FBP vive solo nel manifest.
    """
    return "positivi" if suvr >= SOGLIA_POSITIVI else "negativi"


def scegli_soggetti(disponibili, quanti):
    """Sceglie `quanti` soggetti COPRENDO tutta la coorte, non i primi in
    ordine alfabetico.

    Serve perche' gli archivi ordinano per identificativo e gli strati
    finiscono separati: su FMM i primi 30 sono tutti 'AD_1xx' e nessun 'YHV_',
    quindi un campione dei primi N misurerebbe solo i pazienti. Poiche' la
    dispersione cresce con il carico di amiloide, il risultato sarebbe
    gonfiato e non confrontabile con gli altri traccianti.

    `disponibili` e' una lista di coppie (soggetto, SUVR di riferimento). La
    scelta e' stratificata per stato amiloide, in proporzione alla coorte, e
    dentro ogni strato prende soggetti equispaziati nell'elenco.
    """
    if quanti >= len(disponibili):
        return [s for s, _ in disponibili]
    # 1. Soggetti divisi per strato e quota proporzionale di ciascuno strato
    #    (almeno 1, mai piu' dei soggetti presenti).
    per_gruppo = {}
    for s, v in disponibili:
        per_gruppo.setdefault(strato(v), []).append(s)
    quote = {}
    for g, elenco in sorted(per_gruppo.items()):
        quote[g] = min(len(elenco),
                       max(1, round(quanti * len(elenco) / len(disponibili))))
    # l'arrotondamento per gruppo puo' sforare o mancare il totale richiesto:
    # aggiusto sul gruppo piu' numeroso, senza mai scendere sotto 1 per gruppo.
    ordine = sorted(quote, key=lambda g: -len(per_gruppo[g]))
    while sum(quote.values()) > quanti:
        for g in ordine:
            if quote[g] > 1:
                quote[g] -= 1
                break
        else:
            break
    while sum(quote.values()) < quanti:
        for g in ordine:
            if quote[g] < len(per_gruppo[g]):
                quote[g] += 1
                break
        else:
            break
    # 2. Dentro ogni strato, soggetti presi a passo costante lungo l'elenco.
    scelti = []
    for g, elenco in sorted(per_gruppo.items()):
        n_g = quote[g]
        passo = len(elenco) / n_g
        scelti += [elenco[min(int(i * passo), len(elenco) - 1)]
                   for i in range(n_g)]
    return sorted(set(scelti))


def soggetti_disponibili(cfg, metodo, suvr_min=None):
    """Soggetti con NIfTI, VOI e un SUVR plausibile.

    Il filtro sul SUVR replica quello di `build_final_comparison.py`: nel
    dataset FBP 14 soggetti su 46 hanno PET con scala di intensita' anomala e
    SUVR fra 0.59 e 0.81, e sono esclusi dal file finale. Misurare la stabilita'
    su soggetti che non entrano nei risultati non avrebbe senso.

    Ritorna (coppie (soggetto, SUVR), elenco_scartati_con_motivo). Il SUVR
    serve poi a stratificare il campione.
    """
    nifti_dir = risolvi(cfg["nifti"])
    res_dir = risolvi(cfg["results"])
    buoni, scartati = [], []
    if not nifti_dir.is_dir() or not res_dir.is_dir():
        return buoni, scartati
    for d in sorted(p for p in nifti_dir.iterdir() if p.is_dir()):
        s = d.name
        voi = res_dir / s / metodo / "voi"
        if not ((d / "MR.nii.gz").exists() and (d / "PET.nii.gz").exists()
                and (voi / f"{metodo}_voi_ctx.nii.gz").exists()
                and (voi / f"{metodo}_voi_wc.nii.gz").exists()):
            continue
        v = _suvr_gia_calcolato(res_dir, s, metodo)
        if v is None or v != v:      # assente o NaN: il run precedente e' fallito
            scartati.append((s, "SUVR non calcolabile nel run di riferimento"))
            continue
        if suvr_min is not None and v < suvr_min:
            scartati.append((s, f"SUVR {v:.3f} < {suvr_min}"))
            continue
        buoni.append((s, v))
    return buoni, scartati


def misura_soggetto(s, cfg, metodo, ripetizioni, tmp_root, register):
    """Ripete la registrazione e ritorna la lista dei risultati per giro.

    s            identificativo del soggetto
    cfg          voce di DATASET (tracciante e cartelle)
    metodo       metodo di cui riusare le VOI
    ripetizioni  numero di giri
    tmp_root     cartella di appoggio per le immagini riallineate
    register     funzione di registrazione del modulo 02
    """
    # Le VOI si leggono una volta sola: sono sulla griglia della MRI e non
    # dipendono dalla registrazione.
    mri = risolvi(cfg["nifti"]) / s / "MR.nii.gz"
    pet = risolvi(cfg["nifti"]) / s / "PET.nii.gz"
    voi_dir = risolvi(cfg["results"]) / s / metodo / "voi"
    voi_ctx = sitk.ReadImage(str(voi_dir / f"{metodo}_voi_ctx.nii.gz"))
    voi_wc = sitk.ReadImage(str(voi_dir / f"{metodo}_voi_wc.nii.gz"))

    righe = []
    for i in range(ripetizioni):
        ctx = ProjectContext()
        ctx.subject_id = s
        ctx.output_dir = str(tmp_root / s)

        ok, msg = _load.update_context_with_file(ctx, "anat_1", str(mri))
        if not ok:
            raise RuntimeError(f"[{s}] caricamento MR fallito: {msg}")
        ok, msg = _load.update_context_with_file(ctx, "func_1", str(pet))
        if not ok:
            raise RuntimeError(f"[{s}] caricamento PET fallito: {msg}")
        ctx.selected_tracer = cfg["tracer"]   # dopo il caricamento, che lo reimposta

        t0 = time.perf_counter()
        transform, _msg = register(ctx)
        dt = time.perf_counter() - t0
        if transform is None:
            raise RuntimeError(f"[{s}] registrazione fallita: {_msg}")

        _visual.ImageFusionLogic.save_realigned_images_with_transform(
            ctx, str(tmp_root / s / f"giro_{i:02d}"), transform)
        pet_al = load_image(ctx.realigned_func_path)
        ctx_mean, wc_mean, suvr, cl, _info = _cl.compute_centiloid(
            pet_al, voi_ctx, voi_wc, cfg["tracer"])
        righe.append(dict(soggetto=s, giro=i, ctx_mean=ctx_mean,
                          wc_mean=wc_mean, suvr=suvr, cl_grezzo=cl, secondi=dt))
    return righe


def esegui_dataset(chiave, args, out_dir):
    """Misura un dataset. Ritorna (righe_per_giro, righe_per_soggetto)."""
    cfg = DATASET[chiave]
    tracer = cfg["tracer"]
    slope = SLOPE[tracer]

    # 1. Soggetti utilizzabili, con la soglia di SUVr del dataset (o quella
    #    data da riga di comando).
    soglia = args.suvr_min if args.suvr_min is not None else cfg.get("suvr_min")
    disponibili, scartati = soggetti_disponibili(cfg, args.metodo, soglia)
    if scartati:
        print(f"[{tracer}] esclusi {len(scartati)} soggetti con SUVR non "
              f"plausibile (come nel file finale):")
        for s_, motivo in scartati:
            print(f"    {s_}: {motivo}")
    if not disponibili:
        print(f"[{tracer}] nessun soggetto utilizzabile in {cfg['nifti']} + "
              f"{cfg['results']}/<sogg>/{args.metodo}/voi — saltato.")
        return [], []

    suvr_rif = dict(disponibili)
    mappa_gruppi = gruppi_da_manifest(cfg)

    # 2. Campione: elenco esplicito oppure scelta stratificata.
    if args.soggetti_elenco:
        elenco = [s for s in args.soggetti_elenco if s in suvr_rif]
        mancanti = set(args.soggetti_elenco) - set(elenco)
        if mancanti:
            print(f"[attenzione] saltati (dati incompleti): "
                  f"{', '.join(sorted(mancanti))}")
    else:
        elenco = scegli_soggetti(disponibili, args.soggetti)

    tmp_root = risolvi(args.tmp) / tracer
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Composizione del campione e stima del tempo (circa 7 s per giro fra
    # registrazione, ricampionamento e calcolo).
    n_pos = sum(1 for s in elenco if suvr_rif[s] >= SOGLIA_POSITIVI)
    n_yc = sum(1 for s in elenco if gruppo(s, mappa_gruppi) == "YC")
    print(f"\n{'=' * 66}")
    print(f"Dataset {tracer} — {len(elenco)} soggetti su {len(disponibili)} "
          f"disponibili x {args.ripetizioni} ripetizioni")
    print(f"Composizione del campione: {len(elenco) - n_pos} amiloide-negativi, "
          f"{n_pos} positivi  ({n_yc} controlli giovani, "
          f"{len(elenco) - n_yc} anziani/pazienti)")
    print(f"VOI del metodo '{args.metodo}'. Incertezza in CL ricalibrati "
          f"(a={slope}). Stima: ~{len(elenco) * args.ripetizioni * 7 / 60:.0f} min")
    print("=" * 66)

    register = _realign.get_registration("ants")
    tutte, per_sogg = [], []

    # 3. Misura soggetto per soggetto; un errore salta il soggetto, non
    #    interrompe il dataset.
    for k, s in enumerate(elenco, 1):
        print(f"[{tracer} {k}/{len(elenco)}] {s} … ", end="", flush=True)
        try:
            righe = misura_soggetto(s, cfg, args.metodo, args.ripetizioni,
                                    tmp_root, register)
        except Exception as e:
            print(f"SALTATO ({e})")
            continue
        suvr = [r["suvr"] for r in righe]
        # Un solo nan basta a mandare a nan la media dell'intero dataset: il
        # soggetto 181120 dell'FBP ha la VOI corticale vuota una volta
        # ricampionata sulla griglia PET, e nel primo run aveva inquinato
        # media ed escursione di tutti gli altri.
        if any(v is None or v != v for v in suvr):
            print("SALTATO (SUVR non calcolabile: VOI vuota sulla griglia PET)")
            if not args.conserva_lavoro:
                shutil.rmtree(tmp_root / s, ignore_errors=True)
            continue
        tutte.extend(righe)
        # Dispersione in SUVr, convertita in CL ricalibrati con la sola pendenza.
        sd = statistics.pstdev(suvr) if len(suvr) > 1 else 0.0
        esc = max(suvr) - min(suvr)
        per_sogg.append(dict(
            tracciante=tracer, soggetto=s, gruppo=gruppo(s, mappa_gruppi),
            n=len(suvr),
            suvr_medio=statistics.mean(suvr), suvr_devstd=sd,
            suvr_escursione=esc, cl_devstd=sd * slope,
            cl_escursione=esc * slope,
            secondi_medi=statistics.mean([r["secondi"] for r in righe])))
        print(f"SUVR {statistics.mean(suvr):.6f}  "
              f"dev.std {sd * slope:.3f} CL  escursione {esc * slope:.3f} CL")
        if not args.conserva_lavoro:
            shutil.rmtree(tmp_root / s, ignore_errors=True)

    if not args.conserva_lavoro:
        shutil.rmtree(tmp_root, ignore_errors=True)
    if not per_sogg:
        return [], []

    # 4. File per giro, per soggetto e riepilogo testuale del dataset.
    for r in tutte:
        r["tracciante"] = tracer
    with open(out_dir / f"stabilita_{tracer}_per_giro.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tutte[0]))
        w.writeheader()
        w.writerows(tutte)
    with open(out_dir / f"stabilita_{tracer}_per_soggetto.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_sogg[0]))
        w.writeheader()
        w.writerows(per_sogg)

    testo = riepilogo(tracer, per_sogg, args, slope)
    (out_dir / f"stabilita_{tracer}_riepilogo.txt").write_text(
        testo, encoding="utf-8")
    print(testo)
    return tutte, per_sogg


def _ranghi(v):
    """Ranghi con media dei pari merito, come vuole Spearman."""
    ordine = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(ordine):
        j = i
        while j + 1 < len(ordine) and v[ordine[j + 1]] == v[ordine[i]]:
            j += 1
        medio = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[ordine[k]] = medio
        i = j + 1
    return r


def _spearman(x, y) -> str:
    """rho di Spearman fra due serie, con p se scipy c'e'.

    E' la correlazione fra i ranghi, non fra i valori: misura se al crescere
    del SUVR cresce anche l'incertezza, senza assumere che il legame sia
    lineare. Serve proprio perche' l'effetto atteso e' monotono ma non
    necessariamente proporzionale.
    """
    n = len(x)
    if n < 3:
        return "— (troppo pochi soggetti)"
    try:
        from scipy import stats
        rho, p = stats.spearmanr(x, y)
        return f"{rho:.3f}  (p = {p:.4g}, n = {n})"
    except Exception:
        # senza scipy: correlazione di Pearson calcolata sui ranghi
        rx, ry = _ranghi(x), _ranghi(y)
        mx, my = statistics.mean(rx), statistics.mean(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = (sum((a - mx) ** 2 for a in rx) *
               sum((b - my) ** 2 for b in ry)) ** 0.5
        rho = num / den if den else float("nan")
        return f"{rho:.3f}  (n = {n})"


def riepilogo(tracer, per_sogg, args, slope):
    """Paragrafo di sintesi di un dataset, stampato e salvato in *_riepilogo.txt.

    per_sogg  righe per soggetto prodotte da `esegui_dataset`
    slope     pendenza usata per esprimere la dispersione in CL ricalibrati
    """
    sd_cl = [r["cl_devstd"] for r in per_sogg]
    esc_cl = [r["cl_escursione"] for r in per_sogg]
    peggiore = max(per_sogg, key=lambda r: r["cl_escursione"])
    migliore = min(per_sogg, key=lambda r: r["cl_escursione"])

    # Il confronto fra traccianti regge solo se il campione e' composto allo
    # stesso modo: la dispersione cresce con il carico di amiloide, quindi un
    # campione di soli pazienti darebbe un numero piu' alto senza che ANTs
    # c'entri nulla. Riporto qui la scomposizione, cosi' il numero e'
    # interpretabile invece che solo confrontabile.
    neg = [r["cl_devstd"] for r in per_sogg
           if r["suvr_medio"] < SOGLIA_POSITIVI]
    pos = [r["cl_devstd"] for r in per_sogg
           if r["suvr_medio"] >= SOGLIA_POSITIVI]
    yc = [r["cl_devstd"] for r in per_sogg if r.get("gruppo") == "YC"]
    ad = [r["cl_devstd"] for r in per_sogg if r.get("gruppo") == "AD"]

    def _riga(etichetta, v):
        if not v:
            return f"      {etichetta:<28} —"
        return f"      {etichetta:<28} {statistics.median(v):.3f} CL   (n={len(v)})"

    scomposizione = "\n".join([
        _riga(f"SUVR < {SOGLIA_POSITIVI} (negativi)", neg),
        _riga(f"SUVR >= {SOGLIA_POSITIVI} (positivi)", pos),
        _riga("controlli giovani", yc),
        _riga("anziani / pazienti", ad),
    ])
    if neg and pos:
        rapporto = statistics.median(pos) / statistics.median(neg)
        scomposizione += f"\n      rapporto positivi/negativi   {rapporto:.1f}x"
    scomposizione += f"\n\n      Spearman rho(SUVR, dev.std) = {_spearman([r['suvr_medio'] for r in per_sogg], sd_cl)}"

    return f"""
Stabilita' della registrazione ANTs — dataset {tracer}

{len(per_sogg)} soggetti, {args.ripetizioni} ripetizioni ciascuno, VOI del
metodo '{args.metodo}'. Incertezza in Centiloidi ricalibrati (a = {slope}).

  deviazione standard per soggetto
      mediana   {statistics.median(sd_cl):.3f} CL
      media     {statistics.mean(sd_cl):.3f} CL
      minimo    {min(sd_cl):.3f} CL
      massimo   {max(sd_cl):.3f} CL

  escursione (max - min) per soggetto
      mediana   {statistics.median(esc_cl):.3f} CL
      media     {statistics.mean(esc_cl):.3f} CL
      massimo   {max(esc_cl):.3f} CL

  deviazione standard mediana, per sottogruppo
{scomposizione}

  soggetto piu' stabile    {migliore['soggetto']}  ({migliore['cl_escursione']:.3f} CL)
  soggetto meno stabile    {peggiore['soggetto']}  ({peggiore['cl_escursione']:.3f} CL)

  tempo medio per registrazione: {statistics.mean([r['secondi_medi'] for r in per_sogg]):.1f} s
"""


NOTA_FINALE = """
COME LEGGERLO
La deviazione standard mediana e' l'incertezza tipica che la sola registrazione
introduce sul Centiloide di un soggetto, a parita' di tutto il resto. Va
confrontata con l'errore medio del metodo contro il ground truth GAAIN (MAE),
che per synthseg e' 3.40 CL su FBB, 3.97 su FMM e 6.62 su FBP: dice quale
frazione dell'errore totale sia dovuta all'indeterminazione dell'allineamento
e non alla segmentazione.

Attenzione: la variabilita' NON significa che alcune registrazioni siano
sbagliate. La mutua informazione le giudica equivalenti: sono ottimi
indistinguibili per la metrica, e la differenza nasce dal ricampionamento
sotto-voxel della PET. Per questo la ricetta dei restart usata dal backend
SimpleITK qui non servirebbe, mentre il seme fisso azzera il problema
(CENTILOID_ANTS_DETERMINISTIC=1).
"""


def banner_ambiente() -> str:
    """Quale interprete e quali librerie stanno davvero girando.

    Serve perche' sulla stessa macchina convivono piu' ambienti (conda
    'centiloid' e .venv-wsl) e non e' raro credere di essere in uno mentre si
    sta eseguendo l'altro: basta avere entrambi attivi e il PATH decide da
    solo. Stampandolo qui, ogni risultato porta con se' la prova di dove e'
    stato prodotto.
    """
    righe = [f"Interprete : {sys.executable}",
             f"Python     : {sys.version.split()[0]}",
             f"SimpleITK  : {sitk.Version_VersionString()}",
             f"numpy      : {np.__version__}"]
    try:
        import ants
        righe.append(f"ANTsPy     : {getattr(ants, '__version__', '?')}")
    except ImportError:
        righe.append("ANTsPy     : non disponibile")
    return "\n".join("  " + r for r in righe)


def main():
    """Esegue uno o tutti i dataset e, se ce n'e' piu' d'uno, il confronto."""
    args = parse_args()
    out_dir = risolvi(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ambiente di esecuzione, stampato e salvato accanto ai risultati.
    amb = banner_ambiente()
    print("AMBIENTE DI ESECUZIONE")
    print(amb)
    (out_dir / "ambiente.txt").write_text(
        "Ambiente in cui sono state prodotte queste misure\n\n" + amb + "\n",
        encoding="utf-8")

    chiavi = sorted(DATASET) if args.dataset == "tutti" else [args.dataset]
    if args.dataset == "tutti" and args.soggetti_elenco:
        raise SystemExit("--soggetti-elenco vale per un dataset solo.")

    # 2. Misura dei dataset richiesti.
    raccolta = {}
    for chiave in chiavi:
        _tutte, per_sogg = esegui_dataset(chiave, args, out_dir)
        if per_sogg:
            raccolta[DATASET[chiave]["tracer"]] = per_sogg

    if not raccolta:
        raise SystemExit("Nessun dataset elaborato.")

    # 3. La tabella di confronto si ricostruisce da TUTTI i per-soggetto presenti
    # nella cartella, non solo da quelli appena calcolati: cosi' si possono
    # lanciare i dataset in momenti diversi senza perdere quelli gia' fatti.
    presenti = {}
    for f in sorted(out_dir.glob("stabilita_*_per_soggetto.csv")):
        tr = f.name.split("_")[1]
        with open(f, newline="", encoding="utf-8") as fh:
            righe_f = [r for r in csv.DictReader(fh)]
        if righe_f:
            presenti[tr] = righe_f

    # 4. Confronto fra traccianti: mediane della dispersione, complessive e
    #    separate per negativi e positivi, scritte in CSV e stampate.
    if len(presenti) > 1:
        righe = []
        for tr, per_sogg in presenti.items():
            # I file scritti prima del 10 agosto non hanno la colonna
            # 'gruppo' e possono contenere righe a nan: li leggo lo stesso,
            # ricavando il gruppo dall'identificativo e scartando i nan.
            per_sogg = [r for r in per_sogg
                        if float(r["cl_devstd"]) == float(r["cl_devstd"])]
            if not per_sogg:
                continue
            sd = [float(r["cl_devstd"]) for r in per_sogg]
            esc = [float(r["cl_escursione"]) for r in per_sogg]
            suvr = [float(r["suvr_medio"]) for r in per_sogg]
            pos = [d_ for d_, v in zip(sd, suvr) if v >= SOGLIA_POSITIVI]
            neg = [d_ for d_, v in zip(sd, suvr) if v < SOGLIA_POSITIVI]
            n_yc = sum(1 for r in per_sogg
                       if (r.get("gruppo") or gruppo(r["soggetto"])) == "YC")
            righe.append(dict(
                tracciante=tr, n_soggetti=len(per_sogg),
                n_controlli_giovani=n_yc,
                n_anziani_pazienti=len(per_sogg) - n_yc,
                cl_devstd_mediana=round(statistics.median(sd), 3),
                cl_devstd_media=round(statistics.mean(sd), 3),
                cl_devstd_max=round(max(sd), 3),
                cl_escursione_mediana=round(statistics.median(esc), 3),
                cl_escursione_max=round(max(esc), 3),
                mediana_suvr_sotto_1_2=(round(statistics.median(neg), 3)
                                        if neg else ""),
                mediana_suvr_sopra_1_2=(round(statistics.median(pos), 3)
                                        if pos else ""),
                rapporto_pos_neg=(round(statistics.median(pos) /
                                        statistics.median(neg), 2)
                                  if neg and pos else "")))
        righe.sort(key=lambda r: r["tracciante"])
        with open(out_dir / "stabilita_confronto_traccianti.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(righe[0]))
            w.writeheader()
            w.writerows(righe)
        print("\n" + "=" * 78)
        print("CONFRONTO FRA TRACCIANTI — dev.std del Centiloide per soggetto (CL)")
        print("=" * 78)
        print(f"  {'':4s}  {'n':>3s}  {'gio/anz':>7s}  {'mediana':>8s}  "
              f"{'media':>7s}  {'max':>7s}   {'SUVR<1.2':>9s}  "
              f"{'SUVR>=1.2':>9s}  {'rapp.':>6s}")
        for r in righe:
            comp = f"{r['n_controlli_giovani']}/{r['n_anziani_pazienti']}"
            rapp = f"{r['rapporto_pos_neg']}x" if r['rapporto_pos_neg'] else "—"
            print(f"  {r['tracciante']:4s}  {r['n_soggetti']:3d}  {comp:>7s}  "
                  f"{r['cl_devstd_mediana']:8.3f}  {r['cl_devstd_media']:7.3f}  "
                  f"{r['cl_devstd_max']:7.3f}   "
                  f"{str(r['mediana_suvr_sotto_1_2']):>9s}  "
                  f"{str(r['mediana_suvr_sopra_1_2']):>9s}  {rapp:>6s}")
        print("\n  Le due colonne 'SUVR' separano amiloide-negativi e positivi,")
        print("  l'ultima ne fa il rapporto. Se il rapporto si ripete su tutti")
        print("  e tre i traccianti e' un effetto del metodo e non una")
        print("  particolarita' di un dataset.")
        print("  La colonna 'gio/anz' dice come e' composto il campione: la")
        print("  mediana complessiva va confrontata fra traccianti solo se la")
        print("  composizione e' simile, altrimenti si confrontano le colonne")
        print("  separate, che non dipendono da come e' fatto il campione.")

    print(NOTA_FINALE)
    print(f"Scritto in {Path(args.out)}/")


if __name__ == "__main__":
    main()
