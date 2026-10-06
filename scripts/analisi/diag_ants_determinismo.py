"""Ripetibilita' dell'allineamento ANTs e quanto SUVr varia da un giro all'altro.

Ripete N volte l'allineamento dello stesso soggetto sugli stessi file e
ricalcola ogni volta SUVr e Centiloide. Serve a rispondere a due domande:
l'allineamento e' deterministico? E se non lo e', quanto pesa sul risultato?
Ad ogni giro si calcola anche un'impronta SHA-1 dei voxel e della geometria
delle immagini caricate, per escludere che a cambiare siano gli ingressi e
non l'allineamento.

COME FA
-------
Non serve rifare la segmentazione: le VOI vivono sulla griglia della MRI, che
è l'immagine FISSA della registrazione e quindi non cambia mai. Lo script
riusa le VOI già calcolate, ripete N volte la sola registrazione PET->MR,
ricampiona la PET e ricalcola SUVR e Centiloide. Un giro dura una manciata di
secondi invece dei ~3 minuti di SynthSeg.

USO
---
    python -m scripts.analisi.diag_ants_determinismo                     # 2002, 5 giri
    python -m scripts.analisi.diag_ants_determinismo --ripetizioni 10
    python -m scripts.analisi.diag_ants_determinismo --soggetto 1009 --tracer FBB
    python -m scripts.analisi.diag_ants_determinismo --backend sitk      # confronto

Le quattro modalita' (--modo), misurate sul soggetto 2002 il 9 agosto 2026:

    modo             dev.std SUVR  escurs. CL  ripetibile   tempo/registrazione
    default             0.002496      1.42        no             4.9s
    thread1             0.001097      0.40        no            ~27s
    seme                0.000021      0.008    in pratica        4.9s
    deterministico      0.000000      0.000     al bit          26.9s

Conclusione: l'ingrediente decisivo e' il SEME, non il numero di thread.

Uscita: data/diag_ants_determinismo/<soggetto>/determinismo_<backend>_<modo>.csv
(una riga per giro: medie, SUVr, CL, MI, tempo, impronte, parametri). Le
immagini riallineate di ogni giro si cancellano alla fine, salvo
--conserva-lavoro.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import re
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


def parse_args():
    """Opzioni da riga di comando (vedi la docstring del modulo)."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--soggetto", default="2002")
    p.add_argument("--tracer", default="FBB")
    p.add_argument("--ripetizioni", type=int, default=5)
    p.add_argument("--backend", choices=["ants", "sitk"], default="ants")
    p.add_argument("--modo",
                   choices=["default", "thread1", "seme", "deterministico"],
                   default="default",
                   help="default = come gira la pipeline oggi. "
                        "thread1 = ITK a un solo thread, per isolare il "
                        "multithreading. seme = solo --random-seed, "
                        "multithreading attivo (il compromesso interessante: "
                        "riproducibile senza perdere velocita', se basta). "
                        "deterministico = "
                        "ants.config.set_ants_deterministic(), che mette il "
                        "seme (--random-seed) E un solo thread.")
    p.add_argument("--seme", type=int, default=123)
    p.add_argument("--mri", default=None,
                   help="Default: data/nifti/fbb_project/<sogg>/MR.nii.gz")
    p.add_argument("--pet", default=None,
                   help="Default: data/nifti/fbb_project/<sogg>/PET.nii.gz")
    p.add_argument("--voi-dir", default=None,
                   help="Default: data/results_fbb_ants/<sogg>/synthseg/voi")
    p.add_argument("--tmp", default="data/diag_ants_determinismo")
    p.add_argument("--conserva-lavoro", action="store_true",
                   help="Non cancellare le immagini riallineate di appoggio "
                        "(~40 MB per giro).")
    return p.parse_args()


def main():
    """Ripete la registrazione, stampa la variabilita' e scrive il CSV."""
    args = parse_args()
    s = args.soggetto

    # 1. Configurazione di ITK/ANTs secondo --modo. Va fatto PRIMA della prima
    #    registrazione: ITK legge la variabile quando inizializza il suo pool
    #    di thread.
    if args.modo == "thread1":
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    elif args.modo in ("seme", "deterministico"):
        try:
            import ants
            ants.config.set_ants_deterministic(True, args.seme)
        except AttributeError:
            raise SystemExit(
                "Questa versione di ANTsPy non ha "
                "ants.config.set_ants_deterministic(): usa --modo thread1.")
        if args.modo == "seme":
            # set_ants_deterministic forza anche ITK a un thread: qui lo
            # togliamo per isolare l'effetto del solo seme.
            os.environ.pop("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", None)

    # 2. Ingressi: MRI, PET e le due VOI gia' calcolate (sulla griglia MRI).
    mri = Path(args.mri or risolvi(f"data/nifti/fbb_project/{s}/MR.nii.gz"))
    pet = Path(args.pet or risolvi(f"data/nifti/fbb_project/{s}/PET.nii.gz"))
    voi_dir = Path(args.voi_dir
                   or risolvi(f"data/results_fbb_ants/{s}/synthseg/voi"))
    voi_ctx_p = voi_dir / "synthseg_voi_ctx.nii.gz"
    voi_wc_p = voi_dir / "synthseg_voi_wc.nii.gz"

    for f in (mri, pet, voi_ctx_p, voi_wc_p):
        if not Path(f).exists():
            raise SystemExit(f"File mancante: {f}\n"
                             "Passa i percorsi con --mri/--pet/--voi-dir.")

    # Intestazione con versioni e numero di thread: il risultato dipende
    # dall'ambiente, quindi lo si registra insieme ai numeri.
    print(f"Soggetto {s} · tracciante {args.tracer} · backend {args.backend} "
          f"· modo {args.modo} · {args.ripetizioni} ripetizioni")
    print(f"  MR  : {mri}")
    print(f"  PET : {pet}")
    print(f"  VOI : {voi_dir}")
    nthreads = os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS")
    print(f"  ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS = "
          f"{nthreads if nthreads else '(non impostata)'}")
    print(f"  Interprete: {sys.executable}")
    try:
        import ants
        print(f"  Python {sys.version.split()[0]} · "
              f"ANTsPy {getattr(ants, '__version__', '?')} · "
              f"SimpleITK {sitk.Version_VersionString()} · "
              f"numpy {np.__version__}")
    except ImportError:
        print(f"  Python {sys.version.split()[0]} · "
              f"SimpleITK {sitk.Version_VersionString()}")
    print()

    voi_ctx = sitk.ReadImage(str(voi_ctx_p))
    voi_wc = sitk.ReadImage(str(voi_wc_p))

    tmp = risolvi(args.tmp) / s
    tmp.mkdir(parents=True, exist_ok=True)

    register = _realign.get_registration(args.backend)
    righe = []

    def _impronta(vol):
        """Impronta esatta di un'immagine caricata: voxel + geometria.

        Serve a escludere che la variabilita' osservata nasca a monte della
        registrazione. Se questa impronta e' identica a ogni giro, allora gli
        ingressi di ANTs sono gli stessi bit e l'unica variabile e' ANTs.
        """
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(vol.pixel_data).tobytes())
        h.update(repr((tuple(np.round(vol.spacing, 12)),
                       tuple(np.round(vol.origin, 12)),
                       tuple(np.round(vol.direction, 12)),
                       vol.pixel_data.dtype.str,
                       vol.pixel_data.shape)).encode())
        return h.hexdigest()[:16]

    # 3. Ciclo dei giri: contesto nuovo, caricamento, registrazione,
    #    ricampionamento della PET e centiloide con le stesse VOI.
    for i in range(args.ripetizioni):
        ctx = ProjectContext()
        ctx.subject_id = s
        ctx.selected_tracer = args.tracer
        ctx.output_dir = str(tmp)

        ok, msg = _load.update_context_with_file(ctx, "anat_1", str(mri))
        if not ok:
            raise SystemExit(f"Caricamento MR fallito: {msg}")
        ok, msg = _load.update_context_with_file(ctx, "func_1", str(pet))
        if not ok:
            raise SystemExit(f"Caricamento PET fallito: {msg}")
        ctx.selected_tracer = args.tracer   # il modulo 01 lo sovrascrive

        impronta = (_impronta(ctx.anatomical_1), _impronta(ctx.functional_1))

        t0 = time.perf_counter()
        transform, msg = register(ctx)
        dt = time.perf_counter() - t0
        if transform is None:
            raise SystemExit(f"Registrazione fallita: {msg}")

        # L'informazione mutua finale si legge dal messaggio del modulo 02.
        m = re.search(r"MI\(MR, warped-PET\)=(-?[\d.]+)", msg or "")
        mi = float(m.group(1)) if m else float("nan")

        run_dir = tmp / f"giro_{i:02d}"
        _visual.ImageFusionLogic.save_realigned_images_with_transform(
            ctx, str(run_dir), transform)

        pet_al = load_image(ctx.realigned_func_path)
        ctx_mean, wc_mean, suvr, cl, _info = _cl.compute_centiloid(
            pet_al, voi_ctx, voi_wc, args.tracer)

        try:
            par = tuple(round(float(v), 6) for v in transform.GetParameters())
        except Exception:
            par = ()
        righe.append(dict(giro=i, ctx=ctx_mean, wc=wc_mean, suvr=suvr, cl=cl,
                          par=par, impronta=impronta, mi=mi, sec=dt))
        print(f"  giro {i}: CTX={ctx_mean:12.6f}  WC={wc_mean:12.6f}  "
              f"SUVR={suvr:.6f}  CL={cl:7.3f}  MI={mi:8.4f}  {dt:6.1f}s")
        print(f"           ingressi {impronta[0]}/{impronta[1]}  "
              f"trasformata [{', '.join(f'{v:+.6f}' for v in par[:6])}...]")

    # 4. Sintesi: escursioni di SUVr, CL, MI e tempi; "identici" a 9 decimali
    #    di SUVr, cioe' ben sotto qualunque differenza rilevante.
    suvrs = [r["suvr"] for r in righe]
    cls = [r["cl"] for r in righe]
    identici = len(set(f"{v:.9f}" for v in suvrs)) == 1
    par_identici = len(set(r["par"] for r in righe)) == 1
    ingressi_identici = len(set(r["impronta"] for r in righe)) == 1
    mis = [r["mi"] for r in righe if r["mi"] == r["mi"]]   # scarta i NaN

    print()
    print("─" * 66)
    print(f"SUVR   min={min(suvrs):.6f}  max={max(suvrs):.6f}  "
          f"escursione={max(suvrs) - min(suvrs):.6f}")
    if len(suvrs) > 1:
        print(f"       media={statistics.mean(suvrs):.6f}  "
              f"dev.std={statistics.pstdev(suvrs):.6f}")
    print(f"CL     min={min(cls):.3f}  max={max(cls):.3f}  "
          f"escursione={max(cls) - min(cls):.3f} CL")
    tempi = [r["sec"] for r in righe]
    print(f"Tempo  media={statistics.mean(tempi):.1f}s  "
          f"min={min(tempi):.1f}s  max={max(tempi):.1f}s  (sola registrazione)")
    if mis:
        print(f"MI     min={min(mis):.4f}  max={max(mis):.4f}  "
              f"escursione={max(mis) - min(mis):.4f}  "
              f"(piu' negativa = allineamento migliore)")
    print(f"Parametri della trasformata identici in tutti i giri: "
          f"{'SI' if par_identici else 'NO'}")
    print()
    # 5. Verdetto: prima si escludono differenze negli ingressi, poi si giudica
    #    la ripetibilita' della registrazione.
    print("CONTROLLO SUGLI INGRESSI")
    if ingressi_identici:
        print("  Le immagini date ad ANTs sono IDENTICHE al bit a ogni giro")
        print(f"  (impronta {righe[0]['impronta'][0]}/{righe[0]['impronta'][1]}).")
        print("  Quindi l'unica variabile del ciclo e' la registrazione:")
        print("  qualunque differenza qui sotto viene da li'.")
    else:
        print("  ATTENZIONE: gli ingressi NON sono identici fra i giri!")
        for r in righe:
            print(f"    giro {r['giro']}: {r['impronta'][0]}/{r['impronta'][1]}")
        print("  La variabilita' nasce a monte della registrazione: prima di")
        print("  concludere qualcosa su ANTs, va spiegato questo.")
    print()
    if identici:
        print(f"ESITO: con --modo {args.modo} l'allineamento e' RIPETIBILE.")
    else:
        print("ESITO: l'allineamento NON e' ripetibile con questa configurazione.")
        print(f"       Stessi ingressi, {args.ripetizioni} giri, escursione di "
              f"{max(cls) - min(cls):.2f} CL.")
        print("       Il seme fisso (--modo seme) rende ripetibile il risultato;")
        print("       nella pipeline si attiva con CENTILOID_ANTS_DETERMINISTIC=1.")

    # 6. Tabella per giro e pulizia delle cartelle di lavoro.
    import csv
    out_csv = tmp / f"determinismo_{args.backend}_{args.modo}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["giro", "ctx_mean", "wc_mean", "suvr", "centiloid", "MI",
                    "secondi", "impronta_MR", "impronta_PET", "parametri"])
        for r in righe:
            w.writerow([r["giro"], r["ctx"], r["wc"], r["suvr"], r["cl"],
                        r["mi"], round(r["sec"], 2),
                        r["impronta"][0], r["impronta"][1],
                        " ".join(f"{v:+.8f}" for v in r["par"])])
    if not args.conserva_lavoro:
        for i in range(args.ripetizioni):
            shutil.rmtree(tmp / f"giro_{i:02d}", ignore_errors=True)
    print(f"\nTabella salvata in {out_csv}")


if __name__ == "__main__":
    main()
