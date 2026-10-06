"""VOI e Centiloide dalla parcellazione FreeSurfer (recon-all) dei soggetti DLBS.

Per quasi tutti i soggetti del DLBS e' disponibile l'elaborazione completa di
FreeSurfer (cartelle `sub-XXXX_ses-Y/mri/aparc.DKTatlas+aseg.mgz`). Nella tesi
FreeSurfer non e' uno dei metodi in gara: fa da riferimento ANATOMICO, contro
cui si misura la sovrapposizione (Dice) delle VOI degli altri metodi
(`scripts/analisi/dice_vs_freesurfer.py`).

Per ogni soggetto la parcellazione viene portata sulla griglia della MRI
allineata e passa per gli stessi moduli 05 e 06 degli altri metodi, con la
stessa PET allineata (cache ANTs del DLBS). L'uscita ha lo stesso formato:

    <output>/<soggetto>/recon_all/result.csv, segmentation/, voi/
    <output>/recon_all_gt_summary.csv

La sottocartella si chiama `recon_all` dal nome del comando di FreeSurfer.

    python -m scripts.esecuzione.riferimento_freesurfer_dlbs \\
        --fs-root ../freesurfer --cache-dir data/aligned_dlbs_ants \\
        --output data/results_dlbs
"""
from __future__ import annotations
import argparse, importlib, os, sys, tempfile, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import SimpleITK as sitk          # noqa: E402
import nibabel as nib             # noqa: E402
import pandas as pd               # noqa: E402

_voi = importlib.import_module("05_VOIExtraction")
_cl = importlib.import_module("06_CentiloidCalculation")

# parcellazioni possibili, DKT per prima (e' quella del target Centiloid)
SEG_NAMES = ("aparc.DKTatlas+aseg.mgz", "aparc+aseg.mgz", "aparc.a2009s+aseg.mgz")


def parse_args():
    """Opzioni da riga di comando; --offset e --limit dividono il lotto."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fs-root", default=str(ROOT.parent / "freesurfer"),
                   help="Cartella con le elaborazioni FreeSurfer "
                        "(sub-XXXX_ses-Y/mri/aparc*+aseg.mgz).")
    p.add_argument("--cache-dir", default="data/aligned_dlbs_ants",
                   help="Immagini allineate con ANTs (Realigned_Anatomical/Functional).")
    p.add_argument("--output", default="data/results_dlbs")
    p.add_argument("--tracer", default="FBP")
    p.add_argument("--method-name", default="recon_all")
    p.add_argument("--limit", type=int, default=0, help="Al piu' N soggetti (0 = tutti).")
    p.add_argument("--offset", type=int, default=0, help="Salta i primi N soggetti.")
    return p.parse_args()


def _mgz_to_sitk(mgz_path, tmp):
    """Legge un .mgz conservando la matrice affine (passando da un NIfTI)."""
    # SimpleITK non legge il formato .mgz: lo si riscrive come NIfTI con
    # nibabel, mantenendo affine e intestazione.
    nb = nib.load(mgz_path)
    nii = os.path.join(tmp, "seg.nii.gz")
    nib.save(nib.Nifti1Image(nb.get_fdata().astype("int16"), nb.affine, nb.header), nii)
    return sitk.ReadImage(nii, sitk.sitkInt16)


def _resample_to(ref, img):
    """Etichette sulla griglia di `ref`, con il vicino piu' prossimo."""
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(ref)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetDefaultPixelValue(0)
    return rs.Execute(img)


def trova_parcellazione(fs_subj_dir: Path):
    """(percorso, nome) della prima parcellazione di SEG_NAMES presente."""
    mri = fs_subj_dir / "mri"
    for name in SEG_NAMES:
        if (mri / name).exists():
            return mri / name, name
    return None, None


def elabora_soggetto(sid, fs_dir, cache_dir, out_root, tracer, method):
    """Moduli 05-06 sulla parcellazione FreeSurfer di un soggetto DLBS.

    sid        identificativo ('sub-XXXX')
    fs_dir     cartella FreeSurfer del soggetto (sub-XXXX_ses-Y)
    cache_dir  cache delle immagini allineate con ANTs
    out_root   cartella dei risultati
    tracer     tracciante (FBP per il DLBS)
    method     nome della sottocartella del metodo ('recon_all')

    Restituisce (record, None) se riesce, (None, motivo) se mancano i dati.
    """
    # 1. Ingressi: MRI e PET allineate dalla cache, parcellazione FreeSurfer.
    anat = Path(cache_dir) / sid / "Realigned_Anatomical" / "Realigned_Anatomical.nii.gz"
    pet = Path(cache_dir) / sid / "Realigned_Functional" / "Realigned_Functional.nii.gz"
    if not anat.exists() or not pet.exists():
        return None, f"manca la cache allineata di {sid}"
    seg_path, seg_name = trova_parcellazione(fs_dir)
    if seg_path is None:
        return None, f"nessuna aparc+aseg in {fs_dir}"

    # 2. Parcellazione sulla griglia della MRI allineata (la stessa di PET e
    #    VOI degli altri metodi). Il NIfTI temporaneo serve solo alla lettura.
    mri_sitk = sitk.ReadImage(str(anat))
    pet_sitk = sitk.ReadImage(str(pet))
    with tempfile.TemporaryDirectory() as tmp:
        seg = _mgz_to_sitk(str(seg_path), tmp)
    seg = _resample_to(mri_sitk, seg)

    # 3. Stesse VOI e stesso calcolo degli altri metodi a etichette FreeSurfer.
    voi_ctx, voi_wc, voi_info = _voi.extract_vois(seg, "freesurfer")
    ctx_mean, wc_mean, suvr, cl, _ = _cl.compute_centiloid(pet_sitk, voi_ctx, voi_wc, tracer)

    # 4. Salvataggio con lo stesso formato di result.csv della pipeline.
    subj_dir = out_root / sid / method
    (subj_dir / "voi").mkdir(parents=True, exist_ok=True)
    (subj_dir / "segmentation").mkdir(parents=True, exist_ok=True)
    seg_out = str(subj_dir / "segmentation" / f"{method}_seg.nii.gz")
    sitk.WriteImage(seg, seg_out)
    ctx_p, wc_p = _voi.save_vois(voi_ctx, voi_wc, str(subj_dir / "voi"),
                                 prefix=f"{method}_")

    rec = {"subject_id": sid, "dataset": "dlbs", "tracer": tracer,
           "segmenter": method, "ctx_mean": ctx_mean, "wc_mean": wc_mean,
           # Nessun tempo di segmentazione: recon-all e' stato eseguito altrove.
           "suvr": suvr, "centiloid": cl, "seg_runtime_s": "",
           "anat_path": str(anat), "voi_ctx_path": ctx_p, "voi_wc_path": wc_p,
           "seg_path": seg_out,
           "voi_extraction_method": voi_info.get("method"),
           "ctx_voxels": voi_info.get("ctx_voxels"),
           "wc_voxels": voi_info.get("wc_voxels"),
           "seg_file": seg_name}
    pd.DataFrame([rec]).to_csv(subj_dir / "result.csv", index=False)
    return rec, None


def main():
    """Elabora tutti i soggetti FreeSurfer e scrive il riepilogo."""
    args = parse_args()
    fs_root, out_root = Path(args.fs_root), Path(args.output)
    fs_dirs = sorted(d for d in fs_root.glob("sub-*") if d.is_dir())
    if args.offset:
        fs_dirs = fs_dirs[args.offset:]
    if args.limit:
        fs_dirs = fs_dirs[:args.limit]

    righe, falliti = [], []
    for i, fs_dir in enumerate(fs_dirs, 1):
        sid = fs_dir.name.split("_ses-")[0]          # sub-0169_ses-2 -> sub-0169
        # Soggetti gia' elaborati saltati: un lotto interrotto riparte da solo.
        if (out_root / sid / args.method_name / "result.csv").exists():
            continue
        try:
            rec, err = elabora_soggetto(sid, fs_dir, args.cache_dir, out_root,
                                        args.tracer, args.method_name)
            if rec is None:
                falliti.append((sid, err))
                print(f"[{i}/{len(fs_dirs)}] SALTO {sid}: {err}")
            else:
                righe.append(rec)
                print(f"[{i}/{len(fs_dirs)}] {sid}  SUVr={rec['suvr']:.3f}  "
                      f"CL={rec['centiloid']:.1f}")
        except Exception as e:
            falliti.append((sid, str(e)))
            print(f"[{i}/{len(fs_dirs)}] ERRORE {sid}: {e}")
            traceback.print_exc()

    if righe:
        df = pd.DataFrame(righe)
        out_root.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_root / "recon_all_gt_summary.csv", index=False)
        print(f"\nFatto: {len(righe)} soggetti, {len(falliti)} non riusciti.")
    if falliti:
        print("Non riusciti:", falliti[:10])


if __name__ == "__main__":
    main()
