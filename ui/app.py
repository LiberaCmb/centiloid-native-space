"""Interfaccia grafica (Streamlit) per il calcolo del Centiloide.

Avvio (in WSL2, dove gira FreeSurfer):
    bash ui/run_ui.sh          # oppure: streamlit run ui/app.py

Flusso per ogni paziente:
    1. Inserire MRI e PET (NIfTI o cartella DICOM -> modulo 00 converte)
    2. Scegliere il backend di allineamento
    3. Scegliere la cartella di output
    4. Scegliere le regioni: standard (-> SUVr e Centiloide) oppure libere
       (-> solo SUVr)
    5. Avviare: allineamento ANTs -> segmentazione -> VOI -> calcolo
    6. Anteprima delle VOI sovrapposte a MRI e PET

DUE COSE DELIBERATAMENTE ASSENTI

Il Centiloide non viene mostrato quando le regioni scelte non sono quelle su
cui la scala e' tarata: la regola sta in pipeline_service.centiloide_applicabile
e l'interfaccia si limita a rispettarla. Il SUVr invece si calcola sempre,
perche' e' un rapporto fra medie e resta definito qualunque coppia di regioni
si scelga.

La dilatazione del target non e' esposta. Resta nella pipeline
(`ctx_dilation_iters` di run_single) ed e' discussa in tesi, ma dall'analisi non
e' emerso un miglioramento sistematico, e nell'interfaccia sarebbe una manopola
che cambia il risultato senza che l'utente possa sapere se in meglio.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import streamlit as st

# La pagina si avvia con `streamlit run ui/app.py`: la radice del progetto va
# messa nel path a mano perche' `shared` e `ui` siano importabili.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import percorsi
from ui import freesurfer_labels as fsl
from ui.pipeline_service import (
    run_single,
    centiloide_applicabile,
    SUPPORTED_SEGMENTERS,
    SUPPORTED_TRACERS,
)

# Gruppi di regioni salvati dall'utente (scheda "Gruppi salvati").
GROUPS_FILE = ROOT / "ui" / "label_groups.json"
DEFAULT_OUTPUT_DIR = percorsi.dati("ui_results")

st.set_page_config(page_title="Centiloide — Segmentazione nativa",
                   page_icon="🧠", layout="wide")


# ════════════════════════════════════════════════════════════════════════════
# Persistenza gruppi di label salvati
# ════════════════════════════════════════════════════════════════════════════
def load_groups() -> dict:
    """Gruppi salvati in label_groups.json ({} se assente o illeggibile)."""
    if GROUPS_FILE.exists():
        try:
            return json.loads(GROUPS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_groups(groups: dict) -> None:
    """Riscrive per intero label_groups.json con i gruppi passati."""
    GROUPS_FILE.parent.mkdir(parents=True, exist_ok=True)
    GROUPS_FILE.write_text(json.dumps(groups, indent=2, ensure_ascii=False),
                           encoding="utf-8")


# ════════════════════════════════════════════════════════════════════════════
# Stato delle checkbox (una casella per regione, per ciascuna scheda)
# ════════════════════════════════════════════════════════════════════════════
# prefix "tgt" = target corticale, "ref" = reference
# Ogni casella ha una chiave propria in st.session_state: cosi' i pulsanti
# (preset, deseleziona, carica gruppo) possono spuntarle via codice e la
# selezione sopravvive ai ridisegni della pagina.
def _key(prefix: str, region: "fsl.Region") -> str:
    """Chiave di session_state della casella di `region` nella scheda `prefix`."""
    return f"{prefix}_{region.key}"


def set_standard_preset() -> None:
    """Spunta il target/reference del Centiloide standard."""
    for r in fsl.REGIONS:
        st.session_state[_key("tgt", r)] = r.in_std_ctx
        st.session_state[_key("ref", r)] = r.in_std_wc


def clear_prefix(prefix: str) -> None:
    """Toglie la spunta a tutte le caselle di una scheda."""
    for r in fsl.REGIONS:
        st.session_state[_key(prefix, r)] = False


def clear_all() -> None:
    """Toglie la spunta a target e riferimento."""
    clear_prefix("tgt")
    clear_prefix("ref")


def apply_ids_to_prefix(prefix: str, ids) -> None:
    """Spunta le caselle di una scheda a partire da una lista di etichette.

    Una regione risulta spuntata se almeno una delle sue etichette (sx o dx)
    compare in `ids`.
    """
    ids = set(ids)
    for r in fsl.REGIONS:
        st.session_state[_key(prefix, r)] = any(i in ids for i in r.ids)


def collect_ids(prefix: str) -> list[int]:
    """Etichette FreeSurfer (ordinate, senza duplicati) spuntate in una scheda."""
    out: list[int] = []
    for r in fsl.REGIONS:
        if st.session_state.get(_key(prefix, r), False):
            out.extend(r.ids)
    return sorted(set(out))


# Inizializza lo stato al primo avvio -> parte dal preset standard.
# Streamlit riesegue l'intero script a ogni interazione: il flag evita che il
# preset sovrascriva a ogni ridisegno le scelte fatte dall'utente.
if "initialised" not in st.session_state:
    set_standard_preset()
    st.session_state["initialised"] = True


def render_region_checklist(prefix: str, disabled: bool, mark_attr: str) -> None:
    """Disegna la checklist completa (per categoria) per una scheda.

    mark_attr: 'in_std_ctx' o 'in_std_wc' -> quali regioni marcare con ★.
    """
    by_cat = fsl.regions_by_category()
    for cat in fsl.CATEGORY_ORDER:
        regs = by_cat.get(cat, [])
        if not regs:
            continue
        n_star = sum(1 for r in regs if getattr(r, mark_attr))
        header = f"{cat}  ({len(regs)})" + (f"  · {n_star}★" if n_star else "")
        # Aperte di default solo le categorie con regioni standard, e solo se
        # la scheda e' modificabile.
        with st.expander(header, expanded=(n_star > 0 and not disabled)):
            for r in regs:
                star = "★ " if getattr(r, mark_attr) else ""
                st.checkbox(star + r.label_for_ui, key=_key(prefix, r),
                            disabled=disabled)


def _slug(nome: str | None) -> str:
    """Rende un ID paziente utilizzabile come nome di cartella."""
    if not nome:
        return ""
    pulito = re.sub(r"[^\w.\-]+", "_", nome.strip())
    return pulito.strip("._-")


def resolve_input(label: str, key_prefix: str):
    """Percorso su disco (NIfTI o cartella DICOM) o upload di un NIfTI."""
    mode = st.radio(f"{label} — come fornire il dato",
                    ["Percorso su disco (NIfTI o cartella DICOM)",
                     "Carica file NIfTI"],
                    key=f"{key_prefix}_mode", horizontal=True)
    if mode.startswith("Percorso"):
        path = st.text_input(f"Percorso {label}", key=f"{key_prefix}_path",
                             placeholder=r"C:\...\soggetto_MR.nii.gz  "
                                         r"oppure  C:\...\cartella_DICOM")
        return path.strip().strip('"') or None
    uploaded = st.file_uploader(f"File {label} (.nii / .nii.gz)",
                                type=["nii", "gz"], key=f"{key_prefix}_up")
    if uploaded is not None:
        tmpdir = Path(tempfile.gettempdir()) / "centiloid_ui_uploads"
        tmpdir.mkdir(parents=True, exist_ok=True)
        dest = tmpdir / uploaded.name
        dest.write_bytes(uploaded.getbuffer())
        return str(dest)
    return None


# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
st.title("Calcolo del Centiloide — segmentazione in spazio nativo")
st.caption("Conversione DICOM→NIfTI quando serve · allineamento PET→MRI con "
           "ANTsPy · segmentazione SynthSeg · regioni e calcolo nello spazio "
           "nativo del soggetto.")

col_left, col_right = st.columns([1, 1], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# COLONNA SINISTRA — dati paziente + metodo + opzioni
# ─────────────────────────────────────────────────────────────────────────────
with col_left:
    st.header("1 · Dati del paziente")
    subject_id = st.text_input("ID paziente (facoltativo)", key="subject_id",
                               placeholder="es. AD03")
    mri_path = resolve_input("MRI (T1)", "mri")
    pet_path = resolve_input("PET amiloide", "pet")

    tracer = st.selectbox("Tracciante PET", SUPPORTED_TRACERS, index=0,
                          help="Determina i coefficienti della ricalibrazione "
                               "in spazio nativo, CL = a·SUVr + b.")
    co_acquired = st.checkbox("PET e MRI già co-registrate (salta allineamento)",
                              value=False)

    st.header("2 · Segmentazione e allineamento")
    segmenter = SUPPORTED_SEGMENTERS[0]  # solo SynthSeg
    st.info("Segmentazione: **SynthSeg** (segmentazione cerebrale con deep "
            "learning, ~1–2 min).")
    registration = st.selectbox(
        "Backend di allineamento PET→MRI", ["ants", "sitk"], index=0,
        format_func=lambda s: {"ants": "ANTsPy — MI (consigliato)",
                               "sitk": "SimpleITK — multi-restart MI"}[s])

    st.header("3 · Output")
    # NB: il percorso NON va passato come `value=` di un text_input, perche'
    # Streamlit lo valuta solo al primo disegno della pagina: scrivendo l'ID
    # paziente dopo, il campo restava fermo e ogni run finiva nella stessa
    # cartella sovrascrivendo il precedente. Qui si chiede solo la cartella
    # BASE e il nome del soggetto viene aggiunto a valle, cosi' segue sempre
    # quello che hai scritto sopra.
    base_dir = st.text_input("Cartella base dei risultati",
                             value=str(DEFAULT_OUTPUT_DIR), key="output_base")
    _nome = _slug(subject_id) or "soggetto"
    output_dir = str(Path(base_dir.strip() or str(DEFAULT_OUTPUT_DIR)) / _nome)
    st.caption(f"I risultati finiranno in: `{output_dir}`")
    if Path(output_dir).exists():
        st.warning("Questa cartella esiste già: il run la sovrascriverà. "
                   "Cambia l'ID paziente o la cartella base per conservare "
                   "i risultati precedenti.")


# ─────────────────────────────────────────────────────────────────────────────
# COLONNA DESTRA — selezione label
# ─────────────────────────────────────────────────────────────────────────────
with col_right:
    st.header("4 · Regioni su cui misurare")

    modo = st.radio(
        "Che cosa vuoi ottenere",
        ["Centiloide  ·  regioni standard",
         "Solo SUVr  ·  regioni a scelta"],
        index=0, key="modo_calcolo",
        captions=[
            "Target corticale e cervelletto della specifica Centiloid "
            "(Klunk 2015). È l'unica combinazione per cui la scala Centiloide "
            "è tarata, ed è quella usata in tutta la tesi.",
            "Componi tu target e riferimento fra tutte le regioni di "
            "FreeSurfer. Il rapporto fra le due medie resta un SUVr valido, "
            "ma non è convertibile in Centiloidi."])

    # Da qui in poi tutta la pagina dipende da questa scelta: in modalita'
    # standard le caselle sono di sola lettura e al servizio si passa None.
    use_standard = modo.startswith("Centiloide")

    if use_standard:
        st.caption("Le regioni sono fissate: le schede qui sotto mostrano "
                   "quali sono, ma non si possono modificare.")
    else:
        st.caption("Il **target** è la regione in cui si misura il segnale, il "
                   "**riferimento** è quella con cui lo si normalizza. Sono "
                   "selezionabili tutte le regioni prodotte da SynthSeg, "
                   "quindi la modalità serve anche a ispezionare la "
                   "segmentazione.")

    b1, b2 = st.columns(2)
    with b1:
        st.button("Carica preset standard", on_click=set_standard_preset,
                  use_container_width=True)
    with b2:
        st.button("Deseleziona tutto", on_click=clear_all,
                  use_container_width=True)

    disabled = use_standard

    tab_ctx, tab_ref, tab_groups = st.tabs(
        ["🎯 Target", "⚖️ Riferimento", "💾 Gruppi salvati"])

    with tab_ctx:
        st.caption("Le regioni ★ fanno parte del target Centiloide standard. "
                   "Ogni casella include entrambi gli emisferi (sx+dx).")
        render_region_checklist("tgt", disabled, "in_std_ctx")

    with tab_ref:
        st.caption("Regione con cui si normalizza il segnale. Lo standard usa "
                   "il cervelletto intero, corteccia e sostanza bianca "
                   "insieme. Le regioni ★ sono quelle del riferimento "
                   "standard.")
        render_region_checklist("ref", disabled, "in_std_wc")

    with tab_groups:
        # Gruppi: salvataggio e caricamento di selezioni personali. In
        # modalita' standard salvare/caricare e' disattivato perche' le
        # caselle non sono modificabili.
        groups = load_groups()
        st.subheader("Salva la selezione corrente come gruppo")
        st.caption("Un «gruppo» memorizza con un nome l'insieme di regioni che "
                   "hai spuntato ora (target + riferimento), così puoi "
                   "richiamarlo in futuro senza rifare la selezione. I gruppi "
                   "sono salvati in ui/label_groups.json.")
        gname = st.text_input("Nome del gruppo", key="new_group_name",
                              placeholder="es. Solo frontale+temporale")
        if st.button("💾 Salva gruppo", disabled=disabled):
            ctx_ids = collect_ids("tgt")
            wc_ids = collect_ids("ref")
            if not gname.strip():
                st.warning("Dai un nome al gruppo prima di salvare.")
            elif not ctx_ids:
                st.warning("Il target è vuoto: seleziona almeno una regione.")
            else:
                groups[gname.strip()] = {"ctx": ctx_ids, "wc": wc_ids}
                save_groups(groups)
                st.success(f"Gruppo «{gname.strip()}» salvato.")

        if groups:
            st.subheader("Carica un gruppo salvato")
            chosen = st.selectbox("Gruppi disponibili", list(groups.keys()),
                                  key="load_group_choice")
            cg1, cg2 = st.columns(2)
            with cg1:
                if st.button("📂 Carica", disabled=disabled,
                             use_container_width=True):
                    apply_ids_to_prefix("tgt", groups[chosen].get("ctx", []))
                    apply_ids_to_prefix("ref", groups[chosen].get("wc", []))
                    st.success(f"Gruppo «{chosen}» applicato alle caselle.")
            with cg2:
                if st.button("🗑️ Elimina", use_container_width=True):
                    groups.pop(chosen, None)
                    save_groups(groups)
                    st.info(f"Gruppo «{chosen}» eliminato. Ricarica la pagina.")
        else:
            st.info("Nessun gruppo salvato per ora.")

    # ── Riepilogo: che cosa uscirà da questo run ────────────────────────────
    # Il controllo è lo stesso che poi fa il servizio, chiamato qui in anticipo
    # per dirlo PRIMA di far aspettare tre minuti di segmentazione.
    if use_standard:
        ctx_ids = sorted(fsl.CENTILOID_STANDARD_CTX)
        wc_ids = sorted(fsl.CENTILOID_STANDARD_WC)
    else:
        ctx_ids = collect_ids("tgt")
        wc_ids = collect_ids("ref")

    cl_ok, cl_motivo = centiloide_applicabile(
        None if use_standard else ctx_ids,
        None if use_standard else wc_ids)

    st.write(f"**Selezione attiva** — target: {len(ctx_ids)} etichette · "
             f"riferimento: {len(wc_ids)} etichette")
    if cl_ok:
        st.success("Questa selezione produce **SUVr e Centiloide**: coincide "
                   "con le regioni su cui la scala è tarata.")
    else:
        st.warning(f"Questa selezione produce **solo il SUVr**, perché "
                   f"{cl_motivo}. Il Centiloide non verrà calcolato.")


# ════════════════════════════════════════════════════════════════════════════
# ESECUZIONE
# ════════════════════════════════════════════════════════════════════════════
st.divider()
run = st.button("▶️  Esegui la pipeline e calcola "
                + ("SUVr e Centiloide" if cl_ok else "il SUVr"),
                type="primary", use_container_width=True)

if run:
    # 1. Controlli sugli input prima di avviare minuti di elaborazione.
    if not mri_path:
        st.error("Inserisci l'MRI (percorso o file).")
        st.stop()
    if not pet_path:
        st.error("Inserisci la PET (percorso o file).")
        st.stop()

    # In modalità standard passiamo None -> extract_vois usa le costanti
    # canoniche (riproduce esattamente lo standard). Altrimenti le label scelte.
    ctx_labels = None if use_standard else ctx_ids
    wc_labels = None if use_standard else wc_ids
    if not use_standard and not ctx_labels:
        st.error("Seleziona almeno una regione per il target, oppure torna "
                 "alla modalità «Centiloide · regioni standard».")
        st.stop()
    if not use_standard and not wc_labels:
        st.error("Seleziona almeno una regione per il riferimento: senza "
                 "denominatore il SUVr non è calcolabile.")
        st.stop()

    # 2. Avanzamento: run_single chiama _cb a ogni fase con una frazione 0-1.
    prog = st.progress(0.0, text="Avvio…")
    status = st.empty()

    def _cb(stage: str, frac: float, message: str):
        prog.progress(min(max(frac, 0.0), 1.0), text=message)
        status.write(f"**{stage}** — {message}")

    # 3. Esecuzione della pipeline (moduli 00-06) tramite il servizio.
    with st.spinner("Elaborazione in corso… (la segmentazione può richiedere "
                    "diversi minuti)"):
        res = run_single(
            mri_path=mri_path,
            pet_path=pet_path,
            tracer=tracer,
            segmenter=segmenter,
            output_dir=output_dir,
            registration=registration,
            co_acquired=co_acquired,
            ctx_labels=ctx_labels,
            wc_labels=wc_labels,
            subject_id=subject_id or None,
            progress=_cb,
        )

    # 4. Visualizzazione dei risultati: metriche, anteprima delle VOI,
    #    dettagli del run, avvisi e scaricamento del CSV.
    if res.ok:
        st.success("Pipeline completata.")

        _suvr = f"{res.suvr:.3f}" if res.suvr is not None else "—"
        _ctxm = f"{res.ctx_mean:.3g}" if res.ctx_mean is not None else "—"
        _wcm = f"{res.wc_mean:.3g}" if res.wc_mean is not None else "—"

        if res.centiloid_ok and res.centiloid is not None:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Centiloide (CL)", f"{res.centiloid:.1f}")
            m2.metric("SUVr", _suvr)
            m3.metric("Media target", _ctxm)
            m4.metric("Media riferimento", _wcm)
        else:
            # Niente casella vuota con un trattino al posto del Centiloide:
            # una casella vuota sembra un errore di calcolo, mentre qui il
            # valore non esiste proprio.
            m1, m2, m3 = st.columns(3)
            m1.metric("SUVr", _suvr)
            m2.metric("Media target", _ctxm)
            m3.metric("Media riferimento", _wcm)
            st.info(
                "**Il Centiloide non è stato calcolato**, e non per un "
                f"problema tecnico: {res.centiloid_reason}.\n\n"
                "La scala Centiloide è definita da una retta "
                "CL = a·SUVr + b i cui coefficienti sono stati stimati su una "
                "coppia precisa di regioni, il target corticale della "
                "specifica Centiloid e il cervelletto intero. Su regioni "
                "diverse si misura un SUVr diverso, e quei coefficienti non "
                "gli si applicano: il numero che ne uscirebbe somiglierebbe a "
                "un Centiloide senza esserlo. Il SUVr qui sopra invece è "
                "pienamente valido, ed è confrontabile con altri SUVr "
                "calcolati sulle stesse regioni.")

        # Anteprima VOI su MRI e su PET (immagini PNG prodotte dal servizio)
        _has_mri = res.overlay_mri_path and Path(res.overlay_mri_path).exists()
        _has_pet = res.overlay_pet_path and Path(res.overlay_pet_path).exists()
        if _has_mri or _has_pet:
            st.subheader("Anteprima VOI (coronale · sagittale · assiale)")
            if _has_mri:
                st.image(res.overlay_mri_path, use_container_width=True)
            if _has_pet:
                st.image(res.overlay_pet_path, use_container_width=True)
            st.caption("Target in rosso, riferimento in azzurro. Tagli fissi "
                       "world x=15, y=0, z=0.")

        with st.expander("Dettagli", expanded=True):
            if res.seg_runtime_s:
                st.write(f"- **Metodo segmentazione:** {res.segmenter} "
                         f"(runtime {res.seg_runtime_s:.0f}s)")
            else:
                st.write(f"- **Metodo:** {res.segmenter}")
            st.write(f"- **Tracciante:** {res.tracer}")
            # Presente solo se il Centiloide e' stato calcolato: contiene i
            # coefficienti ricalibrati usati e il CL con la formula pubblicata.
            _rc = (res.info.get("centiloid", {}) or {}).get("recalibration")
            if _rc:
                st.write(f"- **Calibrazione:** ricalibrata SynthSeg (regressione) "
                         f"— CL = {_rc['a']:.4f}·SUVr {_rc['b']:+.4f}")
                st.write(f"- **Provenienza dei coefficienti:** `{_rc.get('source', 'n/d')}`")
                if _rc.get("cl_standard") is not None:
                    st.write(f"- **CL con la formula pubblicata del tracciante "
                             f"(non ricalibrato):** {_rc['cl_standard']:.1f}")
            else:
                st.write("- **Calibrazione:** nessuna, il Centiloide non è "
                         f"applicabile ({res.centiloid_reason})")
            st.write(f"- **Voxel del target:** {res.ctx_voxels} · "
                     f"**voxel del riferimento:** {res.wc_voxels}")
            st.write(f"- **Metodo VOI:** {res.voi_method}")
            st.write(f"- **Allineamento:** {res.registration_message}")
            st.write(f"- **Cartella risultati:** `{res.output_dir}`")

        for w in res.warnings:
            st.warning(w)

        csv_path = Path(res.output_dir) / "result.csv"
        if csv_path.exists():
            st.download_button("⬇️ Scarica result.csv",
                               data=csv_path.read_bytes(),
                               file_name=f"{res.subject_id}_result.csv",
                               mime="text/csv")
    else:
        # Il servizio non solleva eccezioni: l'errore arriva dentro il
        # risultato, con il traceback completo in error_detail.
        st.error(f"Errore durante l'esecuzione: {res.error}")
        with st.expander("Dettaglio tecnico dell'errore"):
            st.code(res.error_detail or "")
