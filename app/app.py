import os
import hashlib
import io
import textwrap
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from openai import OpenAI
import json

from db import init_db, add_dataset, list_datasets, get_dataset, delete_dataset
from utils import (
    infer_period_from_filename,
    detect_domains,
    money_to_float,
    to_numeric_series,
    percent_to_float,
    normalize_position,
    extract_section,
    ctr_model,
    bucket_position,
    validate_expected_columns,
    humanize_df,
    build_scatter_tooltips,
    build_column_config,
    to_number,
    coerce_numeric_for_display,
)

st.set_page_config(
    page_title="SEO Executive Dashboard (Senior)",
    page_icon="📈",
    layout="wide",
)

# --- UI polish (executive dark) ---
st.markdown(
    """
<style>
/* Layout */
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.06); }
h1, h2, h3 { letter-spacing: -0.02em; }

/* KPI cards */
div[data-testid="stMetric"] {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px;
  padding: 12px 14px;
}
div[data-testid="stMetric"] label { opacity: 0.85; }
div[data-testid="stMetric"] div { font-weight: 650; }

/* Tabs */
button[data-baseweb="tab"] { font-size: 0.95rem; padding: 10px 14px; }
div[data-baseweb="tab-list"] { gap: 6px; }

/* Dataframe */
div[data-testid="stDataFrame"] { border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; overflow: hidden; }
</style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_DIR = DATA_DIR / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)


def make_unique_columns(cols: list[str]) -> list[str]:
    """Ensure unique column names for Arrow/Streamlit."""
    seen = {}
    out = []
    for c in cols:
        c = str(c)
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
    return out

def safe_dataframe(df: pd.DataFrame, **kwargs):
    """Render dataframe safely even if duplicate cols appear."""
    if df is None:
        return None
    # Sensible defaults for exec dashboards.
    kwargs.setdefault("use_container_width", True)
    kwargs.setdefault("hide_index", True)
    # Streamlit/Arrow fails with duplicated column names; make them unique.
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if isinstance(df, pd.DataFrame) and df.columns.duplicated().any():
        df = df.copy()
        df.columns = make_unique_columns(df.columns.tolist())
    return st.dataframe(df, **kwargs)


def text_to_csv_bytes(text: str) -> bytes:
    """CSV simple: una fila por línea del informe."""
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    out = io.StringIO()
    out.write("line\n")
    for l in lines:
        out.write('"' + l.replace('"', '""') + '"\n')
    return out.getvalue().encode("utf-8")


def text_to_pdf_bytes(title: str, text: str) -> bytes:
    """Genera un PDF simple (A4) desde texto."""
    # Import local para no romper si falta en entorno (pero lo incluimos en requirements).
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    x = 40
    y = height - 50

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title[:120])
    y -= 26

    c.setFont("Helvetica", 10)
    for line in (text or "").splitlines():
        wrapped_lines = textwrap.wrap(line, width=110) or [""]
        for w in wrapped_lines:
            if y < 50:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - 50
            c.drawString(x, y, w)
            y -= 14

    c.save()
    buf.seek(0)
    return buf.read()


def df_to_csv_snippet(df: pd.DataFrame, max_rows: int = 40, max_cell: int = 180) -> str:
    """Convierte un DF a CSV compacto para enviarlo al LLM (sin depender de tabulate)."""
    if df is None:
        return ""
    d = df.head(max_rows).copy()
    d = d.loc[:, ~d.columns.duplicated()]
    for c in d.columns:
        d[c] = d[c].astype(str).str.slice(0, max_cell)
    return d.to_csv(index=False)


DEFAULT_PRIMARY = os.getenv("PRIMARY_DOMAIN", "")
DEFAULT_NON_RANK = int(os.getenv("NON_RANK_VALUE", "21"))

def load_csv(csv_path: str) -> pd.DataFrame:
    # robust for spanish csv; fallback to utf-8
    try:
        return pd.read_csv(csv_path)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin-1")

def prepare_df(df: pd.DataFrame, domains: list[str], primary_domain: str, non_rank_value: int) -> pd.DataFrame:
    out = df.copy()

    # numeric conversions
    out["difficulty"] = to_numeric_series(out.get("Google Dificultad Palabra Clave"))
    out["volume"] = to_numeric_series(out.get("# de búsquedas"))
    out["impressions"] = to_numeric_series(out.get("Impresiones"))
    out["ctr_gsc"] = percent_to_float(out.get("CTR"))
    out["cpc"] = money_to_float(out.get("CPC prom.", pd.Series([np.nan]*len(out))))
    out["group"] = out.get("Grupo Palabra Clave").astype(str).fillna("(sin grupo)")
    out["keyword"] = out.get("Palabra clave").astype(str)

    # positions + urls + visibility
    for d in domains:
        pos_col = f"Posición en Google {d}"
        vis_col = f"Visibilidad {d}"
        url_col = f"Google URL encontrada {d}"

        if pos_col in out.columns:
            out[f"pos::{d}"] = normalize_position(out[pos_col], non_rank_value=non_rank_value)
        else:
            out[f"pos::{d}"] = non_rank_value

        if vis_col in out.columns:
            out[f"vis::{d}"] = percent_to_float(out[vis_col]).fillna(0.0)
        else:
            out[f"vis::{d}"] = 0.0

        if url_col in out.columns:
            out[f"url::{d}"] = out[url_col].astype(str).replace({"N/D": ""}).fillna("")
        else:
            out[f"url::{d}"] = ""

    # derived for primary
    out["pos_primary"] = out[f"pos::{primary_domain}"]
    out["pos_bucket"] = bucket_position(out["pos_primary"])
    out["ctr_est"] = ctr_model(out["pos_primary"])
    out["traffic_est"] = (out["volume"].fillna(0) * out["ctr_est"]).fillna(0)

    # potential if Top 3 (assume pos 3 CTR)
    top3_ctr = 0.10
    out["traffic_potential_top3"] = (out["volume"].fillna(0) * top3_ctr).fillna(0)
    out["traffic_gain_to_top3"] = (out["traffic_potential_top3"] - out["traffic_est"]).clip(lower=0)

    # value estimate (if cpc present)
    out["value_gain_to_top3"] = (out["traffic_gain_to_top3"] * out["cpc"]).replace([np.inf, -np.inf], np.nan)

    # KEI
    out["kei"] = out["volume"] / out["difficulty"].replace({0: np.nan})
    out["kei"] = out["kei"].replace([np.inf, -np.inf], np.nan)

    # sections from primary URL
    out["url_primary"] = out.get(f"url::{primary_domain}", "").astype(str)
    out["section"] = out["url_primary"].apply(extract_section)

    return out

def compute_sov(df_prep: pd.DataFrame, domains: list[str]) -> pd.DataFrame:
    totals = {}
    for d in domains:
        totals[d] = float(df_prep[f"vis::{d}"].sum())
    total_all = sum(totals.values()) or 1.0
    sov = pd.DataFrame({
        "domain": list(totals.keys()),
        "visibility_sum": list(totals.values())
    })
    sov["sov_pct"] = sov["visibility_sum"] / total_all * 100.0
    sov = sov.sort_values("sov_pct", ascending=False)
    return sov

def overlap_matrix(df_prep: pd.DataFrame, primary_domain: str, domains: list[str]) -> pd.DataFrame:
    rows = []
    our = df_prep[f"pos::{primary_domain}"]
    for d in domains:
        if d == primary_domain:
            continue
        comp = df_prep[f"pos::{d}"]
        rows.append({
            "Competidor": d,
            "Keywords donde comp. gana": int((comp < our).sum()),
            "Keywords donde nosotros ganamos": int((our < comp).sum()),
            "Empate (misma pos.)": int((our == comp).sum()),
            "Comp. Top10 y nosotros >20": int(((comp <= 10) & (our > 20)).sum()),
            "Nosotros Top3 y comp 4-6": int(((our <= 3) & (comp.between(4,6))).sum()),
        })
    return pd.DataFrame(rows).sort_values("Keywords donde comp. gana", ascending=False)

def render_kpi_row(df_prep: pd.DataFrame, primary_domain: str):
    total = len(df_prep)
    top3 = int((df_prep["pos_primary"] <= 3).sum())
    top10 = int((df_prep["pos_primary"] <= 10).sum())
    top20 = int((df_prep["pos_primary"] <= 20).sum())
    avg_pos = float(df_prep["pos_primary"].mean()) if total else 0.0
    traffic = float(df_prep["traffic_est"].sum())
    gain = float(df_prep["traffic_gain_to_top3"].sum())
    value_gain = float(df_prep["value_gain_to_top3"].sum(skipna=True)) if "value_gain_to_top3" in df_prep else np.nan

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Keywords", f"{total:,}".replace(",", "."))
    c2.metric("Top 3", f"{top3:,}".replace(",", "."))
    c3.metric("Top 10", f"{top10:,}".replace(",", "."))
    c4.metric("Top 20", f"{top20:,}".replace(",", "."))
    c5.metric("Posición media (<=20=21)", f"{avg_pos:.1f}")
    c6.metric("Tráfico est. (CTR)", f"{traffic:,.0f}".replace(",", "."))
    st.caption("Estimaciones: CTR por posición (heurística) y potencial Top3 asumiendo CTR pos3≈10%.")

    if not np.isnan(value_gain):
        st.info(f"💰 **Valor potencial (equiv. CPC)** si todas las keywords objetivo subieran a Top 3 (estimación): **{value_gain:,.0f} €**".replace(",", "."))

def page_import():
    st.subheader("📥 Importar CSV(s) (mismo formato)")
    st.write("Sube uno o varios CSV. Cada import crea un dataset para comparar periodos y generar tendencias.")
    project = st.text_input("Proyecto/Cliente", value="Radiofonics")
    period_override = st.text_input("Periodo (AAAA-MM) opcional", value="")
    files = st.file_uploader("CSV", type=["csv"], accept_multiple_files=True)

    if files:
        imported = 0
        errors = 0
        for f in files:
            content = f.read()
            sha1 = hashlib.sha1(content).hexdigest()
            dataset_id = hashlib.sha1((sha1 + str(datetime.utcnow().timestamp())).encode("utf-8")).hexdigest()[:16]
            period = period_override.strip() or infer_period_from_filename(f.name)
            csv_path = str((CSV_DIR / f"{dataset_id}.csv").resolve())
            with open(csv_path, "wb") as out:
                out.write(content)

            try:
                df = load_csv(csv_path)
                ok, missing = validate_expected_columns(df)
                if not ok:
                    st.error(f"❌ {f.name}: faltan columnas requeridas: {', '.join(missing)}")
                    errors += 1
                    Path(csv_path).unlink(missing_ok=True)
                    continue
                add_dataset(
                    dataset_id=dataset_id,
                    project=project.strip(),
                    period=period,
                    original_filename=f.name,
                    csv_path=csv_path,
                    sha1=sha1,
                    row_count=len(df),
                )
                imported += 1
            except Exception as e:
                st.error(f"❌ Error importando {f.name}: {e}")
                errors += 1
                Path(csv_path).unlink(missing_ok=True)

        if imported:
            st.success(f"✅ Importados {imported} CSV(s).")
        if errors:
            st.warning(f"⚠️ Hubo {errors} error(es).")

def page_datasets():
    st.subheader("🗂️ Datasets")
    project = st.text_input("Filtrar por proyecto", value="Radiofonics")
    items = list_datasets(project=project.strip() or None)
    if not items:
        st.info("No hay datasets todavía. Ve a 'Importar' para subir CSVs.")
        return

    df_list = pd.DataFrame(items)[["id","project","period","original_filename","row_count","created_at"]]
    df_list_ui = humanize_df(df_list, primary_domain="")
    df_list_ui = coerce_numeric_for_display(df_list_ui)
    cfg = build_column_config(df_list_ui)
    safe_dataframe(df_list_ui, column_config=cfg, use_container_width=True, hide_index=True)

    del_id = st.text_input("ID a borrar (cuidado)", value="")
    if st.button("Borrar dataset", type="secondary", disabled=not del_id.strip()):
        delete_dataset(del_id.strip())
        st.success("Dataset borrado. Refresca la página.")

def load_selected_dataset(project: str):
    items = list_datasets(project=project)
    if not items:
        return None, None
    options = {f"{it['period']} · {it['original_filename']} ({it['row_count']} filas)": it["id"] for it in items}
    label = st.sidebar.selectbox("Dataset", list(options.keys()))
    selected_id = options[label]
    meta = get_dataset(selected_id)
    df = None
    # FIX: Reconstruct path relative to current CSV_DIR (ignoring absolute path in DB which might be from Docker)
    csv_filename = f"{meta['id']}.csv"
    csv_path = CSV_DIR / csv_filename
    
    if not csv_path.exists():
        # Fallback: try the path stored in DB exactly as is (in case it IS a valid local absolute path)
        if Path(meta["csv_path"]).exists():
             csv_path = Path(meta["csv_path"])
        else:
             st.error(f"⚠️ El archivo CSV ({csv_filename}) no se encuentra en `data/csv`. ¿Quizás borraste la carpeta o moviste la BD?")
             return meta, None

    try:
        df = load_csv(str(csv_path))
    except Exception as e:
        st.error(f"Error cargando CSV: {e}")
        return meta, None
        
    return meta, df

def page_dashboard(meta, df):
    st.subheader("📈 SEO Decision Dashboard (Executive + Copywriters)")

    domains = detect_domains(df)
    if not domains:
        st.error("No se detectaron columnas tipo 'Visibilidad <dominio>' en el CSV. Revisa el export.")
        return

    primary_default = DEFAULT_PRIMARY if DEFAULT_PRIMARY in domains else domains[0]

    # ===== Sidebar controls
    st.sidebar.markdown("### Vista")
    persona = st.sidebar.radio("Equipo", ["Dirección", "Copywriters"], index=0)

    st.sidebar.markdown("### Dominio")
    primary = st.sidebar.selectbox("Dominio principal", domains, index=domains.index(primary_default))
    non_rank = st.sidebar.number_input("Valor para 'No Top20'", min_value=21, max_value=200, value=DEFAULT_NON_RANK, step=1)

    df_prep = prepare_df(df, domains=domains, primary_domain=primary, non_rank_value=int(non_rank))

    # ===== Helper: register contexts for AI
    if "llm_contexts" not in st.session_state:
        st.session_state["llm_contexts"] = {}

    def _register_ctx(key: str, df_ctx: pd.DataFrame, description: str):
        if df_ctx is None:
            return
        df_ui = humanize_df(df_ctx, primary_domain=primary)
        df_ui = coerce_numeric_for_display(df_ui)
        # Store a small copy (avoid huge memory)
        st.session_state["llm_contexts"][key] = {
            "description": description,
            "df": df_ui.copy()
        }

    # ===== Compute best competitor (gap / threats)
    comp_domains = [d for d in domains if d != primary]
    if comp_domains:
        comp_mat = np.vstack([df_prep[f"pos::{d}"].to_numpy() for d in comp_domains]).T
        best_idx = np.argmin(comp_mat, axis=1)
        df_prep["best_comp_domain"] = np.array(comp_domains, dtype=object)[best_idx]
        df_prep["best_comp_pos"] = comp_mat[np.arange(len(df_prep)), best_idx]
    else:
        df_prep["best_comp_domain"] = ""
        df_prep["best_comp_pos"] = np.nan

    # Action flags
    df_prep["is_striking"] = df_prep["pos_primary"].between(4, 10)
    df_prep["is_gap"] = (df_prep["pos_primary"] >= int(non_rank)) & (df_prep["best_comp_pos"] <= 10)
    df_prep["is_threat"] = (df_prep["pos_primary"] <= 3) & (df_prep["best_comp_pos"].between(4, 6))

    # Opportunity score (simple + interpretable)
    df_prep["opp_score"] = (
        df_prep["traffic_gain_to_top3"].fillna(0)
        * (1 + np.log1p(df_prep["cpc"].fillna(0)))
        * (1 + (df_prep["kei"].fillna(0) / (df_prep["kei"].fillna(0).median() + 1e-9)).clip(0, 3) / 3)
    )

    st.caption(
        f"**Proyecto:** {meta['project']} · **Periodo:** {meta['period']} · **Archivo:** {meta['original_filename']} · "
        f"**Dominio principal:** {primary}"
    )


    # ===== Slices accionables (definidas una vez para todas las pestañas)
    striking = df_prep[df_prep["is_striking"]].copy()
    gap = df_prep[df_prep["is_gap"]].copy()
    threats = df_prep[df_prep["is_threat"]].copy()

    def _action_tbl(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in is None or df_in.empty:
            return pd.DataFrame(columns=["Keyword","Grupo","Posición","Volumen","KD","CPC (€)","Ganancia Top3 (clics)","Valor Top3 (€)","URL","Competidor","Posición competidor"])
        cols = {
            "Keyword": "keyword",
            "Grupo": "group",
            "Posición": "pos_primary",
            "Volumen": "volume",
            "KD": "difficulty",
            "CPC (€)": "cpc",
            "Ganancia Top3 (clics)": "traffic_gain_to_top3",
            "Valor Top3 (€)": "value_gain_to_top3",
            "URL": "url_primary",
            "Competidor": "best_comp_domain",
            "Posición competidor": "best_comp_pos",
        }
        existing = {k:v for k,v in cols.items() if v in df_in.columns}
        out = df_in[list(existing.values())].rename(columns={v:k for k,v in existing.items()})
        # Orden: impacto económico, luego tráfico, luego volumen
        if "Valor Top3 (€)" in out.columns:
            out["Valor Top3 (€)"] = pd.to_numeric(out["Valor Top3 (€)"], errors="coerce").fillna(0)
        if "Ganancia Top3 (clics)" in out.columns:
            out["Ganancia Top3 (clics)"] = pd.to_numeric(out["Ganancia Top3 (clics)"], errors="coerce").fillna(0)
        if "Volumen" in out.columns:
            out["Volumen"] = pd.to_numeric(out["Volumen"], errors="coerce").fillna(0)

        sort_cols = [c for c in ["Valor Top3 (€)","Ganancia Top3 (clics)","Volumen"] if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols, ascending=False)

        return out

    striking_tbl = _action_tbl(striking)
    gap_tbl = _action_tbl(gap)
    threats_tbl = _action_tbl(threats)

    # ===== Main tabs
    tabA, tabB, tabC, tabD, tabE = st.tabs([
        "📌 Acción (1 vistazo)",
        "📊 Análisis",
        "✍️ Backlog copy",
        "🤖 IA (interpretación)",
        "🧼 Datos & definiciones",
    ])

    # =========================
    # TAB A — Action in one glance
    # =========================
    with tabA:
        st.markdown("#### Qué pasa y qué hacemos ahora")
        render_kpi_row(df_prep, primary_domain=primary)

        # ✅ Layout vertical (1 columna) — mejor para dirección y copywriters (sin apretar tablas)
        def action_block(title: str, subtitle: str, tbl: pd.DataFrame, ctx_name: str, ctx_desc: str, dl_slug: str):
            st.markdown(f"### {title}")
            st.caption(subtitle)
            df_ui = humanize_df(tbl, primary_domain=primary)
            df_display = coerce_numeric_for_display(df_ui)
            cfg = build_column_config(df_display)
            safe_dataframe(df_display, column_config=cfg, use_container_width=True, hide_index=True)
            _register_ctx(ctx_name, df_display, ctx_desc)
            st.download_button(
                "⬇️ Descargar CSV",
                data=df_display.to_csv(index=False).encode("utf-8"),
                file_name=f"{dl_slug}_{meta['project']}_{meta['period']}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.divider()

        action_block(
            "🎯 Atacar ahora (Striking Distance)",
            f"{len(striking):,} keywords en pos 4–10",
            striking_tbl,
            "Atacar · Striking Distance",
            "Top keywords en posición 4–10 ordenadas por ganancia estimada a Top 3.",
            "atacar_striking",
        )

        action_block(
            "🧱 Crear (Content Gap)",
            f"{len(gap):,} keywords con competidor Top10 y nosotros fuera",
            gap_tbl,
            "Crear · Content Gap",
            "Keywords donde un competidor está en Top10 y el dominio principal no aparece en Top20.",
            "crear_gap",
        )

        action_block(
            "🛡️ Defender (Amenazas)",
            f"{len(threats):,} keywords Top3 con competidor 4–6",
            threats_tbl,
            "Defender · Amenazas",
            "Keywords donde estamos en Top3 pero un competidor está cerca (pos 4–6).",
            "defender_amenazas",
        )


        # Exec summary bullets
        st.markdown("#### Decisiones sugeridas (automáticas)")
        bullets = []
        if len(striking) > 0:
            bullets.append(f"**Atacar**: priorizar *{min(10, len(striking))}* keywords en pos 4–10 con mayor *Gain Top3* (optimización on-page + enlazado interno).")
        if len(gap) > 0:
            bullets.append(f"**Crear**: abrir clústeres nuevos en {gap['group'].nunique()} grupos (keywords con competidor Top10 y nosotros fuera).")
        if len(threats) > 0:
            bullets.append(f"**Defender**: refrescar y reforzar {min(10, len(threats))} URLs Top3 donde el competidor está a 1–2 posiciones.")
        if not bullets:
            bullets.append("No se detectaron oportunidades claras con los umbrales actuales. Ajusta el valor 'No Top20' o revisa el dataset.")

        st.markdown("\n".join([f"- {b}" for b in bullets]))

    # =========================
    # TAB B — Analysis (charts)
    # =========================
    with tabB:
        st.markdown("#### Lectura estratégica (mercado, potencial y prioridades)")

        sov = compute_sov(df_prep, domains)
        colA, colB = st.columns([1.2, 1])
        with colA:
            fig = px.pie(sov, values="sov_pct", names="domain", title="Market Share SEO (SoV) – % visibilidad")
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            st.write("**SoV por dominio**")
            tbl = sov[["domain","sov_pct"]].rename(columns={"domain":"Dominio","sov_pct":"SoV %"}).round({"SoV %":2})
            tbl_ui = humanize_df(tbl, primary_domain=primary)
            tbl_ui = coerce_numeric_for_display(tbl_ui)
            cfg = build_column_config(tbl_ui)
            safe_dataframe(tbl_ui, column_config=cfg, use_container_width=True, hide_index=True)
            _register_ctx("SoV · Market Share", tbl_ui, "Tabla de Share of Voice (% visibilidad) por dominio.")

        st.divider()

        st.write("#### Tendencia (histórico del proyecto)")
        items = list_datasets(project=meta["project"])
        trend_rows = []
        for it in items:
            try:
                dfi = load_csv(it["csv_path"])
                doms = detect_domains(dfi)
                if primary not in doms:
                    continue
                dprep = prepare_df(dfi, domains=doms, primary_domain=primary, non_rank_value=int(non_rank))
                sov_i = compute_sov(dprep, doms)
                our_sov = float(sov_i.loc[sov_i["domain"] == primary, "sov_pct"].iloc[0]) if (sov_i["domain"] == primary).any() else 0.0
                trend_rows.append({
                    "period": it["period"],
                    "sov_pct": our_sov,
                    "avg_pos": float(dprep["pos_primary"].mean()),
                    "keywords": int(len(dprep)),
                })
            except Exception:
                continue
        if trend_rows:
            trend = pd.DataFrame(trend_rows)
            trend["period_label"] = trend["period"].astype(str).str.slice(0, 7)
            trend = trend.sort_values("period_label")
            fig = px.line(
                trend,
                x="period_label",
                y="sov_pct",
                markers=True,
                labels={"period_label": "Periodo", "sov_pct": "SoV %"},
                title=f"Evolución SoV – {primary}"
            )
            fig.update_xaxes(type="category")
            st.plotly_chart(fig, use_container_width=True)
            _register_ctx("Tendencia · SoV histórico", trend, "Evolución del SoV del dominio principal a lo largo de los datasets del proyecto.")
        else:
            st.info("Aún no hay suficiente histórico (sube más de un CSV) para ver tendencias.")

        st.divider()

        st.write("#### Prioridades (Scatter: Volumen vs Dificultad · burbuja=CPC)")
        scatter = df_prep.copy()
        scatter_ui = humanize_df(scatter, primary_domain=primary)

        df_scatter = scatter_ui.copy()
        x_col = "KD" if "KD" in df_scatter else "difficulty"
        y_col = "Volumen" if "Volumen" in df_scatter else "volume"
        size_col = "CPC (€)" if "CPC (€)" in df_scatter else ("cpc" if "cpc" in df_scatter else None)

        df_scatter["difficulty_num"] = to_number(df_scatter[x_col]) if x_col in df_scatter else np.nan
        df_scatter["volume_num"] = to_number(df_scatter[y_col]) if y_col in df_scatter else np.nan
        if size_col:
            df_scatter["size_num"] = to_number(df_scatter[size_col])
        else:
            df_scatter["size_num"] = np.nan

        if df_scatter["size_num"].dropna().empty or df_scatter["size_num"].max(skipna=True) <= 0:
            df_scatter["size_num"] = 1

        custom_cols, hovertemplate = build_scatter_tooltips(df_scatter)
        color_col = "Bucket posición" if "Bucket posición" in df_scatter else None
        fig = px.scatter(
            df_scatter,
            x="difficulty_num",
            y="volume_num",
            size="size_num",
            color=color_col,
            custom_data=custom_cols,
            hover_data=[],
            labels={
                "difficulty_num": "Dificultad (KD)",
                "volume_num": "Volumen",
                "size_num": "CPC (€)",
            },
            title="Prioridades: alto volumen + baja dificultad + mejor CPC (burbuja)"
        )
        if df_scatter["difficulty_num"].notna().any():
            max_kd = df_scatter["difficulty_num"].max(skipna=True)
            upper = 100 if max_kd <= 100 else float(max_kd * 1.05)
            fig.update_xaxes(range=[0, upper])
        if hovertemplate:
            fig.update_traces(hovertemplate=hovertemplate, hoverlabel=dict(namelength=-1))
        st.plotly_chart(fig, use_container_width=True)
        ctx_cols = [c for c in ["Keyword","Grupo","Volumen","KD","CPC (€)","Posición","Ganancia Top3 (clics)","Competidor","Posición competidor"] if c in scatter_ui.columns]
        _register_ctx("Scatter · Prioridades", scatter_ui[ctx_cols].head(200),
                      "Puntos (keywords) para el scatter de prioridades.")

    # =========================
    # TAB C — Copywriter backlog
    # =========================
    with tabC:
        st.markdown("#### Backlog accionable para redactores (qué publicar / qué actualizar)")
        st.caption("Filtra por clúster o sección. Exporta un backlog para Trello/Notion/Jira.")

        groups = sorted([g for g in df_prep["group"].dropna().unique().tolist() if g and g != "nan"])
        sections = sorted([s for s in df_prep["section"].dropna().unique().tolist() if s and s != "nan"])

        fcol1, fcol2, fcol3 = st.columns([1, 1, 1])
        with fcol1:
            sel_groups = st.multiselect("Filtrar por Grupo", groups, default=[])
        with fcol2:
            sel_sections = st.multiselect("Filtrar por Sección (carpeta raíz)", sections, default=[])
        with fcol3:
            bucket = st.selectbox("Filtrar por Bucket", ["(todos)","Top 3","4–10","11–20","No Top20"], index=0)

        flt = df_prep.copy()
        if sel_groups:
            flt = flt[flt["group"].isin(sel_groups)]
        if sel_sections:
            flt = flt[flt["section"].isin(sel_sections)]
        if bucket != "(todos)":
            flt = flt[flt["pos_bucket"] == bucket]

        def _action(row):
            if row.get("is_gap", False):
                return "Crear contenido nuevo (gap)"
            if row.get("is_striking", False):
                return "Optimizar contenido existente (striking distance)"
            if row.get("is_threat", False):
                return "Defender/actualizar (amenaza)"
            if not row.get("url_primary", ""):
                return "Crear contenido nuevo"
            return "Optimizar"

        backlog = flt.copy()
        backlog["accion"] = backlog.apply(_action, axis=1)

        backlog_show = backlog[[
            "accion","keyword","group","section","pos_primary","best_comp_domain","best_comp_pos",
            "volume","difficulty","cpc","traffic_gain_to_top3","value_gain_to_top3","url_primary"
        ]].copy()

        backlog_show = backlog_show.rename(columns={
            "accion":"Acción","keyword":"Keyword","group":"Grupo","section":"Sección",
            "pos_primary":"Posición","best_comp_domain":"Competidor","best_comp_pos":"Posición competidor",
            "volume":"Volumen","difficulty":"KD","cpc":"CPC (€)",
            "traffic_gain_to_top3":"Ganancia Top3 (clics)","value_gain_to_top3":"Valor Top3 (€)",
            "url_primary":"URL"
        })

        # Sort: most valuable first (fallback to traffic gain)
        for col in ["Valor Top3 (€)","Ganancia Top3 (clics)","Volumen"]:
            if col in backlog_show:
                backlog_show[col] = to_number(backlog_show[col]).fillna(0)
        backlog_show = backlog_show.sort_values(["Valor Top3 (€)","Ganancia Top3 (clics)","Volumen"], ascending=False).head(200)

        backlog_ui = humanize_df(backlog_show, primary_domain=primary)
        backlog_ui = coerce_numeric_for_display(backlog_ui)
        backlog_render = backlog_ui.round({"KD":1,"CPC (€)":2,"Ganancia Top3 (clics)":0,"Valor Top3 (€)":0})
        cfg = build_column_config(backlog_ui)
        safe_dataframe(backlog_render, column_config=cfg, use_container_width=True, hide_index=True)
        _register_ctx("Backlog · Copywriters", backlog_ui.head(60), "Backlog priorizado para copywriters con acción sugerida y métricas de impacto.")

        st.download_button(
            "⬇️ Descargar backlog (CSV)",
            data=backlog_ui.to_csv(index=False).encode("utf-8"),
            file_name=f"backlog_{meta['project']}_{meta['period']}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # =========================
    # TAB D — AI interpretation
    # =========================
    with tabD:
        st.markdown("#### IA para interpretar y redactar conclusiones (opcional)")
        st.caption("La IA NO es obligatoria: el dashboard funciona sin ella. Úsala para generar informes y briefs.")
        st.warning(
            "Seguridad: la API key debe usarse **solo en servidor** y no debe exponerse en el navegador. "
            "En local estás bien; en producción guarda la clave como variable de entorno."
        )

        # Key handling
        env_key = os.getenv("OPENAI_API_KEY", "")
        if env_key:
            st.success("✅ API Key autenticada por el sistema (modo seguro).")
            st.session_state["OPENAI_API_KEY"] = env_key
            api_key = env_key
        else:
            api_key = st.text_input("OpenAI API key", type="password", value=st.session_state.get("OPENAI_API_KEY", ""))
            if api_key:
                st.session_state["OPENAI_API_KEY"] = api_key

        model_choices = [
            "gpt-5.2",
            "gpt-5.1",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4o",
            "gpt-4o-mini",
        ]
        model = st.selectbox("Modelo", model_choices, index=2)
        max_out = st.slider("Máx. tokens de salida", min_value=256, max_value=4096, value=1200, step=64)

        contexts = st.session_state.get("llm_contexts", {})
        if not contexts:
            st.info("Aún no hay contextos disponibles para analizar (navega por las pestañas para generarlos).")
            return

        ctx_key = st.selectbox("Elemento del dashboard a interpretar", list(contexts.keys()))
        ctx = contexts[ctx_key]
        st.write("**Contexto seleccionado:**", ctx["description"])
        ctx_preview = humanize_df(ctx["df"].head(20), primary_domain=primary)
        ctx_preview = coerce_numeric_for_display(ctx_preview)
        cfg = build_column_config(ctx_preview)
        safe_dataframe(ctx_preview, column_config=cfg, use_container_width=True, hide_index=True)

        audience = st.radio("Audiencia del informe", ["Dirección", "Copywriters"], horizontal=True, index=0)
        tone = st.selectbox("Formato", ["Bullets ejecutivos", "Informe (1 página)", "Checklist de acciones"], index=0)

        user_extra = st.text_area("Instrucciones extra (opcional)", value="", placeholder="Ej: enfócate en cursos de locución, prioriza ROI, etc.")

        if st.button("🧠 Generar informe con IA", use_container_width=True, disabled=not bool(api_key)):
            try:
                client = OpenAI(api_key=api_key)
                sys = (
                    "Eres un analista SEO senior. Interpreta datos de un dashboard de posicionamiento y competencia. "
                    "Sé directo, accionable y evita jerga."
                )
                if audience == "Dirección":
                    goal = (
                        "Escribe conclusiones ejecutivas: qué significa, qué riesgo/oportunidad hay, "
                        "y las 5 decisiones recomendadas para los próximos 14 días."
                    )
                else:
                    goal = (
                        "Escribe un brief para copywriters: qué contenidos crear/actualizar, estructura sugerida, "
                        "y una lista de tareas priorizadas."
                    )

                fmt = {
                    "Bullets ejecutivos": "Devuelve 10–15 bullets muy claros y priorizados.",
                    "Informe (1 página)": "Devuelve un informe breve (máx 350–500 palabras) + 5 acciones.",
                    "Checklist de acciones": "Devuelve checklist con tareas, dueño (SEO/Copy), y criterio de éxito.",
                }[tone]
                payload = (
                    f"Contexto: {ctx_key}\n"
                    f"Descripción: {ctx['description']}\n\n"
                    f"Datos (CSV):\n{df_to_csv_snippet(ctx['df'])}\n\n"
                    f"Objetivo: {goal}\n"
                    f"Formato: {fmt}\n"
                )
                if user_extra.strip():
                    payload += f"\nInstrucciones extra del usuario: {user_extra.strip()}\n"

                resp = client.responses.create(
                    model=model,
                    instructions=sys,
                    input=payload,
                    max_output_tokens=int(max_out),
                )
                # openai python sdk exposes output_text convenience on response
                out_text = getattr(resp, "output_text", None)
                if not out_text:
                    # Fallback: try to join message outputs if present
                    out_text = str(resp)
                st.success("Informe generado")
                st.markdown(out_text)

                st.markdown("---")
                st.write("**Descargar informe**")
                d1, d2, d3, d4 = st.columns(4)
                with d1:
                    st.download_button(
                        "⬇️ TXT",
                        data=out_text.encode("utf-8"),
                        file_name=f"informe_ia_{meta['project']}_{meta['period']}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with d2:
                    st.download_button(
                        "⬇️ CSV",
                        data=text_to_csv_bytes(out_text),
                        file_name=f"informe_ia_{meta['project']}_{meta['period']}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with d3:
                    try:
                        pdf_bytes = text_to_pdf_bytes(f"Informe IA · {meta['project']} · {meta['period']}", out_text)
                        st.download_button(
                            "⬇️ PDF",
                            data=pdf_bytes,
                            file_name=f"informe_ia_{meta['project']}_{meta['period']}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    except Exception as _pdf_err:
                        st.caption(f"No se pudo generar PDF: {_pdf_err}")
                with d4:
                    st.download_button(
                        "⬇️ MD",
                        data=out_text.encode("utf-8"),
                        file_name=f"informe_ia_{meta['project']}_{meta['period']}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Error llamando a OpenAI: {e}")

    # =========================
    # TAB E — Data hygiene and definitions
    # =========================
    with tabE:
        st.markdown("#### Cómo se calculan las métricas (para que el cliente confíe)")
        st.markdown(
            f"""
- **Normalización posiciones**: textos tipo *"No está entre las primeras 20"* → **{int(non_rank)}**
- **SoV (Market Share SEO)**: se reparte la visibilidad por keyword entre dominios y se suma (porcentaje final).
- **CTR estimado**: modelo simple por posición (para estimar tráfico potencial).
- **Ganancia Top3**: tráfico estimado si subimos a Top3 – tráfico actual (nunca negativo).
- **KEI**: Volumen / Dificultad (aprox. “easy wins”).
- **Sección**: carpeta raíz extraída de la URL (ej. `/cursos/`).
"""
        )

        st.divider()
        st.write("**Ejemplo (crudo vs normalizado)**")
        cols_raw = ["Palabra clave", f"Posición en Google {primary}", "CPC prom.", "Grupo Palabra Clave", f"Visibilidad {primary}", f"Google URL encontrada {primary}"]
        cols_norm = ["keyword","pos_primary","cpc","group","section","traffic_est","traffic_gain_to_top3","kei","best_comp_domain","best_comp_pos"]

        raw = df[[c for c in cols_raw if c in df.columns]].head(25).copy()
        raw = raw.rename(columns={c: f"raw::{c}" for c in raw.columns})
        norm = df_prep[[c for c in cols_norm if c in df_prep.columns]].head(25).copy()
        norm = norm.rename(columns={c: f"norm::{c}" for c in norm.columns})

        preview = pd.concat([raw, norm], axis=1)
        preview_ui = humanize_df(preview, primary_domain=primary)
        preview_ui = coerce_numeric_for_display(preview_ui)
        cfg = build_column_config(preview_ui)
        safe_dataframe(preview_ui, column_config=cfg, use_container_width=True, hide_index=True)



def main():
    init_db()

    st.sidebar.title("SEO Exec Dashboard")
    page = st.sidebar.radio("Sección", ["Dashboard", "Importar", "Datasets"], index=0)

    if page == "Importar":
        page_import()
        return
    if page == "Datasets":
        page_datasets()
        return

    project = st.sidebar.text_input("Proyecto", value="Radiofonics")
    meta, df = load_selected_dataset(project=project.strip())
    if meta is None:
        st.info("Sube tu primer CSV en 'Importar' para empezar.")
        return
    if df is None:
        st.warning("No se pudo cargar el dataset seleccionado. Revisa si falta el archivo CSV.")
        return
    page_dashboard(meta, df)

if __name__ == "__main__":
    main()
