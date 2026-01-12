import os
import json
import io
import textwrap
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from utils import humanize_df, build_column_config, coerce_numeric_for_display

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


DEFAULT_MODELS = [
    "gpt-5.2",
    "gpt-5",
    "gpt-4o",
    "gpt-4o-mini",
]


def _text_to_csv_bytes(text: str) -> bytes:
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    out = io.StringIO()
    out.write("line\n")
    for l in lines:
        out.write('"' + l.replace('"', '""') + '"\n')
    return out.getvalue().encode("utf-8")


def _text_to_pdf_bytes(title: str, text: str) -> bytes:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    x, y = 40, height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title[:120])
    y -= 26
    c.setFont("Helvetica", 10)
    for line in (text or "").splitlines():
        for w in textwrap.wrap(line, width=110) or [""]:
            if y < 50:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - 50
            c.drawString(x, y, w)
            y -= 14
    c.save()
    buf.seek(0)
    return buf.read()


def ai_sidebar_settings() -> Tuple[str, str, str, bool]:
    """Returns (api_key, model, audience, enabled)."""
    st.sidebar.subheader("🧠 AI Copilot")
    api_key_env = os.getenv("OPENAI_API_KEY", "")
    api_key = st.sidebar.text_input("OpenAI API key", value=st.session_state.get("openai_api_key", api_key_env), type="password")
    model = st.sidebar.selectbox("Modelo", options=DEFAULT_MODELS, index=DEFAULT_MODELS.index(st.session_state.get("openai_model", "gpt-4o-mini")) if st.session_state.get("openai_model", "gpt-4o-mini") in DEFAULT_MODELS else DEFAULT_MODELS.index("gpt-4o-mini"))
    custom = st.sidebar.text_input("Modelo personalizado (opcional)", value="")
    if custom.strip():
        model = custom.strip()

    audience = st.sidebar.selectbox("Audiencia", ["Dirección", "Copywriters", "Ambos"], index=2)
    remember = st.sidebar.checkbox("Recordar en esta sesión", value=True, help="No guardamos la clave en disco. Solo se mantiene en memoria del navegador/servidor mientras dura la sesión.")
    enabled = bool(api_key.strip()) and OpenAI is not None

    if remember:
        st.session_state["openai_api_key"] = api_key
        st.session_state["openai_model"] = model

    if OpenAI is None:
        st.sidebar.warning("Falta la librería `openai`. Rebuild del contenedor.")
    elif not api_key.strip():
        st.sidebar.info("Introduce la API key para activar el copilot.")
    else:
        st.sidebar.success("Copilot listo ✅")

    return api_key, model, audience, enabled


def _df_payload(df: pd.DataFrame, max_rows: int = 25) -> Dict[str, Any]:
    if df is None:
        return {"rows": [], "columns": []}
    if isinstance(df, pd.Series):
        df = df.to_frame()
    d = humanize_df(df, primary_domain="")
    d = d.copy()
    d = d.head(max_rows)
    # Convert to JSON-safe primitives
    d = d.replace({pd.NA: None})
    return {"columns": list(d.columns), "rows": d.to_dict(orient="records"), "row_count": int(len(df))}


@st.cache_data(show_spinner=False)
def ai_generate_insights_cached(
    api_key: str,
    model: str,
    audience: str,
    artifact_name: str,
    meta: Dict[str, Any],
    payload_json: str,
) -> str:
    """Cache by (inputs + payload)."""
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK no disponible")
    client = OpenAI(api_key=api_key)

    system = (
        "Eres un Analista SEO Senior (nivel ejecutivo) y Content Strategist. "
        "Responde en español. Sé prescriptivo: prioriza, cuantifica impacto y da acciones concretas. "
        "Estructura tu salida en Markdown con secciones y bullets. "
        "No inventes datos; si falta algo, dilo y propone cómo medirlo."
    )

    # Small guardrail: keep it short for execs, actionable for writers
    if audience == "Dirección":
        style = "Enfócate en impacto, riesgo, priorización, y decisiones de inversión. Máximo 12 bullets + 5 acciones."
    elif audience == "Copywriters":
        style = "Enfócate en tareas concretas de contenido: qué optimizar/crear, ángulos, estructura, títulos y FAQs. Máximo 15 bullets + checklist."
    else:
        style = "Divide en 2 bloques: Dirección y Copywriters, con acciones para cada equipo."

    user = {
        "artifact": artifact_name,
        "meta": meta,
        "data": json.loads(payload_json),
        "instruction": style,
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": "Analiza el siguiente artefacto del dashboard y devuelve conclusiones accionables."},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        text={"verbosity": "low"},
    )
    return getattr(resp, "output_text", "").strip()


def ai_insights_box(
    api_key: str,
    model: str,
    audience: str,
    enabled: bool,
    meta: Dict[str, Any],
    artifacts: Dict[str, Any],
    default_artifact: Optional[str] = None,
    primary_domain: str = "",
):
    st.write("## 🧠 AI Copilot — Interpretación y conclusiones")
    st.caption("Selecciona una tabla/gráfico (artefacto) y pide un informe para Dirección o Copywriters.")
    names = list(artifacts.keys())
    if not names:
        st.info("No hay artefactos disponibles para interpretar en esta vista.")
        return

    idx = 0
    if default_artifact and default_artifact in names:
        idx = names.index(default_artifact)

    sel = st.selectbox("Qué quieres que interprete", names, index=idx)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("**Modelo:**", model)
    with col2:
        st.write("**Audiencia:**", audience)

    with st.expander("Ver datos que se enviarán al modelo (preview)", expanded=False):
        obj = artifacts.get(sel)
        if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
            df_preview = humanize_df(obj.head(25), primary_domain=primary_domain)
            df_preview = coerce_numeric_for_display(df_preview)
            cfg = build_column_config(df_preview)
            st.dataframe(df_preview, column_config=cfg, use_container_width=True, hide_index=True)
        else:
            st.code(str(obj)[:2000])

    if not enabled:
        st.warning("Para usar AI Copilot: añade tu API key en la barra lateral.")
        return

    if st.button("Generar conclusiones", type="primary"):
        with st.spinner("Generando conclusiones..."):
            obj = artifacts.get(sel)
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                df_payload = obj if isinstance(obj, pd.DataFrame) else obj.to_frame()
                payload = _df_payload(humanize_df(df_payload, primary_domain=primary_domain))
            else:
                payload = {"note": str(obj)[:5000]}

            payload_json = json.dumps(payload, ensure_ascii=False)
            try:
                out = ai_generate_insights_cached(
                    api_key=api_key,
                    model=model,
                    audience=audience,
                    artifact_name=sel,
                    meta=meta,
                    payload_json=payload_json,
                )
                if out:
                    st.markdown(out)
                    st.markdown("---")
                    st.write("**Descargar informe**")
                    d1, d2, d3 = st.columns(3)
                    with d1:
                        st.download_button(
                            "⬇️ Markdown",
                            data=out.encode("utf-8"),
                            file_name=f"ai-insights-{meta.get('project','proyecto')}-{meta.get('period','periodo')}.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )
                    with d2:
                        st.download_button(
                            "⬇️ CSV",
                            data=_text_to_csv_bytes(out),
                            file_name=f"ai-insights-{meta.get('project','proyecto')}-{meta.get('period','periodo')}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    with d3:
                        try:
                            pdf_bytes = _text_to_pdf_bytes(f"AI insights · {meta.get('project','proyecto')} · {meta.get('period','periodo')}", out)
                            st.download_button(
                                "⬇️ PDF",
                                data=pdf_bytes,
                                file_name=f"ai-insights-{meta.get('project','proyecto')}-{meta.get('period','periodo')}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                        except Exception as _pdf_err:
                            st.caption(f"No se pudo generar PDF: {_pdf_err}")
                else:
                    st.error("La respuesta llegó vacía.")
            except Exception as e:
                st.error(f"Error llamando a OpenAI: {e}")
