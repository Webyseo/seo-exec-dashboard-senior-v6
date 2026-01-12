import re
from urllib.parse import urlparse
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st


def _clean_numeric_str(x: pd.Series) -> pd.Series:
    """Clean numeric-looking strings (Spanish/EN) like '1.234', '1,234', '33,3%', '$2.34', 'N/D'."""
    out = x.astype(str).str.strip()
    out = out.replace({"N/D": np.nan, "ND": np.nan, "": np.nan, "nan": np.nan, "None": np.nan})
    # remove currency and percent symbols
    out = out.str.replace("€", "", regex=False)
    out = out.str.replace("$", "", regex=False)
    out = out.str.replace("%", "", regex=False)
    out = out.str.replace("\u00a0", " ", regex=False).str.replace(" ", "", regex=False)
    # normalize decimals: if contains comma and not dot, treat comma as decimal
    # also remove thousands separators
    # strategy: replace '.' thousands when comma used as decimal
    # 1) if has comma, replace '.' with '' then comma with '.'
    has_comma = out.str.contains(",", na=False)
    out = out.where(~has_comma, out.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    # 2) else: keep dot decimal, but remove thousands commas
    out = out.where(has_comma, out.str.replace(",", "", regex=False))
    return out

def to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(_clean_numeric_str(s), errors="coerce")

def percent_to_float(s: pd.Series) -> pd.Series:
    """Returns percent in 0-100 scale as float."""
    return to_float_series(s)

MONTHS_ES = {
    "enero": "01",
    "febrero": "02",
    "marzo": "03",
    "abril": "04",
    "mayo": "05",
    "junio": "06",
    "julio": "07",
    "agosto": "08",
    "septiembre": "09",
    "setiembre": "09",
    "octubre": "10",
    "noviembre": "11",
    "diciembre": "12",
}

def infer_period_from_filename(filename: str) -> str:
    # Accept patterns like "noviembre 2025.csv" or "Nov-2025.csv"
    name = filename.lower()
    m = re.search(r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\s*(\d{4})", name)
    if m:
        month = MONTHS_ES[m.group(1)]
        year = m.group(2)
        return f"{year}-{month}"
    m = re.search(r"(\d{4})[-_ ](\d{2})", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return "unknown"

def detect_domains(df: pd.DataFrame) -> List[str]:
    domains = []
    for c in df.columns:
        if c.startswith("Visibilidad "):
            domains.append(c.replace("Visibilidad ", "").strip())
    # keep consistent order: primary first if possible
    return domains

def money_to_float(s: pd.Series) -> pd.Series:
    return to_float_series(s)

def to_numeric_series(s: pd.Series) -> pd.Series:
    return to_float_series(s)

def normalize_position(s: pd.Series, non_rank_value: int = 21) -> pd.Series:
    raw = s.astype(str).str.strip()
    raw = raw.replace({
        "No está entre las primeras 20": str(non_rank_value),
        "N/D": str(non_rank_value),
        "ND": str(non_rank_value),
        "": str(non_rank_value),
        "nan": str(non_rank_value),
    })
    # if contains digits, extract
    digits = raw.str.extract(r"(\d+)", expand=False)
    out = pd.to_numeric(digits, errors="coerce").fillna(non_rank_value).astype(int)
    return out

def extract_section(url: str) -> str:
    if not isinstance(url, str) or not url or url == "N/D":
        return "(sin URL)"
    try:
        p = urlparse(url)
        path = p.path or "/"
        parts = [x for x in path.split("/") if x]
        if not parts:
            return "/"
        return f"/{parts[0]}"
    except Exception:
        return "(sin URL)"

def ctr_model(pos: pd.Series) -> pd.Series:
    # Heurística CTR por posición (desktop+mobile mezclado, suficiente para estimar potencial)
    # Values beyond 20 -> ~0
    mapping = {
        1: 0.30,
        2: 0.15,
        3: 0.10,
        4: 0.07,
        5: 0.05,
        6: 0.04,
        7: 0.03,
        8: 0.025,
        9: 0.020,
        10: 0.018,
        11: 0.012,
        12: 0.010,
        13: 0.009,
        14: 0.008,
        15: 0.007,
        16: 0.006,
        17: 0.006,
        18: 0.005,
        19: 0.005,
        20: 0.004,
    }
    p = pos.clip(lower=1, upper=100).astype(int)
    ctr = p.map(mapping).fillna(0.0)
    # If >20 -> 0
    ctr = ctr.where(p <= 20, 0.0)
    return ctr.astype(float)

def bucket_position(pos: pd.Series) -> pd.Series:
    p = pos.astype(int)
    bins = []
    for v in p:
        if v <= 3:
            bins.append("Top 3")
        elif v <= 10:
            bins.append("4-10")
        elif v <= 20:
            bins.append("11-20")
        else:
            bins.append(">20")
    return pd.Series(bins, index=pos.index)

def validate_expected_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    required = [
        "Palabra clave",
        "Google Dificultad Palabra Clave",
        "# de búsquedas",
        "Grupo Palabra Clave",
    ]
    missing = [c for c in required if c not in df.columns]
    return (len(missing) == 0, missing)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop raw prefixes and tidy whitespace in column headers."""
    if df is None:
        return df
    if isinstance(df, pd.Series):
        df = df.to_frame()
    out = df.copy()
    cleaned = []
    for col in out.columns:
        name = str(col)
        if name.startswith("raw::") or name.startswith("norm::"):
            name = name.split("::", 1)[1]
        name = name.strip()
        name = re.sub(r"\s{2,}", " ", name)
        cleaned.append(name)
    out.columns = cleaned
    return out


def build_ui_cols_map(primary_domain: str) -> Dict[str, str]:
    primary = (primary_domain or "").strip()
    ui_map: Dict[str, str] = {
        "Palabra clave": "Keyword",
        "keyword": "Keyword",
        "Grupo Palabra Clave": "Grupo",
        "group": "Grupo",
        "pos_primary": "Posición",
        "# de búsquedas": "Volumen",
        "# de busquedas": "Volumen",
        "busquedas": "Volumen",
        "volume": "Volumen",
        "Dificultad": "KD",
        "difficulty": "KD",
        "KD": "KD",
        "Google Dificultad Palabra Clave": "KD",
        "url_primary": "URL",
        "URL": "URL",
        "CPC prom.": "CPC (€)",
        "cpc": "CPC (€)",
        "CPC": "CPC (€)",
        "project": "Proyecto",
        "period": "Periodo",
        "original_filename": "Archivo",
        "row_count": "Filas",
        "created_at": "Creado",
        "id": "ID",
        "section": "Sección",
        "pos_bucket": "Bucket posición",
        "value_gain_to_top3": "Valor Top3 (€)",
        "traffic_est": "Tráfico estimado",
        "best_comp_domain": "Competidor",
        "best_comp_pos": "Posición competidor",
        "traffic_gain_to_top3": "Ganancia Top3 (clics)",
        "Gain Top3": "Ganancia Top3 (clics)",
        "€ Gain Top3": "Valor Top3 (€)",
        "Vol": "Volumen",
        "Pos": "Posición",
        "Pos comp": "Posición competidor",
    }
    if primary:
        ui_map[f"Posición en Google {primary}"] = "Posición"
        ui_map[f"Visibilidad {primary}"] = "Visibilidad"
        ui_map[f"Visibilidad 100% {primary}"] = "Visibilidad (100%)"
        ui_map[f"Google URL encontrada {primary}"] = "URL"
    return ui_map


def humanize_df(df: pd.DataFrame, primary_domain: str = "") -> pd.DataFrame:
    """Make dataframe headers human-friendly for UI rendering."""
    if df is None:
        return df
    normalized = normalize_columns(df)
    ui_map = build_ui_cols_map(primary_domain)
    renamed = normalized.rename(columns={k: v for k, v in ui_map.items() if k in normalized.columns})
    # Drop duplicated columns after renaming to keep Streamlit happy
    return renamed.loc[:, ~renamed.columns.duplicated()].copy()


def build_scatter_tooltips(df_ui: pd.DataFrame) -> Tuple[List[str], str | None]:
    """Return (custom_data columns, hovertemplate) for Plotly scatter with human labels."""
    if df_ui is None:
        return ([], None)
    order = [
        "Keyword",
        "Grupo",
        "Posición",
        "Volumen",
        "KD",
        "URL",
        "Competidor",
        "Posición competidor",
        "Ganancia Top3 (clics)",
    ]
    fmt_map = {
        "Posición": ":.0f",
        "Volumen": ":,.0f",
        "KD": ":.1f",
        "CPC (€)": ":,.2f",
        "SoV %": ":.1f",
        "Visibilidad": ":.1f",
        "Visibilidad (100%)": ":.1f",
        "Ganancia Top3 (clics)": ":,.0f",
    }
    available = [c for c in order if c in df_ui.columns]
    lines = []
    for idx, col in enumerate(available):
        fmt = fmt_map.get(col, "")
        lines.append(f"{col}: %{{customdata[{idx}]{fmt}}}")
    hovertemplate = "<br>".join(lines) + "<extra></extra>" if lines else None
    return (available, hovertemplate)


def build_column_config(df_ui: pd.DataFrame) -> Dict[str, Any]:
    """Construct Streamlit column configs for consistent numeric formatting."""
    if df_ui is None:
        return {}
    if isinstance(df_ui, pd.Series):
        df_ui = df_ui.to_frame()
    cols = set(df_ui.columns)
    cfg: Dict[str, Any] = {}
    if "URL" in cols:
        cfg["URL"] = st.column_config.LinkColumn("URL", display_text="Abrir")
    if "Posición" in cols:
        cfg["Posición"] = st.column_config.NumberColumn("Posición", format="%d")
    if "Volumen" in cols:
        cfg["Volumen"] = st.column_config.NumberColumn("Volumen", format="%d")
    if "KD" in cols:
        cfg["KD"] = st.column_config.NumberColumn("KD", format="%.1f")
    if "CPC (€)" in cols:
        cfg["CPC (€)"] = st.column_config.NumberColumn("CPC (€)", format="€%.2f")
    if "SoV %" in cols:
        cfg["SoV %"] = st.column_config.NumberColumn("SoV %", format="%.1f%%")
    for c in cols:
        if c.lower().startswith("visibilidad"):
            cfg[c] = st.column_config.NumberColumn(c, format="%.1f%%")
    if "Ganancia Top3 (clics)" in cols:
        cfg["Ganancia Top3 (clics)"] = st.column_config.NumberColumn("Ganancia Top3 (clics)", format="%d")
    if "Posición competidor" in cols:
        cfg["Posición competidor"] = st.column_config.NumberColumn("Posición competidor", format="%d")
    if "Valor Top3 (€)" in cols:
        cfg["Valor Top3 (€)"] = st.column_config.NumberColumn("Valor Top3 (€)", format="€%.0f")
    return cfg


def to_number(s: pd.Series) -> pd.Series:
    """Robust numeric coercion for UI columns (handles €/%/thousands/commas/N/D)."""
    if s is None:
        return pd.Series(dtype=float)
    if not isinstance(s, pd.Series):
        s = pd.Series(s)

    cleaned = []
    for val in s.astype(str):
        v = val.strip().replace("\u00a0", "")
        for rm in ["raw::", "norm::", "€", "%"]:
            v = v.replace(rm, "")
        if v in ("", "N/D", "—", "nan", "None"):
            cleaned.append(np.nan)
            continue

        has_comma = "," in v
        has_dot = "." in v
        if has_comma and has_dot:
            # assume dot thousands, comma decimal
            v = v.replace(".", "").replace(",", ".")
        elif has_comma:
            # comma decimal
            v = v.replace(",", ".")
        else:
            # keep dot decimal as-is
            v = v
        cleaned.append(v)

    out = pd.to_numeric(cleaned, errors="coerce")
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=s.index)
    else:
        out.index = s.index
    return out


def coerce_numeric_for_display(df_ui: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric-like UI columns to numbers to avoid Streamlit formatting warnings."""
    if df_ui is None:
        return df_ui
    out = df_ui.copy()
    numeric_cols = {
        "Posición",
        "Volumen",
        "KD",
        "CPC (€)",
        "Ganancia Top3 (clics)",
        "Valor Top3 (€)",
        "Posición competidor",
        "SoV %",
    }
    for col in out.columns:
        if col in numeric_cols or col.lower().startswith("visibilidad"):
            out[col] = to_number(out[col])
    return out
