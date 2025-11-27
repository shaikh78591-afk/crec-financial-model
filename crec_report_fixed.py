import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy_financial as npf
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    ListFlowable,
    ListItem,
)
import io
import os
import re
import html
from datetime import datetime
import numpy as np

st.set_page_config(page_title="CREC Financial Digital Twin", layout="wide")

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - optional dependency
    TfidfVectorizer = None
    cosine_similarity = None


LLM_MODEL = st.secrets.get("OPENAI_CHAT_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
MAX_CONTEXT_CHARS = 6000


def create_openai_client():
    if OpenAI is None:
        return None

    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not api_key:
        return None

    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


OPENAI_CLIENT = create_openai_client()


def trigger_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def dataframe_to_chunks(df: pd.DataFrame, sheet_name: str, max_chars: int = 900, max_rows: int = 40):
    df = df.copy().fillna('')
    df.columns = [str(col) for col in df.columns]
    chunks = []
    current_lines = []
    current_length = 0
    start_idx = None
    last_idx = None

    for idx, row in df.reset_index(drop=True).iterrows():
        row_values = []
        for col, value in row.items():
            value_str = str(value).strip()
            if not value_str or value_str.lower() == 'nan':
                continue
            row_values.append(f"{col}: {value_str}")
        if not row_values:
            continue

        line = f"Row {idx}: " + "; ".join(row_values)
        projected_length = current_length + len(line)
        if current_lines and (projected_length > max_chars or len(current_lines) >= max_rows):
            end_idx = last_idx if last_idx is not None else idx
            chunks.append({
                'text': "\n".join(current_lines),
                'source': f"{sheet_name} rows {start_idx}-{end_idx}"
            })
            current_lines = []
            current_length = 0
            start_idx = None

        if not current_lines:
            start_idx = idx

        current_lines.append(line)
        current_length += len(line)
        last_idx = idx

    if current_lines:
        end_idx = last_idx if last_idx is not None else start_idx
        chunks.append({
            'text': "\n".join(current_lines),
            'source': f"{sheet_name} rows {start_idx}-{end_idx}"
        })

    if not chunks:
        chunks.append({'text': f"No structured rows found in {sheet_name}.", 'source': sheet_name})

    return chunks


def build_rag_index(dataframes: dict):
    documents = []
    for sheet_name, df in dataframes.items():
        if isinstance(df, pd.DataFrame):
            documents.extend(dataframe_to_chunks(df, sheet_name))

    texts = [doc['text'] for doc in documents]
    if not texts:
        return {'vectorizer': None, 'matrix': None, 'documents': documents}

    if TfidfVectorizer and cosine_similarity:
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            matrix = vectorizer.fit_transform(texts)
            return {'vectorizer': vectorizer, 'matrix': matrix, 'documents': documents}
        except Exception:
            pass

    return {'vectorizer': None, 'matrix': None, 'documents': documents}


def search_documents(index_data, query: str, top_k: int = 3):
    if not index_data or not query:
        return []

    documents = index_data['documents']
    if not documents:
        return []

    if index_data['vectorizer'] is not None and index_data['matrix'] is not None:
        query_vec = index_data['vectorizer'].transform([query])
        scores = cosine_similarity(query_vec, index_data['matrix']).flatten()
        ranked_indices = scores.argsort()[::-1]
        results = []
        for idx in ranked_indices[:top_k]:
            score = float(scores[idx])
            if score <= 0:
                continue
            doc = documents[idx]
            results.append({'score': score, 'text': doc['text'], 'source': doc['source']})
        return results

    query_tokens = set(query.lower().split())
    scored = []
    for doc in documents:
        doc_tokens = set(doc['text'].lower().split())
        overlap = len(query_tokens & doc_tokens)
        if overlap == 0:
            continue
        scored.append({'score': overlap / (len(query_tokens) + 1e-6), 'text': doc['text'], 'source': doc['source']})

    scored.sort(key=lambda item: item['score'], reverse=True)
    return scored[:top_k]


def call_llm_with_context(context: str, question: str):
    if not OPENAI_CLIENT:
        return None

    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial analyst assistant. Use ONLY the provided context to answer the question. "
                "If the context does not contain the answer, say you do not know. "
                "Do not reference UI sections or controls; respond with data-driven insights only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n"
                "Answer in a concise paragraph or bullets and cite the specific context rows where possible."
            ),
        },
    ]

    try:
        response = OPENAI_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


UNKNOWN_RESPONSE_PATTERNS = (
    "i do not know",
    "i don't know",
)


def is_unknown_response(text: str) -> bool:
    if not text:
        return False
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    normalized = re.sub(r"[.!?]+$", "", normalized).strip()
    return any(
        normalized == pattern or normalized.startswith(f"{pattern} ")
        for pattern in UNKNOWN_RESPONSE_PATTERNS
    )

# Enhanced Custom CSS for glassy, modern dark theme with glow effects
# Also load external theme overrides if available
try:
    from pathlib import Path
    theme_path = Path(__file__).parent / "assets" / "theme.css"
    if theme_path.exists():
        st.markdown(f"<style>{theme_path.read_text()}</style>", unsafe_allow_html=True)
except Exception:
    pass
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --app-font: 'Inter', 'Source Sans Pro', 'Helvetica Neue', Arial, sans-serif;
        --heading-font: 'Inter', 'Source Sans Pro', 'Helvetica Neue', Arial, sans-serif;
        --app-body-width: 1180px;
    }
    html, body, .stApp {
        font-family: var(--app-font) !important;
        font-size: 16px;
        line-height: 1.6;
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
        letter-spacing: 0.01em;
    }
    .block-container {
        max-width: var(--app-body-width) !important;
        transition: max-width 0.3s ease;
    }
    @media (max-width: 1400px) {
        .block-container {
            max-width: calc(100% - 3rem) !important;
        }
    }
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6 {
        font-family: var(--heading-font) !important;
        font-weight: 600;
        letter-spacing: -0.01em;
        line-height: 1.15;
    }
    .stApp h1 { font-size: 2rem; }
    .stApp h2 { font-size: 1.6rem; }
    .stApp h3 { font-size: 1.35rem; }
    .stApp h4 { font-size: 1.2rem; }

    .stApp em,
    .stApp i {
        font-style: normal !important;
        font-family: var(--app-font) !important;
    }
    .chart-source {
        font-size: 0.85rem;
        color: rgba(224, 224, 224, 0.7);
        margin-bottom: 0.4rem;
    }
    .block-container {
        padding-top: 0.75rem !important;
        padding-bottom: 1.5rem !important;
    }
    .chart-actions-title {
        font-weight: 600;
        margin-top: 0.6rem;
        margin-bottom: 0.25rem;
    }
    .chart-actions {
        margin-top: 0;
        margin-bottom: 0.6rem;
        padding-left: 1.2rem;
    }
    div[data-testid="stMarkdownContainer"],
    .stAlert {
        font-family: var(--app-font) !important;
    }
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    .stAlert p,
    .stAlert li {
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 0.65rem;
    }
    button,
    input,
    textarea,
    select,
    option,
    div[data-baseweb="input"] *,
    div[data-baseweb="select"] *,
    div[data-baseweb="tag"] * {
        font-family: var(--app-font) !important;
    }
    .stApp { 
        background-color: #121212; 
        color: #e0e0e0; 
    }
    div[data-testid="stToolbar"] {
        padding-top: 0 !important;
    }
    .stSlider { 
        background: linear-gradient(45deg, #024c4c, #069494); 
        border-radius: 12px; 
        padding: 12px; 
        box-shadow: 0 0 15px rgba(6, 148, 148, 0.45); 
    }
    .stSlider [role="slider"] {
        background-color: #9ca3af !important;
        border: 2px solid #4b5563 !important;
        box-shadow: 0 0 10px rgba(148, 163, 184, 0.35);
    }
    .stSlider div[data-testid="stTickBar"] {
        background: linear-gradient(90deg, #4b5563, #9ca3af) !important;
    }
    .stButton { 
        background: linear-gradient(45deg, #035f5f, #069494); 
        color: white; 
        border-radius: 8px; 
        box-shadow: 0 0 10px rgba(6, 148, 148, 0.35); 
        transition: all 0.3s ease; 
    }
    .stButton:hover { 
        box-shadow: 0 0 20px rgba(6, 148, 148, 0.55); 
    }
    .base-overview-list {
        margin: 0.25rem 0 0.6rem;
        padding-left: 1.25rem;
        list-style: disc;
    }
    .base-overview-list li {
        margin-bottom: 0.25rem;
        font-size: 1rem;
        line-height: 1.6;
    }
    button[data-testid="baseButton-secondary"] {
        border-radius: 999px;
        padding: 0.35rem 1.1rem;
        border: 1px solid rgba(6, 148, 148, 0.7);
        background: transparent;
        color: #e0e0e0;
        box-shadow: none;
    }
    button[data-testid="baseButton-secondary"]:hover {
        background: rgba(148, 163, 184, 0.18);
    }
    .plotly-chart { 
        box-shadow: 0 4px 12px rgba(0,0,0,0.6); 
        border-radius: 12px; 
        overflow: hidden; 
    }
    .sidebar .sidebar-content { 
        background: #1e1e1e; 
        box-shadow: 0 0 10px rgba(0,0,0,0.5); 
    }
</style>
""", unsafe_allow_html=True)

components.html(
    """
    <script>
    (function() {
      const w = window.parent;
      if (!w || w.__crecWidthObserver) { return; }
      w.__crecWidthObserver = true;
      const doc = w.document;
      const root = doc.documentElement;
      const update = () => {
        const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
        const collapsed = !sidebar || sidebar.offsetWidth < 36 || sidebar.getAttribute('aria-expanded') === 'false';
        root.style.setProperty('--app-body-width', collapsed ? '1380px' : '1180px');
      };
      update();
      const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
      if (sidebar) {
        const observer = new MutationObserver(update);
        observer.observe(sidebar, { attributes: true, attributeFilter: ['style', 'class', 'aria-expanded'] });
      }
      w.addEventListener('resize', () => window.setTimeout(update, 60));
    })();
    </script>
    """,
    height=0,
    width=0,
)

# Load Excel (using cache_resource for unserializable ExcelFile)
@st.cache_resource
def load_excel():
    return pd.ExcelFile("crec_model.xlsx")

xl = load_excel()

# Parse sheets
data_sources = pd.read_excel(xl, "Data Sources", header=None)
financials_yearly = pd.read_excel(xl, "Financials Yearly", header=1)
summary = pd.read_excel(xl, "Summary")
npv_irr = pd.read_excel(xl, "NPV IRR")
npv_irr_raw = pd.read_excel(xl, "NPV IRR", header=None)
stage1 = pd.read_excel(xl, "Stage 1 Mass Balance")
stage2 = pd.read_excel(xl, "Stage 2 Mass Balance")
stage3_power = pd.read_excel(xl, "Stage 3 Power Opt")
plant_salaries = pd.read_excel(xl, "Plant Salaries", header=2)
plant_salaries.columns = [str(col).replace("\n", " ").strip() for col in plant_salaries.columns]
capex = pd.read_excel(xl, "CapEx", header=None)
power_production = pd.read_excel(xl, "Power Production")
emp_labor = pd.read_excel(xl, "Emp Labor Cost", header=4)
emp_labor.columns = [str(col).replace("\n", " ").strip() for col in emp_labor.columns]

# Build RAG index once per session
rag_inputs = {
    "Data Sources": data_sources,
    "Financials Yearly": financials_yearly,
    "Summary": summary,
    "NPV IRR": npv_irr,
    "Stage 1 Mass Balance": stage1,
    "Stage 2 Mass Balance": stage2,
    "Stage 3 Power Opt": stage3_power,
    "Plant Salaries": plant_salaries,
    "CapEx": capex,
    "Power Production": power_production,
    "Emp Labor Cost": emp_labor,
}

if 'rag_index_data' not in st.session_state:
    st.session_state['rag_index_data'] = build_rag_index(rag_inputs)

# ---- Data Sources: parse three switches + related params if present ----
def _norm_key(text):
    if not isinstance(text, str):
        text = str(text)
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def _bool_from_cell(val) -> bool | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (int, float)):
        return bool(val != 0)
    s = str(val).strip().lower()
    if s in ("on", "yes", "true", "enabled", "enable", "y", "1"):
        return True
    if s in ("off", "no", "false", "disabled", "disable", "n", "0"):
        return False
    return None


def parse_data_sources_switches_and_params(df: pd.DataFrame):
    switches = {'graphene': None, 'rng': None, 'tires_only': None}
    params = {}
    if df is None or df.empty:
        return switches, params
    for _, row in df.iterrows():
        key_raw = row.iloc[0] if len(row) > 0 else None
        # Many models use column C for values; prefer index 2, fallback to 1
        val_raw = None
        if len(row) > 2 and pd.notna(row.iloc[2]):
            val_raw = row.iloc[2]
        elif len(row) > 1:
            val_raw = row.iloc[1]
        if pd.isna(key_raw):
            continue
        key = _norm_key(key_raw)
        # capture three switches by fuzzy matching
        if "graphene" in key and "switch" in key:
            b = _bool_from_cell(val_raw)
            if b is not None:
                switches['graphene'] = b
            continue
        if ("rng" in key or "renewable_natural_gas" in key) and "switch" in key:
            b = _bool_from_cell(val_raw)
            if b is not None:
                switches['rng'] = b
            continue
        if ("tires" in key and "only" in key) and "switch" in key:
            b = _bool_from_cell(val_raw)
            if b is not None:
                switches['tires_only'] = b
            continue
        # store potential pricing/yield params to enable richer modeling if present
        if any(tok in key for tok in ("graphene", "rng")):
            try:
                params[key] = float(val_raw)
            except Exception:
                params[key] = val_raw
    return switches, params


if ('switch_graphene' not in st.session_state
    or 'switch_rng' not in st.session_state
    or 'switch_tires_only' not in st.session_state
    or 'ds_params' not in st.session_state):
    sw, pm = parse_data_sources_switches_and_params(data_sources)
    st.session_state['switch_graphene'] = bool(sw.get('graphene')) if sw.get('graphene') is not None else False
    st.session_state['switch_rng'] = bool(sw.get('rng')) if sw.get('rng') is not None else False
    st.session_state['switch_tires_only'] = bool(sw.get('tires_only')) if sw.get('tires_only') is not None else False
    st.session_state['ds_params'] = pm

# Base metrics with dynamic row selection
item_col = financials_yearly.columns[1]
years = [col for col in financials_yearly.columns if isinstance(col, (int, float)) and 2025 <= int(col) <= 2047]
if not years:
    st.error("Error: Year columns 2025-2047 not found in 'Financials Yearly' sheet. Please check the sheet structure.")
    st.stop()

def get_numeric_row(label_substring):
    mask = financials_yearly[item_col].astype(str).str.contains(label_substring, case=False, na=False)
    if not mask.any():
        st.error(f"Error: '{label_substring}' row not found in 'Financials Yearly' sheet. Please check the sheet structure.")
        st.stop()
    row = financials_yearly.loc[mask, years].iloc[0]
    numeric_row = pd.to_numeric(row, errors='coerce').astype(float)
    if np.any(np.isnan(numeric_row)):
        st.error(f"Error: Non-numeric or missing values detected in '{label_substring}' row. Please ensure the data is complete.")
        st.stop()
    return numeric_row.values

base_revenue = get_numeric_row("Total Revenue")
base_expenses = get_numeric_row("Total Expenses")

# Find "Total Project Cash Flow" row dynamically
base_cf = get_numeric_row("Total Project Cash Flow")

def find_row_index(raw_df, label):
    col = raw_df.iloc[:, 1].astype(str)
    matches = col[col.str.contains(label, case=False, na=False)]
    return matches.index[0] if not matches.empty else None


def get_scalar_below(raw_df, label):
    idx = find_row_index(raw_df, label)
    if idx is None:
        return float('nan')
    label_row = raw_df.iloc[idx]
    label_col = label_row[label_row.astype(str).str.contains(label, case=False, na=False)].index[0]
    # search downward for first numeric value in same column
    for offset in (1, 0, 2):
        if idx + offset < raw_df.shape[0]:
            value = pd.to_numeric(raw_df.iat[idx + offset, label_col], errors='coerce')
            if pd.notna(value):
                return float(value)
    return float('nan')


def get_series_by_label(raw_df, label):
    idx = find_row_index(raw_df, label)
    if idx is None:
        return []
    row = raw_df.iloc[idx, 2:]
    return pd.to_numeric(row, errors='coerce').dropna().tolist()


def get_dated_series(raw_df, label):
    values = get_series_by_label(raw_df, label)
    if not values:
        return {}
    # locate the nearest row containing datetime stamps (same block structure)
    base_idx = find_row_index(raw_df, label)
    window_start = max(base_idx - 15, 0)
    window_end = base_idx + 15
    date_row = raw_df.iloc[window_start:window_end]
    date_series = None
    for _, candidate in date_row.iterrows():
        converted = pd.to_datetime(candidate.iloc[2:], errors='coerce')
        valid = converted.dropna()
        if valid.size >= len(values):
            if valid.dt.year.between(1900, 2100).mean() >= 0.8 and valid.dt.month.nunique() > 1:
                date_series = converted
                break
    if date_series is None:
        return {}
    records = {}
    for idx_val, cash_val in enumerate(values):
        if idx_val >= len(date_series):
            break
        date_val = date_series.iloc[idx_val]
        if pd.isna(date_val):
            continue
        records.setdefault(int(date_val.year), 0.0)
        records[int(date_val.year)] += float(cash_val)
    return records


def _summary_find_cell(label: str):
    target = label.strip().lower()
    for row_idx in range(summary.shape[0]):
        for col_idx in range(summary.shape[1]):
            cell = summary.iat[row_idx, col_idx]
            if isinstance(cell, str) and cell.strip().lower() == target:
                return row_idx, col_idx
    return None, None


summary_year_columns = []
for col_idx, value in enumerate(summary.iloc[0]):
    if pd.notna(value):
        try:
            summary_year_columns.append((col_idx, int(value)))
        except Exception:
            continue


def extract_summary_series(label: str) -> dict:
    anchor = _summary_find_cell(label)
    if anchor == (None, None):
        return {}
    row_idx, _ = anchor
    series = {}
    for col_idx, year in summary_year_columns:
        try:
            value = pd.to_numeric(summary.iat[row_idx, col_idx], errors='coerce')
        except Exception:
            value = float('nan')
        if pd.notna(value):
            series[int(year)] = float(value)
    return series


def extract_summary_scalar(label: str) -> float:
    anchor = _summary_find_cell(label)
    if anchor == (None, None):
        return float('nan')
    row_idx, col_idx = anchor
    for offset in range(1, 5):
        if col_idx + offset < summary.shape[1]:
            value = pd.to_numeric(summary.iat[row_idx, col_idx + offset], errors='coerce')
            if pd.notna(value):
                return float(value)
    for offset in range(1, 5):
        if row_idx + offset < summary.shape[0]:
            value = pd.to_numeric(summary.iat[row_idx + offset, col_idx], errors='coerce')
            if pd.notna(value):
                return float(value)
    return float('nan')


def align_summary_series(series_dict: dict, target_years: list) -> np.ndarray:
    if not series_dict:
        return np.array([], dtype=float)
    aligned = []
    for year in target_years:
        try:
            year_key = int(str(year))
        except Exception:
            year_key = year
        aligned.append(float(series_dict.get(year_key, float('nan'))))
    return np.array(aligned, dtype=float)


base_discount = get_scalar_below(npv_irr_raw, 'WACC')
base_equity_irr = get_scalar_below(npv_irr_raw, 'Total IRR %') * 100
base_project_irr = get_scalar_below(npv_irr_raw, 'Total Project IRR %') * 100
base_npv = get_scalar_below(npv_irr_raw, 'Net Present Value')
base_capex = float('nan')

project_cash_flow_by_year = get_dated_series(npv_irr_raw, 'Total Project Cash Flow')
base_cash_flows_full = []
if project_cash_flow_by_year:
    base_cash_flows_full = [project_cash_flow_by_year.get(int(year), float('nan')) for year in years]
    if not any(np.isfinite(base_cash_flows_full)):
        base_cash_flows_full = []

if not base_cash_flows_full:
    base_cash_flows_full = [float(base_capex)] + list(np.array(base_cf, dtype=float)) if 'base_cf' in locals() else []

base_capex = float(base_cash_flows_full[0]) if base_cash_flows_full else float('nan')
base_cash_flow_series = base_cash_flows_full[1:] if len(base_cash_flows_full) > 1 else []

if (not np.isfinite(base_capex)) or base_capex == 0:
    capital_contrib_by_year = get_dated_series(npv_irr_raw, 'Total Capital Contributions')
    if capital_contrib_by_year:
        base_capex = float(capital_contrib_by_year.get(int(years[0]), base_capex))
        if base_cash_flows_full:
            base_cash_flows_full[0] = base_capex

# Fallback discount if Excel did not specify
if not np.isfinite(base_discount) or base_discount == 0:
    base_discount = 0.08

base_irr = base_project_irr

BASE_TOTAL_REVENUE = float(np.nansum(base_revenue)) if getattr(base_revenue, "size", 0) else float('nan')

# Portion of operating expenses that scale with throughput (remainder treated as fixed)
EXPENSE_VARIABLE_SHARE = 0.7


summary_series_raw = {
    'revenue': extract_summary_series('Revenue'),
    'ebitda': extract_summary_series('EBITDA'),
    'net_income': extract_summary_series('Net Income'),
    'project_cash_flow': extract_summary_series('Project Cash Flow'),
}

summary_series_aligned = {
    key: align_summary_series(series, years)
    for key, series in summary_series_raw.items()
}

summary_scalar_highlights = {
    'total_equity_contributed': extract_summary_scalar('Total Equity Contributed'),
    'total_project_cash_flow': extract_summary_scalar('Total Project Cash Flow'),
    'total_project_irr_pct': extract_summary_scalar('Total Project IRR %') * 100,
    'total_equity_irr_pct': extract_summary_scalar('Total Project Equity IRR %') * 100,
    'net_present_value': extract_summary_scalar('Net Present Value'),
    'loan_to_value': extract_summary_scalar('Loan-to-Value'),
}


def compute_scenario_metrics(data: dict, capex_value: float):
    revenue = np.array(data['revenue'], dtype=float)
    expenses = np.array(data['expenses'], dtype=float)
    cash_flows_full = np.array(data.get('cash_flows_full', []), dtype=float)
    if cash_flows_full.size > 0:
        capex_value = float(cash_flows_full[0])
        cash_flows = np.array(cash_flows_full[1:], dtype=float)
    else:
        cash_flows = np.array(data['cf'], dtype=float)
    total_revenue = float(np.nansum(revenue))
    total_expenses = float(np.nansum(expenses))
    operating_margin = float((total_revenue - total_expenses) / total_revenue) if total_revenue else float('nan')
    cumulative = np.cumsum(np.concatenate([[capex_value], cash_flows]))
    payback_year_value = float('nan')
    payback_year_label = 'N/A'
    payback_index = np.where(cumulative >= 0)[0]
    if payback_index.size > 0 and payback_index[0] > 0:
        raw_year = years[payback_index[0] - 1]
        try:
            payback_year_value = float(raw_year)
            payback_year_label = str(int(payback_year_value)) if payback_year_value.is_integer() else str(payback_year_value)
        except Exception:
            payback_year_label = str(raw_year)

    return {
        'irr': float(data.get('irr', np.nan)),
        'npv': float(data.get('npv', np.nan)),
        'capex': float(capex_value),
        'total_revenue': total_revenue,
        'total_expenses': total_expenses,
        'margin': operating_margin,
        'total_cf': float(np.nansum(cash_flows)),
        'avg_cf': float(np.nanmean(cash_flows)) if cash_flows.size else float('nan'),
        'peak_cf': float(np.nanmax(cash_flows)) if cash_flows.size else float('nan'),
        'min_cf': float(np.nanmin(cash_flows)) if cash_flows.size else float('nan'),
        'payback_year': payback_year_value,
        'payback_year_label': payback_year_label,
        'cash_on_cash': float(np.nansum(cash_flows) / abs(capex_value)) if capex_value else float('nan'),
    }


def evaluate_custom_kpi(formula: str, metrics: dict):
    safe_locals = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not np.isnan(v)}
    safe_locals.update(metrics)
    safe_locals.update({
        'abs': abs,
        'max': max,
        'min': min,
        'round': round,
        'np': np,
    })
    return eval(formula, {"__builtins__": {}}, safe_locals)


def format_metric_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    try:
        if np.isnan(value):
            return "N/A"
    except Exception:
        pass
    if isinstance(value, (int, float)):
        magnitude = abs(value)
        if magnitude >= 1_000_000_000:
            return f"${value/1_000_000_000:.2f}B"
        if magnitude >= 1_000_000:
            return f"${value/1_000_000:.2f}M"
        if magnitude >= 1_000:
            return f"${value:,.0f}"
        if magnitude >= 1:
            return f"${value:,.2f}"
        return f"{value:.4f}"
    return str(value)


def format_currency_delta(value):
    if value is None:
        return "N/A"
    try:
        if np.isnan(value):
            return "N/A"
    except TypeError:
        pass
    sign = "+"
    if value < 0:
        sign = "-"
    elif abs(value) < 1e-9:
        sign = "±"
    magnitude = abs(value)
    if magnitude < 1e-9:
        formatted = "$0"
    else:
        formatted = format_metric_value(magnitude)
        if not formatted.startswith("$"):
            if magnitude >= 1:
                formatted = f"${magnitude:,.2f}"
            else:
                formatted = f"${magnitude:.4f}"
    return f"{sign}{formatted}" if sign in ("+", "-", "±") else formatted


def format_ratio_delta(value):
    if value is None:
        return "N/A"
    try:
        if np.isnan(value):
            return "N/A"
    except TypeError:
        pass
    sign = "+" if value > 0 else ("-" if value < 0 else "±")
    return f"{sign}{abs(value):.2f}x"


def format_percent_delta(value):
    if value is None:
        return "N/A"
    try:
        if np.isnan(value):
            return "N/A"
    except TypeError:
        pass
    sign = "+" if value > 0 else ("-" if value < 0 else "±")
    return f"{sign}{abs(value):.2f} pts"


def safe_percent(value):
    if value is None:
        return "N/A"
    try:
        if np.isnan(value):
            return "N/A"
    except TypeError:
        pass
    return f"{value:.2f}%"


def format_rate(value):
    if value is None:
        return "N/A"
    try:
        if np.isnan(value):
            return "N/A"
    except TypeError:
        pass
    return f"{value * 100:.2f}%" if value < 1 else f"{value:.2f}%"


def estimate_equity_irr(project_irr_value):
    base_equity_value = base_equity_irr
    if not np.isfinite(base_equity_value):
        base_equity_value = summary_scalar_highlights.get('total_equity_irr_pct', float('nan'))
    if (
        np.isfinite(project_irr_value)
        and np.isfinite(base_equity_value)
        and np.isfinite(base_project_irr)
    ):
        return float(base_equity_value + (project_irr_value - base_project_irr))
    return float(base_equity_value) if np.isfinite(base_equity_value) else float('nan')


PARAM_CHANGE_KEYS = ('msw_tpd', 'tires_tpd', 'power_price', 'inflation', 'capex_mult')
PARAM_HISTORY_LIMIT = 12


def build_parameter_change_entry(params: dict, base_params: dict, sim_data: dict):
    if not params or not sim_data:
        return None
    base_params = base_params or {}

    def _value_or_nan(key):
        try:
            return float(params.get(key, float('nan')))
        except Exception:
            return float('nan')

    def _base_or_nan(key):
        try:
            return float(base_params.get(key, float('nan')))
        except Exception:
            return float('nan')

    msw_val = _value_or_nan('msw_tpd')
    tires_val = _value_or_nan('tires_tpd')
    price_val = _value_or_nan('power_price')
    inflation_val = _value_or_nan('inflation')
    capex_mult_val = _value_or_nan('capex_mult')

    msw_base = _base_or_nan('msw_tpd')
    tires_base = _base_or_nan('tires_tpd')
    price_base = _base_or_nan('power_price')
    inflation_base = _base_or_nan('inflation')
    capex_mult_base = _base_or_nan('capex_mult')

    param_parts = []
    if np.isfinite(msw_val):
        delta = msw_val - msw_base if np.isfinite(msw_base) else float('nan')
        part = f"MSW {msw_val:.0f} TPD"
        if np.isfinite(delta) and abs(delta) > 0.5:
            part += f" ({delta:+.0f})"
        param_parts.append(part)
    if np.isfinite(tires_val):
        delta = tires_val - tires_base if np.isfinite(tires_base) else float('nan')
        part = f"Tires {tires_val:.0f} TPD"
        if np.isfinite(delta) and abs(delta) > 0.5:
            part += f" ({delta:+.0f})"
        param_parts.append(part)
    if np.isfinite(price_val):
        delta = price_val - price_base if np.isfinite(price_base) else float('nan')
        part = f"Power ${price_val:.3f}/kWh"
        if np.isfinite(delta) and abs(delta) > 5e-4:
            part += f" ({delta:+.3f})"
        param_parts.append(part)
    if np.isfinite(inflation_val):
        delta = inflation_val - inflation_base if np.isfinite(inflation_base) else float('nan')
        part = f"Inflation {inflation_val*100:.2f}%"
        if np.isfinite(delta) and abs(delta) > 5e-4:
            part += f" ({delta*100:+.2f} pts)"
        param_parts.append(part)
    if np.isfinite(capex_mult_val):
        delta = capex_mult_val - capex_mult_base if np.isfinite(capex_mult_base) else float('nan')
        part = f"CapEx Mult {capex_mult_val:.2f}x"
        if np.isfinite(delta) and abs(delta) > 5e-3:
            part += f" ({delta:+.2f})"
        param_parts.append(part)

    if not param_parts:
        param_parts.append("Inputs match base case sliders.")

    project_irr_value = float(sim_data.get('irr', float('nan')))
    equity_irr_value = float(sim_data.get('equity_irr', float('nan')))
    npv_value = float(sim_data.get('npv', float('nan')))
    revenue_series = np.array(sim_data.get('revenue', []), dtype=float)
    total_revenue_value = float(np.nansum(revenue_series)) if revenue_series.size else float('nan')

    base_equity_value = base_equity_irr
    if not np.isfinite(base_equity_value):
        base_equity_value = summary_scalar_highlights.get('total_equity_irr_pct', float('nan'))

    metrics_parts = []
    if np.isfinite(project_irr_value) and np.isfinite(base_project_irr):
        delta = project_irr_value - base_project_irr
        metrics_parts.append(f"Project IRR {project_irr_value:.2f}% ({delta:+.2f} pts)")
    elif np.isfinite(project_irr_value):
        metrics_parts.append(f"Project IRR {project_irr_value:.2f}%")

    if np.isfinite(equity_irr_value) and np.isfinite(base_equity_value):
        delta = equity_irr_value - base_equity_value
        metrics_parts.append(f"Equity IRR {equity_irr_value:.2f}% ({delta:+.2f} pts)")
    elif np.isfinite(equity_irr_value):
        metrics_parts.append(f"Equity IRR {equity_irr_value:.2f}%")

    if np.isfinite(npv_value):
        npv_delta = npv_value - base_npv if np.isfinite(base_npv) else float('nan')
        delta_text = format_currency_delta(npv_delta) if np.isfinite(npv_delta) else ""
        metrics_parts.append(
            f"NPV {format_metric_value(npv_value)}" + (f" ({delta_text})" if delta_text else "")
        )

    if np.isfinite(total_revenue_value):
        rev_delta = total_revenue_value - BASE_TOTAL_REVENUE if np.isfinite(BASE_TOTAL_REVENUE) else float('nan')
        delta_text = format_currency_delta(rev_delta) if np.isfinite(rev_delta) else ""
        metrics_parts.append(
            f"Total Revenue {format_metric_value(total_revenue_value)}" + (f" ({delta_text})" if delta_text else "")
        )

    summary_parts = ["Inputs: " + ", ".join(param_parts)]
    if metrics_parts:
        summary_parts.append("Metrics vs base: " + "; ".join(metrics_parts))

    timestamp = datetime.now().strftime("%H:%M:%S")
    return {
        'timestamp': timestamp,
        'summary': " | ".join(summary_parts),
        'params': {key: params.get(key) for key in PARAM_CHANGE_KEYS},
    }


def record_parameter_change(params: dict, base_params: dict, sim_data: dict):
    entry = build_parameter_change_entry(params, base_params, sim_data)
    if not entry:
        return
    key = tuple(round(float(params.get(k, float('nan'))), 6) for k in PARAM_CHANGE_KEYS)
    if st.session_state.get('last_param_key') == key:
        return
    history = st.session_state.setdefault('param_change_history', [])
    history.append(entry)
    st.session_state['param_change_history'] = history[-PARAM_HISTORY_LIMIT:]
    st.session_state['last_param_key'] = key


PAGE_MARGIN = 50


def _array_to_list(values) -> list:
    if values is None:
        return []
    arr = np.array(values, dtype=float)
    if arr.ndim == 0:
        return [float(arr)]
    return [float(item) for item in arr]


def serialize_scenario_state(sim_data: dict) -> dict:
    data = sim_data or {}
    serialized = {
        'revenue': _array_to_list(data.get('revenue')),
        'expenses': _array_to_list(data.get('expenses')),
        'cf': _array_to_list(data.get('cf')),
        'cash_flows_full': _array_to_list(data.get('cash_flows_full')),
        'irr': float(data.get('irr')) if data.get('irr') is not None else float('nan'),
        'npv': float(data.get('npv')) if data.get('npv') is not None else float('nan'),
        'capex': float(data.get('capex')) if data.get('capex') is not None else float('nan'),
        'msw_tpd': float(data.get('msw_tpd')) if data.get('msw_tpd') is not None else float('nan'),
        'tires_tpd': float(data.get('tires_tpd')) if data.get('tires_tpd') is not None else float('nan'),
        'power_price': float(data.get('power_price')) if data.get('power_price') is not None else float('nan'),
        'inflation': float(data.get('inflation')) if data.get('inflation') is not None else float('nan'),
        'capex_mult': float(data.get('capex_mult')) if data.get('capex_mult') is not None else float('nan'),
        'equity_irr': float(data.get('equity_irr')) if data.get('equity_irr') is not None else float('nan'),
        'rdf_yield': float(data.get('rdf_yield')) if data.get('rdf_yield') is not None else float('nan'),
        'tdf_yield': float(data.get('tdf_yield')) if data.get('tdf_yield') is not None else float('nan'),
        'syngas_yield': float(data.get('syngas_yield')) if data.get('syngas_yield') is not None else float('nan'),
        'power_mwe': float(data.get('power_mwe')) if data.get('power_mwe') is not None else float('nan'),
        'scale': float(data.get('scale')) if data.get('scale') is not None else float('nan'),
    }
    return serialized


APP_KNOWLEDGE_BASE = (
    (
        "Data Sources",
        "Excel workbook `crec_model.xlsx` feeds the app. 'Financials Yearly' supplies revenue/expense projections by year. "
        "'Summary' holds headline totals (revenue, EBITDA, net income, cash flow). "
        "'NPV IRR' provides cash flow streams, total IRR %, equity IRR %, WACC, and capital contribution schedule. "
        "Stage tabs ('Stage 1 Mass Balance', 'Stage 2 Mass Balance', 'Stage 3 Power Opt') define throughput yields; "
        "'Plant Salaries', 'CapEx', 'Power Production', and 'Emp Labor Cost' feed the corresponding charts and metrics."
    ),
    (
        "Scenario Engine",
        "The `build_scenario` routine scales base results using mass-balance yields. "
        "Scale factor = simulated power (TPD inputs mapped through Stage 1) divided by 46 MW design point. "
        "Revenue = base revenue × scale factor × (power price / 0.07) × inflation drift vs 2% baseline. "
        "Expenses blend fixed (30%) and variable (70%) components scaled by throughput and inflation. "
        "Cash flow stream uses scaled revenue minus expenses with CapEx applied in year 0; project IRR/NPV derive from that stream."
    ),
    (
        "Equity IRR Logic",
        "If the Excel sheet supplies an equity IRR, the simulator shifts that value by the delta between simulated project IRR and the Excel project IRR baseline. "
        "When only project IRR is known, the app treats equity IRR as the base equity IRR plus the project IRR delta to keep sponsor returns aligned with headline IRR moves."
    ),
    (
        "Base Highlights & Tuning",
        "The Base Model Highlights section compares scenario totals to Excel summary values. "
        "Quick tuning sliders apply percentage shocks to revenue, expenses, CapEx, and discount rate, recomputing IRR, NPV, cash flow totals, equity contributions, and payback. "
        "Without tuning, metrics fall back to the Excel-driven baseline."
    ),
    (
        "Scenario Metrics",
        "`compute_scenario_metrics` sums revenue/expenses, computes operating margin, cumulative cash, average cash flow, and payback year from the cash stream. "
        "Cash-on-cash = total operating cash divided by absolute CapEx. "
        "Saved scenarios are serialized with revenue, expenses, cash flow arrays, IRR, NPV, yields, and current slider inputs."
    ),
    (
        "Monte Carlo Engine",
        "Monte Carlo sampling perturbs MSW TPD, Tires TPD, and power price using truncated normal draws around the baseline inputs. "
        "Each draw runs through `build_scenario`, captures IRR, NPV, revenue, expenses, average cash flow, cash-on-cash, and payback, producing summary percentiles for risk analysis."
    ),
    (
        "Parameter Ranges & Commands",
        "Sidebar sliders enforce: MSW 400-800 TPD, Tires 50-200 TPD, power price $0.05-$0.15/kWh, inflation 1%-5%, CapEx multiplier 0.8x-1.2x. "
        "Chat commands mirror these ranges (e.g., 'set msw tpd to 650', 'set power price to 0.09'). "
        "The command parser clamps values inside permitted bounds and rounds power price/inflation to three decimals, CapEx multiplier to two decimals."
    ),
    (
        "Visualization Data Flow",
        "Each chart pulls directly from the active scenario set: revenue vs expenses, cumulative cash flow, IRR/NPV bars, sensitivity proxy, revenue donut, mass-balance Sankey, salary pie, CapEx by country, power production profile, heatmap, IRR vs NPV scatter, and cash-flow box plot. "
        "The helper `render_chart_context` logs chart title, data source, narrative, and recommended actions for chat retrieval."
    ),
)


def build_scenario(msw_value, tires_value, price_value, inflation_value, capex_multiplier):
    base_inputs_ref = st.session_state.get('base_inputs_static')
    base_switches_ref = st.session_state.get('base_switches_static')
    if base_inputs_ref and base_switches_ref is not None:
        same_inputs = (
            np.isclose(msw_value, base_inputs_ref['msw_tpd'])
            and np.isclose(tires_value, base_inputs_ref['tires_tpd'])
            and np.isclose(price_value, base_inputs_ref['power_price'])
            and np.isclose(inflation_value, base_inputs_ref['inflation'])
            and np.isclose(capex_multiplier, base_inputs_ref['capex_mult'])
        )
        # Only shortcut to base if switch states also match baseline snapshot
        current_switches = {
            'graphene': bool(st.session_state.get('switch_graphene', False)),
            'rng': bool(st.session_state.get('switch_rng', False)),
            'tires_only': bool(st.session_state.get('switch_tires_only', False)),
        }
        same_switches = (
            current_switches.get('graphene') == base_switches_ref.get('graphene')
            and current_switches.get('rng') == base_switches_ref.get('rng')
            and current_switches.get('tires_only') == base_switches_ref.get('tires_only')
        )
        if same_inputs and same_switches:
            return serialize_scenario_state(base_scenario_raw)

    # Apply switches from Data Sources
    tires_only = bool(st.session_state.get('switch_tires_only', False))
    rng_on = bool(st.session_state.get('switch_rng', False))
    graphene_on = bool(st.session_state.get('switch_graphene', False))
    ds_params = st.session_state.get('ds_params', {}) or {}

    # Excel evidence shows Tires Only switch primarily gates CapEx items; keep MSW throughput unchanged here
    msw_effective = msw_value
    tires_effective = tires_value

    # Raw yields (no switches baked in)
    rdf_yield, tdf_yield, syngas_yield, _ = simulate_mass_balance(msw_effective, tires_effective)
    # Compute power components explicitly
    rdf_power_comp = (syngas_yield / 450.3) * 36.5
    tdf_power_comp = (tdf_yield / 85) * 9.8
    power_mwe = rdf_power_comp + tdf_power_comp
    scale_factor = (power_mwe / 46.0) if 46.0 != 0 else 1.0

    # Inflation as per slider
    inf_factor = np.cumprod(np.full(len(years), 1 + inflation_value)) / np.cumprod(np.full(len(years), 1 + 0.02))
    # RNG switch in Excel zeros the electrical price (Data Sources!C30) rather than removing power;
    # mirror that by zeroing price multiplier when RNG is on.
    effective_price = 0.0 if rng_on else price_value
    sim_revenue = base_revenue * scale_factor * (effective_price / 0.07) * inf_factor

    # Expenses scale by throughput share as before
    expense_scale = (1 - EXPENSE_VARIABLE_SHARE) + EXPENSE_VARIABLE_SHARE * scale_factor
    sim_expenses = base_expenses * expense_scale * inf_factor

    # Optional incremental revenues if params exist in Data Sources
    extra_revenue = np.zeros_like(sim_revenue, dtype=float)
    # Graphene: tires-driven revenue (if params present)
    try:
        g_yield = float(next((v for k, v in ds_params.items() if 'graphene' in k and 'yield' in k), None))
        g_price = float(next((v for k, v in ds_params.items() if 'graphene' in k and 'price' in k), None))
        g_divert = float(next((v for k, v in ds_params.items() if 'graphene' in k and 'divert' in k), 0.0))
        if graphene_on and np.isfinite(g_yield) and np.isfinite(g_price):
            # Yearly graphene revenue ~= tires_tpd * yield_per_ton * price * 365
            graphene_rev_year = float(tires_effective) * g_yield * g_price * 365.0
            extra_revenue = extra_revenue + (graphene_rev_year * (inf_factor / inf_factor[0]))
            # Optional diversion reduces TDF power contribution
            if np.isfinite(g_divert) and 0.0 < g_divert < 1.0:
                tdf_power_comp = tdf_power_comp * (1.0 - g_divert)
                power_mwe = rdf_power_comp + tdf_power_comp
                scale_factor = (power_mwe / 46.0) if 46.0 != 0 else 1.0
                sim_revenue = base_revenue * scale_factor * (price_value / 0.07) * inf_factor
    except Exception:
        pass
    # RNG: syngas-to-RNG revenue if price/yield present
    try:
        r_yield = float(next((v for k, v in ds_params.items() if ('rng' in k) and ('yield' in k)), None))
        r_price = float(next((v for k, v in ds_params.items() if ('rng' in k) and ('price' in k)), None))
        if rng_on and np.isfinite(r_yield) and np.isfinite(r_price):
            rng_rev_year = float(syngas_yield) * r_yield * r_price * 365.0
            extra_revenue = extra_revenue + (rng_rev_year * (inf_factor / inf_factor[0]))
    except Exception:
        pass
    sim_revenue = sim_revenue + extra_revenue

    sim_cf = sim_revenue - sim_expenses
    sim_capex = base_capex * capex_multiplier
    if sim_cf.size > 1:
        sim_cash_flows = np.concatenate([[sim_capex], sim_cf[1:]])
        operating_cf = sim_cf[1:]
    else:
        sim_cash_flows = np.array([sim_capex])
        operating_cf = np.array([])
    sim_irr = npf.irr(sim_cash_flows) * 100 if sim_cash_flows.size > 1 and np.any(sim_cash_flows[1:] > 0) else float('nan')
    sim_npv = npf.npv(base_discount, sim_cash_flows[1:]) + sim_cash_flows[0] if sim_cash_flows.size > 1 else sim_cash_flows[0]
    equity_irr_est = estimate_equity_irr(sim_irr)
    return {
        'revenue': sim_revenue,
        'expenses': sim_expenses,
        'cf': operating_cf,
        'cash_flows_full': sim_cash_flows,
        'irr': sim_irr,
        'npv': sim_npv,
        'equity_irr': equity_irr_est,
        'scale': scale_factor,
        'rdf_yield': rdf_yield,
        'tdf_yield': tdf_yield,
        'syngas_yield': syngas_yield,
        'capex': sim_capex,
        'msw_tpd': msw_effective,
        'tires_tpd': tires_effective,
        'power_price': price_value,
        'inflation': inflation_value,
        'capex_mult': capex_multiplier,
    }


def generate_forecast(series, base_years, horizon_years=5):
    series = np.asarray(series, dtype=float)
    valid_mask = ~np.isnan(series)
    series = series[valid_mask]
    if series.size == 0:
        return [], []
    x = np.arange(series.size)
    if series.size == 1:
        slope = 0.0
        intercept = series[0]
    else:
        slope, intercept = np.polyfit(x, series, 1)
    future_x = np.arange(series.size, series.size + horizon_years)
    forecast_values = intercept + slope * future_x
    try:
        last_year = int(str(base_years[-1]))
    except Exception:
        last_year = len(base_years)
    future_years = [last_year + i + 1 for i in range(horizon_years)]
    return future_years, forecast_values, slope


def format_forecast_summary(metric_name, slope, future_values):
    if slope > 0:
        trend_text = f"an upward trend of {format_metric_value(slope)} per year"
    elif slope < 0:
        trend_text = f"a downward trend of {format_metric_value(abs(slope))} per year"
    else:
        trend_text = "a flat trend"
    final_value = format_metric_value(future_values[-1]) if len(future_values) else "N/A"
    return f"Forecast indicates {trend_text} for {metric_name.lower()}, reaching approximately {final_value} by the end of the horizon."


def summarize_series(values):
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {'mean': float('nan'), 'std': float('nan'), 'p10': float('nan'), 'p50': float('nan'), 'p90': float('nan')}
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if arr.size > 1 else float('nan'),
        'p10': float(np.percentile(arr, 10)),
        'p50': float(np.percentile(arr, 50)),
        'p90': float(np.percentile(arr, 90)),
    }


def run_monte_carlo(iterations, base_inputs, volatilities, inflation_value, capex_multiplier, seed=None):
    rng = np.random.default_rng(seed)

    def sample(base, pct, minimum=None):
        if pct <= 0:
            return np.full(iterations, base, dtype=float)
        sigma = pct / 100.0
        samples = rng.normal(loc=base, scale=abs(base) * sigma, size=iterations)
        lower = base * max(0.0, 1 - 3 * sigma)
        upper = base * (1 + 3 * sigma)
        if minimum is not None:
            lower = max(lower, minimum)
        samples = np.clip(samples, lower, upper)
        return samples

    msw_samples = sample(base_inputs['msw_tpd'], volatilities['msw_tpd'], minimum=1)
    tires_samples = sample(base_inputs['tires_tpd'], volatilities['tires_tpd'], minimum=1)
    price_samples = sample(base_inputs['power_price'], volatilities['power_price'], minimum=0.01)

    irr_values = []
    npv_values = []
    total_revenue_values = []
    total_expense_values = []
    avg_cf_values = []
    cash_on_cash_values = []
    payback_values = []

    for msw_val, tire_val, price_val in zip(msw_samples, tires_samples, price_samples):
        scenario = build_scenario(msw_val, tire_val, price_val, inflation_value, capex_multiplier)
        metrics = compute_scenario_metrics(scenario, scenario['capex'])
        irr_values.append(scenario['irr'])
        npv_values.append(scenario['npv'])
        total_revenue_values.append(metrics['total_revenue'])
        total_expense_values.append(metrics['total_expenses'])
        avg_cf_values.append(metrics['avg_cf'])
        cash_on_cash_values.append(metrics['cash_on_cash'])
        payback_values.append(metrics['payback_year'])

    outputs = {
        'irr': np.array(irr_values),
        'npv': np.array(npv_values),
        'total_revenue': np.array(total_revenue_values),
        'total_expenses': np.array(total_expense_values),
        'avg_cf': np.array(avg_cf_values),
        'cash_on_cash': np.array(cash_on_cash_values),
        'payback_year': np.array(payback_values),
    }

    summary = {
        'IRR (%)': summarize_series(outputs['irr']),
        'NPV ($)': summarize_series(outputs['npv']),
        'Total Revenue ($)': summarize_series(outputs['total_revenue']),
        'Average Annual CF ($)': summarize_series(outputs['avg_cf']),
        'Cash-on-Cash (x)': summarize_series(outputs['cash_on_cash']),
    }

    irr_stats = summary['IRR (%)']
    npv_stats = summary['NPV ($)']
    summary_text = (
        f"IRR mean {irr_stats['mean']:.2f}% (P10 {irr_stats['p10']:.2f}%, P90 {irr_stats['p90']:.2f}%) | "
        f"NPV mean {format_metric_value(npv_stats['mean'])} (P10 {format_metric_value(npv_stats['p10'])}, P90 {format_metric_value(npv_stats['p90'])})"
    )

    return {
        'iterations': iterations,
        'inputs': {
            'msw_tpd': msw_samples,
            'tires_tpd': tires_samples,
            'power_price': price_samples,
        },
        'outputs': outputs,
        'summary': summary,
        'summary_text': summary_text,
        'config': {
            'volatilities': volatilities,
            'inflation': inflation_value,
            'capex_mult': capex_multiplier,
            'base_inputs': base_inputs,
        },
    }


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def generate_recommendations(reference_scenario):
    base_metrics = compute_scenario_metrics(reference_scenario, reference_scenario.get('capex', base_capex))
    base_inputs = {
        'msw_tpd': reference_scenario.get('msw_tpd', msw_tpd),
        'tires_tpd': reference_scenario.get('tires_tpd', tires_tpd),
        'power_price': reference_scenario.get('power_price', power_price),
        'inflation': reference_scenario.get('inflation', inflation),
        'capex_mult': reference_scenario.get('capex_mult', capex_mult),
    }

    suggestions = []
    adjustments = [
        ("Increase MSW by 10%", {'msw_tpd': clamp(base_inputs['msw_tpd'] * 1.10, 400, 800)}),
        ("Increase Tires by 10%", {'tires_tpd': clamp(base_inputs['tires_tpd'] * 1.10, 50, 200)}),
        ("Raise power price by $0.01", {'power_price': clamp(base_inputs['power_price'] + 0.01, 0.05, 0.15)}),
        ("Negotiate CapEx -10%", {'capex_mult': clamp(base_inputs['capex_mult'] * 0.90, 0.8, 1.2)}),
        ("Reduce MSW by 10% (stress)", {'msw_tpd': clamp(base_inputs['msw_tpd'] * 0.90, 400, 800)}),
    ]

    for label, override in adjustments:
        test_inputs = base_inputs.copy()
        test_inputs.update(override)
        scenario = build_scenario(
            test_inputs['msw_tpd'],
            test_inputs['tires_tpd'],
            test_inputs['power_price'],
            test_inputs['inflation'],
            test_inputs['capex_mult'],
        )
        metrics = compute_scenario_metrics(scenario, scenario['capex'])
        irr_delta = metrics['irr'] - base_metrics['irr']
        npv_delta = metrics['npv'] - base_metrics['npv']
        suggestions.append({
            'label': label,
            'scenario': scenario,
            'metrics': metrics,
            'irr_delta': irr_delta,
            'npv_delta': npv_delta,
            'summary': f"IRR {metrics['irr']:.2f}% ({format_percent_delta(irr_delta)}) | NPV {format_metric_value(metrics['npv'])} ({format_currency_delta(npv_delta)})"
        })

    suggestions.sort(key=lambda item: item['irr_delta'], reverse=True)
    return suggestions[:3]


def optimize_scenario(target_irr, sample_count, tolerance, seed=None):
    current_reference = st.session_state.get('sim') or st.session_state['scenarios'].get('Base')
    if current_reference is None:
        current_reference = build_scenario(msw_tpd, tires_tpd, power_price, inflation, capex_mult)

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(sample_count):
        candidate_inputs = {
            'msw_tpd': rng.uniform(400, 800),
            'tires_tpd': rng.uniform(50, 200),
            'power_price': rng.uniform(0.05, 0.15),
            'inflation': clamp(current_reference.get('inflation', inflation), 0.0, 0.2),
            'capex_mult': rng.uniform(0.8, 1.2),
        }
        scenario = build_scenario(
            candidate_inputs['msw_tpd'],
            candidate_inputs['tires_tpd'],
            candidate_inputs['power_price'],
            candidate_inputs['inflation'],
            candidate_inputs['capex_mult'],
        )
        metrics = compute_scenario_metrics(scenario, scenario['capex'])
        irr = metrics['irr']
        meets = irr >= target_irr - tolerance

        record = {
            'scenario': scenario,
            'metrics': metrics,
            'meets_target': meets,
            'irr_gap': abs(target_irr - irr),
            'npv': metrics['npv'],
        }

        if best is None:
            best = record
            continue

        record_gap = record['irr_gap']
        best_gap = best['irr_gap']

        if record_gap < best_gap - 1e-6:
            best = record
            continue

        if abs(record_gap - best_gap) <= 1e-6:
            if record['meets_target'] and not best['meets_target']:
                best = record
            elif record['npv'] > best['npv']:
                best = record

    if best is None:
        best = {
            'scenario': current_reference,
            'metrics': compute_scenario_metrics(current_reference, current_reference.get('capex', base_capex)),
            'meets_target': False,
            'irr_gap': None,
            'npv': current_reference.get('npv', base_npv),
        }

    scenario = best['scenario']
    metrics = best['metrics']
    summary = (
        f"IRR {metrics['irr']:.2f}% vs target {target_irr:.2f}% | NPV {format_metric_value(metrics['npv'])} | "
        f"MSW {scenario['msw_tpd']:.0f} TPD | Tires {scenario['tires_tpd']:.0f} TPD | Power ${scenario['power_price']:.3f}/kWh | CapEx x{scenario['capex_mult']:.2f}"
    )
    best['summary'] = summary
    best['target'] = target_irr
    best['samples'] = sample_count
    best['tolerance'] = tolerance
    best['seed'] = seed
    return best


def generate_optimizer_explanation(opt_result: dict) -> str:
    scenario = opt_result.get('scenario', {}) or {}
    metrics = opt_result.get('metrics', {}) or {}
    target = opt_result.get('target')
    meets = opt_result.get('meets_target', False)
    irr_value = metrics.get('irr')
    npv_value = metrics.get('npv')
    avg_cf_value = metrics.get('avg_cf')
    payback_label = metrics.get('payback_year_label') or metrics.get('payback_year')

    def safe_percent(value):
        try:
            if value is None or np.isnan(value):
                return "N/A"
            return f"{float(value):.2f}%"
        except Exception:
            return "N/A"

    def safe_number(value, decimals=0, suffix=""):
        try:
            numeric = float(value)
            return f"{numeric:,.{decimals}f}{suffix}"
        except Exception:
            return "N/A"

    target_text = safe_percent(target)
    irr_text = safe_percent(irr_value)
    npv_text = format_metric_value(npv_value)
    avg_cf_text = format_metric_value(avg_cf_value)
    msw_text = safe_number(scenario.get('msw_tpd'), 0, " TPD")
    tires_text = safe_number(scenario.get('tires_tpd'), 0, " TPD")
    power_price_raw = scenario.get('power_price')
    try:
        power_price_text = f"${float(power_price_raw):.3f}/kWh"
    except Exception:
        power_price_text = "N/A"
    capex_raw = scenario.get('capex_mult')
    try:
        capex_mult_text = f"x{float(capex_raw):.2f}"
    except Exception:
        capex_mult_text = "N/A"

    status_text = "meets or exceeds" if meets else "falls short of"
    gap = opt_result.get('irr_gap')
    if gap is not None and not isinstance(gap, str):
        try:
            if not np.isnan(gap):
                gap = float(gap)
            else:
                gap = None
        except Exception:
            gap = None
    gap_text = f" (short by {abs(gap):.2f} percentage points)" if gap not in (None, "N/A") and not meets else ""

    lines = []
    lines.append(f"- Target IRR: {target_text}")
    state = "meets or exceeds" if meets else "falls short of"
    lines.append(f"- Scenario IRR: {irr_text} ({state}{gap_text})")
    lines.append(f"- NPV: {npv_text}")
    lines.append(f"- Average annual cash flow: {avg_cf_text}")
    lines.append(f"- Payback: {payback_label if payback_label is not None else 'N/A'}")
    lines.append(f"- Throughput: MSW {msw_text}, Tires {tires_text}")
    lines.append(f"- Power price: {power_price_text}")
    lines.append(f"- CapEx multiplier: {capex_mult_text}")
    header = "Optimized scenario summary"
    return header + "\n" + "\n".join(lines)

# Function for precise mass balance simulation
def simulate_mass_balance(msw_tpd, tires_tpd):
    moisture_loss = 0.2
    if isinstance(stage1_processed, pd.DataFrame) and 'Moisture' in stage1_processed.columns:
        moisture_series = pd.to_numeric(stage1_processed['Moisture'], errors='coerce')
        if moisture_series.notna().any():
            moisture_loss = moisture_series.dropna().mean()
    rdf_yield = msw_tpd * (1 - moisture_loss) * 0.8187
    syngas_yield = rdf_yield * 0.45
    tdf_yield = tires_tpd * 0.85
    power_mwe = ((syngas_yield / 450.3) * 36.5) + ((tdf_yield / 85) * 9.8)
    return rdf_yield, tdf_yield, syngas_yield, power_mwe


# ---- Switch validation helpers ----
def _total_revenue_from_scenario(scn: dict) -> float:
    rev = scn.get('revenue')
    if rev is None:
        return float('nan')
    arr = np.array(rev, dtype=float)
    return float(np.nansum(arr)) if arr.size else float('nan')


def run_switch_validation():
    # Save current switch states
    prev_graphene = bool(st.session_state.get('switch_graphene', False))
    prev_rng = bool(st.session_state.get('switch_rng', False))
    prev_tires_only = bool(st.session_state.get('switch_tires_only', False))
    try:
        msw = float(st.session_state.get('msw_tpd', 550))
        tires = float(st.session_state.get('tires_tpd', 100))
        pp = float(st.session_state.get('power_price', 0.07))
        inf = float(st.session_state.get('inflation', 0.02))
        cx = float(st.session_state.get('capex_mult', 1.0))
    except Exception:
        msw, tires, pp, inf, cx = 550.0, 100.0, 0.07, 0.02, 1.0

    results = []
    def add_result(name, passed, detail):
        results.append({'check': name, 'passed': bool(passed), 'detail': str(detail)})

    # Baseline: all switches off
    st.session_state['switch_graphene'] = False
    st.session_state['switch_rng'] = False
    st.session_state['switch_tires_only'] = False
    base = build_scenario(msw, tires, pp, inf, cx)
    base_rev = _total_revenue_from_scenario(base)
    base_scale = float(base.get('scale', float('nan')))

    # Tires Only: In Excel this gates CapEx; throughput should remain unchanged
    st.session_state['switch_tires_only'] = True
    st.session_state['switch_rng'] = False
    st.session_state['switch_graphene'] = False
    tires_only_scn = build_scenario(msw, tires, pp, inf, cx)
    cond_msw_unchanged = float(tires_only_scn.get('msw_tpd', -1)) == float(msw)
    add_result(
        "Tires Only leaves MSW unchanged (CapEx gating in Excel)",
        cond_msw_unchanged,
        f"msw_tpd={tires_only_scn.get('msw_tpd')} (CapEx gating not modeled in-app)"
    )

    # RNG: Excel zeros electric price; expect revenue drop, scale unchanged
    st.session_state['switch_tires_only'] = False
    st.session_state['switch_rng'] = True
    st.session_state['switch_graphene'] = False
    rng_scn = build_scenario(msw, tires, pp, inf, cx)
    rng_scale = float(rng_scn.get('scale', float('nan')))
    rng_rev = _total_revenue_from_scenario(rng_scn)
    cond_scale_same = (np.isfinite(base_scale) and np.isfinite(rng_scale) and abs(rng_scale - base_scale) < 1e-9)
    cond_rev_lower = (np.isfinite(base_rev) and np.isfinite(rng_rev) and (rng_rev < base_rev - 1e-9))
    add_result(
        "RNG zeros electricity price (revenue drops; scale unchanged)",
        bool(cond_scale_same and cond_rev_lower),
        f"base_scale={base_scale:.4f}, rng_scale={rng_scale:.4f}; base_rev={base_rev:,.0f}, rng_rev={rng_rev:,.0f}"
    )

    # Graphene: if params exist, revenue should increase
    st.session_state['switch_tires_only'] = False
    st.session_state['switch_rng'] = False
    st.session_state['switch_graphene'] = True
    g_scn = build_scenario(msw, tires, pp, inf, cx)
    g_rev = _total_revenue_from_scenario(g_scn)
    ds_params = st.session_state.get('ds_params', {}) or {}
    has_graphene_params = any(('graphene' in k and 'yield' in k) for k in ds_params) and any(('graphene' in k and 'price' in k) for k in ds_params)
    if has_graphene_params and np.isfinite(base_rev) and np.isfinite(g_rev):
        add_result("Graphene adds incremental revenue (with params)", g_rev > base_rev, f"base_rev={base_rev:,.0f}, graphene_rev={g_rev:,.0f}")
    else:
        add_result("Graphene effect skipped (no graphene params found)", True, "Provide graphene_yield and graphene_price in Data Sources to test revenue impact.")

    # Restore states
    st.session_state['switch_graphene'] = prev_graphene
    st.session_state['switch_rng'] = prev_rng
    st.session_state['switch_tires_only'] = prev_tires_only
    return results

# Data Inference Functions
def compute_cagr(data):
    return (data[-1] / data[0]) ** (1 / (len(data) - 1)) - 1 if data[0] != 0 else 0

def compute_correlation(revenue, expenses):
    return np.corrcoef(revenue, expenses)[0, 1]


def align_year_labels(length: int):
    if length == len(years):
        return years
    if length == len(years) - 1:
        return years[1:]
    if length == len(years) + 1:
        return [years[0] - 1] + years
    start_year = years[0]
    return [start_year + idx for idx in range(length)]


def prepare_cash_flow_plot_data(scenario_data: dict):
    cf_full = np.array(scenario_data.get('cash_flows_full', []), dtype=float)
    if cf_full.size:
        x_years = align_year_labels(cf_full.size)
        return x_years, np.cumsum(cf_full)
    cf_series = np.array(scenario_data.get('cf', []), dtype=float)
    capex_val = float(scenario_data.get('capex', base_capex))
    if cf_series.size:
        stream = np.concatenate([[capex_val], cf_series])
    else:
        stream = np.array([capex_val])
    x_years = align_year_labels(stream.size)
    return x_years, np.cumsum(stream)


def generate_insights(data):
    insights = []
    rev_cagr = compute_cagr(data['revenue'])
    exp_cagr = compute_cagr(data['expenses'])
    corr = compute_correlation(data['revenue'], data['expenses'])
    insights.append(f"Revenue CAGR: {rev_cagr*100:.2f}% – Indicates strong growth if >5%.")
    insights.append(f"Expenses CAGR: {exp_cagr*100:.2f}% – Aim to keep below revenue CAGR.")
    insights.append(f"Revenue-Expenses Correlation: {corr:.2f} – High positive suggests scaling costs; optimize for decoupling.")
    cf_series = np.array(data.get('cf', []), dtype=float)
    if cf_series.size:
        cf_years = align_year_labels(cf_series.size)
        max_index = int(np.nanargmax(cf_series))
        insights.append(
            f"Peak Cash Flow in {cf_years[max_index]}: ${cf_series[max_index]:,.0f} – Focus on sustaining post-peak."
        )
    return insights


def render_chart_context(title, source, explanation, actions):
    if not title:
        title = "Untitled Chart"
    catalog = st.session_state.setdefault('chart_context_catalog', {})
    catalog[title] = {
        'source': source,
        'explanation': explanation,
        'actions': list(actions) if actions else [],
    }
    actions_html = "".join(f"<li>{item}</li>" for item in actions)
    body = [
        f"<p class='chart-source'>Source: {source}</p>",
        f"<p>{explanation}</p>",
    ]
    if actions_html:
        body.append(f"<p class='chart-actions-title'>Recommended Actions:</p>")
        body.append(f"<ul class='chart-actions'>{actions_html}</ul>")
    st.markdown("".join(body), unsafe_allow_html=True)


def get_relevant_chart_contexts(question: str, max_items: int = 4):
    catalog = st.session_state.get('chart_context_catalog', {})
    if not catalog or not question:
        return []
    question_tokens = set(re.findall(r"[A-Za-z0-9%]+", question.lower()))
    if not question_tokens:
        return []
    scored = []
    for title, meta in catalog.items():
        haystack = " ".join([title, meta.get('source', ''), meta.get('explanation', '')] + meta.get('actions', []))
        haystack_tokens = haystack.lower()
        score = sum(1 for token in question_tokens if token and token in haystack_tokens)
        if score:
            scored.append((score, title, meta))
    if not scored:
        return []
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(title, meta) for score, title, meta in scored[:max_items]]


def _scenario_or_sim(label: str | None = None) -> tuple[str, dict] | tuple[None, None]:
    scenarios = st.session_state.get('scenarios', {})
    if label and label in scenarios:
        return label, scenarios[label]
    selected = st.session_state.get('selected_scenarios_current') or st.session_state.get('selected_scenarios')
    if selected:
        for key in selected:
            if key in scenarios:
                return key, scenarios[key]
    if scenarios:
        key = next(iter(scenarios))
        return key, scenarios[key]
    sim = st.session_state.get('sim')
    if isinstance(sim, dict) and sim:
        return "Active Sim", sim
    return None, None


def _collect_metrics(label: str | None, payload: dict | None):
    if not payload:
        return None
    capex_value = float(payload.get('capex', base_capex))
    try:
        metrics = compute_scenario_metrics(payload, capex_value)
    except Exception:
        metrics = None
    return metrics


def _compute_scenario_yields(payload: dict | None):
    if not payload:
        return None
    msw = float(payload.get('msw_tpd', 0) or 0)
    tires = float(payload.get('tires_tpd', 0) or 0)
    rdf = payload.get('rdf_yield')
    tdf = payload.get('tdf_yield')
    syngas = payload.get('syngas_yield')
    if any(val is None for val in (rdf, tdf, syngas)):
        calc_rdf, calc_tdf, calc_syngas, _ = simulate_mass_balance(msw, tires)
        if rdf is None:
            rdf = calc_rdf
        if tdf is None:
            tdf = calc_tdf
        if syngas is None:
            syngas = calc_syngas
    if max(msw, tires, float(rdf or 0), float(tdf or 0), float(syngas or 0)) <= 0:
        return None
    return {
        'msw_tpd': float(msw),
        'tires_tpd': float(tires),
        'rdf_yield': float(rdf or 0),
        'tdf_yield': float(tdf or 0),
        'syngas_yield': float(syngas or 0),
    }


def _format_delta(current, base, unit=""):
    if current is None or base is None or not np.isfinite(current) or not np.isfinite(base):
        return "N/A"
    delta = current - base
    if unit == "%":
        return f"{current:.2f}% ({delta:+.2f} pts vs base)"
    if unit == "x":
        return f"{current:.2f}x ({delta:+.2f}x vs base)"
    return f"{format_metric_value(current)} ({format_currency_delta(delta)} vs base)"


def _build_forecast_context():
    forecasts = st.session_state.get('forecast_results', [])
    if not forecasts:
        return None
    last = forecasts[-1]
    return (
        f"Scenario {last['scenario']} ({last['metric']} over {last['horizon']} yrs) "
        f"forecast summary: {last['summary']}"
    )


def _build_optimizer_context():
    opt_res = st.session_state.get('optimizer_result')
    if not opt_res:
        return None
    scen = opt_res.get('scenario') or {}
    metrics = opt_res.get('metrics') or {}
    return (
        f"Optimizer {'met' if opt_res.get('meets_target') else 'fell short of'} IRR target {opt_res.get('target', 0):.2f}% "
        f"with IRR {metrics.get('irr', float('nan')):.2f}%, NPV {format_metric_value(metrics.get('npv'))}, "
        f"MSW {scen.get('msw_tpd', 'N/A')} TPD, Tires {scen.get('tires_tpd', 'N/A')} TPD, "
        f"Power price ${scen.get('power_price', 'N/A')}/kWh."
    )


def _build_change_history_context(limit: int = 5):
    history = st.session_state.get('param_change_history', [])
    if not history:
        return None
    lines = []
    for entry in history[-limit:]:
        lines.append(f"{entry['timestamp']}: {entry['summary']}")
    return "Recent parameter changes: " + " | ".join(lines)


def _direct_chat_response(question: str):
    q = question.lower()
    scenarios = st.session_state.get('scenarios', {})
    base_label = 'Base' if 'Base' in scenarios else None
    base_payload = scenarios.get('Base') if base_label else base_scenario_raw
    base_metrics = _collect_metrics(base_label, base_payload)
    active_label, active_payload = _scenario_or_sim(None)
    active_metrics = _collect_metrics(active_label, active_payload)

    def build_sources(*names):
        filtered = [name for name in names if name]
        return f"\n\nSources: {', '.join(filtered)}" if filtered else ""

    if 'compare' in q and 'base' in q and ('optim' in q or 'target' in q):
        opt_res = st.session_state.get('optimizer_result')
        if not base_metrics or not opt_res:
            return "Optimizer results are not available yet. Run the scenario optimizer to generate a comparison."
        opt_metrics = opt_res.get('metrics') or _collect_metrics("Optimized", opt_res.get('scenario'))
        if not opt_metrics:
            return "Optimizer metrics are unavailable. Try re-running the optimizer."
        lines = [
            f"IRR: Base {base_metrics['irr']:.2f}% vs Optimized {opt_metrics['irr']:.2f}% "
            f"({opt_metrics['irr'] - base_metrics['irr']:+.2f} pts).",
            f"NPV: Base {format_metric_value(base_metrics['npv'])} vs Optimized {format_metric_value(opt_metrics['npv'])} "
            f"({format_currency_delta(opt_metrics['npv'] - base_metrics['npv'])}).",
            f"Cash-on-Cash: Base {base_metrics['cash_on_cash']:.2f}x vs Optimized {opt_metrics['cash_on_cash']:.2f}x "
            f"({opt_metrics['cash_on_cash'] - base_metrics['cash_on_cash']:+.2f}x).",
            f"Throughput: MSW {opt_res['scenario'].get('msw_tpd', 'N/A')} TPD, Tires {opt_res['scenario'].get('tires_tpd', 'N/A')} TPD.",
        ]
        return "\n".join(lines) + build_sources("Scenario Optimizer", "Scenario Metrics (Financials Yearly)")

    if ('stage 1' in q or 'mass balance' in q) and ('yield' in q or 'rdf' in q or 'syngas' in q):
        yields = _compute_scenario_yields(active_payload or base_payload)
        if not yields:
            return "No mass-balance yields are available yet. Run a scenario so MSW and tire inputs are populated."
        lines = [
            f"MSW feed: {yields['msw_tpd']:.0f} TPD, Tires: {yields['tires_tpd']:.0f} TPD.",
            f"RDF yield: {yields['rdf_yield']:.2f} TPD, TDF yield: {yields['tdf_yield']:.2f} TPD, Syngas: {yields['syngas_yield']:.2f} TPD.",
        ]
        power_mwe = (yields['syngas_yield'] / 450.3) * 36.5 + (yields['tdf_yield'] / 85.0) * 9.8
        lines.append(f"Estimated net power: {power_mwe:.2f} MW (scale {power_mwe / 46.0:.2f}× design basis).")
        return "\n".join(lines) + build_sources("Stage 1 Mass Balance", "Mass-Balance Calculation")

    if 'forecast' in q:
        entry = _build_forecast_context()
        if not entry:
            return "No forecast results are available yet. Generate a projection from the Forecast Explorer first."
        return entry + build_sources("Forecast Explorer")

    if 'revenue' in q and ('top' in q or 'streams' in q or 'breakdown' in q):
        rev_series = (active_payload or {}).get('revenue')
        if rev_series is None:
            return "Revenue data isn’t loaded yet. Select or run a scenario to populate revenue."
        total = float(np.nansum(rev_series))
        if total <= 0:
            return "Revenue totals are zero; the scenario may not be initialized."
        # Current app uses placeholder split for the donut chart; reflect that explicitly.
        mix = {
            'Power sales': 0.30,
            'Byproducts': 0.40,
            'Other/fees': 0.30,
        }
        lines = ["Revenue mix (placeholder assumptions used in the chart):"]
        for label, share in mix.items():
            lines.append(f"- {label}: {share*100:.1f}% (~{format_metric_value(total * share)})")
        lines.append("Update the Revenue Breakdown logic with actual product splits for higher fidelity.")
        return "\n".join(lines) + build_sources("Revenue Breakdown (assumed)")

    if 'monte carlo' in q:
        mc_store = st.session_state.get('monte_carlo')
        summary = st.session_state.get('monte_carlo_summary_text')
        if not mc_store or not summary:
            return "No Monte Carlo run has been executed yet. Run the Monte Carlo simulation to populate results."
        irr_stats = mc_store['summary'].get('IRR (%)', {})
        npv_stats = mc_store['summary'].get('NPV ($)', {})
        irr_p10 = irr_stats.get('p10')
        irr_p90 = irr_stats.get('p90')
        npv_p10 = npv_stats.get('p10')
        npv_p90 = npv_stats.get('p90')
        lines = [summary]
        if irr_p10 is not None and irr_p90 is not None:
            lines.append(f"IRR distribution: P10 {irr_p10:.2f}%, P90 {irr_p90:.2f}%.")
        if npv_p10 is not None and npv_p90 is not None:
            lines.append(
                f"NPV range: {format_metric_value(npv_p10)} at P10 up to {format_metric_value(npv_p90)} at P90."
            )
        return "\n".join(lines) + build_sources("Monte Carlo (Revenue Drivers)")

    if 'sankey' in q or ('msw' in q and 'tire' in q and 'syngas' in q):
        yields = _compute_scenario_yields(active_payload or base_payload)
        if not yields:
            return "No mass-balance data is available yet for the Sankey. Run a scenario first."
        lines = [
            f"Inputs: {yields['msw_tpd']:.0f} TPD MSW, {yields['tires_tpd']:.0f} TPD tires.",
            f"Outputs: {yields['rdf_yield']:.2f} TPD RDF → {yields['syngas_yield']:.2f} TPD syngas and "
            f"{yields['tdf_yield']:.2f} TPD tire-derived fuel.",
            f"Sankey branches route syngas to power ({yields['syngas_yield']:.2f} units) and byproducts "
            f"({yields['syngas_yield']*0.7:.2f} units under the 70% assumption)."
        ]
        return "\n".join(lines) + build_sources("Stage 1 Mass Balance", "Mass-Balance Sankey Assumptions")

    if ('expense' in q and 'growth' in q) or ('largest' in q and 'expenses' in q):
        scenario = active_payload or base_payload
        if not scenario or 'expenses' not in scenario:
            return "Expense trajectories are unavailable. Load a scenario with expense data to analyze growth."
        expenses = np.array(scenario.get('expenses'), dtype=float)
        if expenses.size <= 1:
            return "Not enough expense data points to evaluate growth."
        diffs = np.diff(expenses)
        years_local = align_year_labels(expenses.size)
        growth = list(zip(years_local[1:], diffs))
        growth.sort(key=lambda item: item[1], reverse=True)
        top = growth[:3]
        lines = ["Largest year-over-year expense increases:"]
        for year, delta in top:
            lines.append(f"- {int(year)}: {format_currency_delta(delta)} vs prior year.")
        lines.append("Drill into Financials Yearly to isolate the categories driving these jumps.")
        return "\n".join(lines) + build_sources("Financials Yearly")

    if ('tuning' in q or 'what-if' in q) and 'payback' in q:
        if not base_metrics:
            return "Baseline metrics are missing, so payback comparisons cannot be calculated."
        tuned_payload = st.session_state.get('sim')
        tuned_metrics = _collect_metrics("Tuned", tuned_payload)
        if not tuned_metrics or tuned_metrics.get('payback_year') in (None, float('nan')):
            return "No tuned scenario metrics available yet. Adjust sliders in the tuning panel to generate them."
        base_payback = base_metrics.get('payback_year')
        tuned_payback = tuned_metrics.get('payback_year')
        if not np.isfinite(base_payback) or not np.isfinite(tuned_payback):
            return "Payback year is undefined in either the base or tuned case."
        delta = tuned_payback - base_payback
        direction = "later" if delta > 0 else "earlier"
        return (
            f"Payback year shifts from {int(base_payback)} to {int(tuned_payback)}, "
            f"{abs(delta):.1f} years {direction} than the Excel baseline."
            + build_sources("What-if Tuning", "Scenario Metrics (Financials Yearly)")
        )

    if 'optimizer' in q or ('optimized scenario' in q):
        context = _build_optimizer_context()
        if not context:
            return "No optimizer run has been recorded yet. Execute the Scenario Optimizer to produce a summary."
        return context + build_sources("Scenario Optimizer")

    if 'change history' in q or 'parameter tweaks' in q or 'logged' in q:
        history_text = _build_change_history_context()
        if not history_text:
            return "No parameter changes have been logged in this session."
        return history_text + build_sources("Change Log")

    return None


CHAT_COMMAND_HELP = (
    "Chat Controls: set msw tpd to <400-800>, set tires tpd to <50-200>, "
    "set power price to <0.05-0.15>, set inflation to <0.01-0.05>, "
    "set capex multiplier to <0.8-1.2>. "
    "Optimizer: try 'optimize to hit 20% irr' or 'optimize irr to 18%'."
)


def get_app_knowledge_segments(question: str, max_items: int = 6):
    if not APP_KNOWLEDGE_BASE:
        return []
    if not question:
        return list(APP_KNOWLEDGE_BASE[:max_items])
    question_tokens = set(re.findall(r"[A-Za-z0-9%]+", question.lower()))
    if not question_tokens:
        return list(APP_KNOWLEDGE_BASE[:max_items])
    scored = []
    for idx, (title, content) in enumerate(APP_KNOWLEDGE_BASE):
        haystack = f"{title} {content}".lower()
        score = sum(1 for token in question_tokens if token and token in haystack)
        if score:
            scored.append((score, idx, title, content))
    if not scored:
        return list(APP_KNOWLEDGE_BASE[:max_items])
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [(title, content) for _, _, title, content in scored[:max_items]]


def get_capex_summary():
    data = capex.iloc[0:5, [2, 3]].copy()
    data.columns = ['Country', 'Value']
    data['Value'] = pd.to_numeric(data['Value'], errors='coerce')
    data = data.dropna().sort_values('Value', ascending=False)
    return data


def get_salary_summary():
    salary_df = plant_salaries.copy()
    salary_df['Salary Yearly'] = pd.to_numeric(salary_df['Salary Yearly'], errors='coerce')
    salary_df = salary_df.dropna(subset=['Salary Yearly'])
    if 'Name' in salary_df.columns:
        salary_df['Name'] = salary_df['Name'].fillna('Unlabeled Role')
    if 'Country' in salary_df.columns:
        salary_df['Country'] = salary_df['Country'].fillna('Unknown')
    salary_df = salary_df.sort_values('Salary Yearly', ascending=False)
    return salary_df


def prepare_stage1_dataframe(raw_df):
    first_col = raw_df.iloc[:, 0].astype(str)
    header_rows = raw_df[first_col.str.contains('MSW Composition', case=False, na=False)]
    if header_rows.empty:
        return raw_df
    header_idx = header_rows.index[0]
    header_values = raw_df.iloc[header_idx].ffill()
    data = raw_df.iloc[header_idx + 1:].copy()
    data.columns = header_values
    data = data.replace('', np.nan)
    return data


# Preprocess Stage 1 sheet once
stage1_processed = prepare_stage1_dataframe(stage1)


# App Layout
st.title("CREC Financial Digital Twin Simulator")
st.info(
    "CREC's Digital Twin Financial Simulator is an interactive tool that sits on top of the Excel financial model. "
    "It allows users to adjust key project parameters, run scenario analyses, and visualize financial outcomes in real-time."
)

# Inputs
st.session_state.setdefault('custom_kpis', [])
st.session_state.setdefault('forecast_results', [])
st.session_state.setdefault('optimizer_result', None)
st.session_state.setdefault('optimizer_timestamp', None)
st.session_state.setdefault('recommendations', [])
st.session_state.setdefault('whatif_rev_shift', 0.0)
st.session_state.setdefault('whatif_expense_shift', 0.0)
st.session_state.setdefault('whatif_capex_shift', 0.0)
st.session_state.setdefault('whatif_discount_shift_bps', 0)
st.session_state.setdefault('forecast_horizon', 5)
st.session_state.setdefault('forecast_scenario_select', None)
st.session_state.setdefault('forecast_metric_select', "Total Revenue")

st.session_state.setdefault('msw_tpd', 550)
st.session_state.setdefault('tires_tpd', 100)
st.session_state.setdefault('power_price', 0.07)
st.session_state.setdefault('inflation', 0.02)
st.session_state.setdefault('capex_mult', 1.0)

def render_switches_sidebar():
    with st.sidebar.expander("Switches (from Data Sources)", expanded=False):
        st.caption("These toggles are read from the Excel 'Data Sources' sheet.")
        g = st.checkbox("Graphene Switch", value=bool(st.session_state.get('switch_graphene', False)), key="ui_switch_graphene")
        r = st.checkbox("RNG Switch", value=bool(st.session_state.get('switch_rng', False)), key="ui_switch_rng")
        t = st.checkbox("Tires Only Switch", value=bool(st.session_state.get('switch_tires_only', False)), key="ui_switch_tires_only")
        # Persist any user overrides
        st.session_state['switch_graphene'] = bool(g)
        st.session_state['switch_rng'] = bool(r)
        st.session_state['switch_tires_only'] = bool(t)
        if st.button("Reset from Excel", key="reset_ds_switches"):
            sw, pm = parse_data_sources_switches_and_params(data_sources)
            st.session_state['switch_graphene'] = bool(sw.get('graphene')) if sw.get('graphene') is not None else False
            st.session_state['switch_rng'] = bool(sw.get('rng')) if sw.get('rng') is not None else False
            st.session_state['switch_tires_only'] = bool(sw.get('tires_only')) if sw.get('tires_only') is not None else False
            st.session_state['ds_params'] = pm
            trigger_rerun()


def render_validation_sidebar():
    with st.sidebar.expander("Validation & Diagnostics", expanded=False):
        st.caption("Run automated checks to verify switch impacts.")
        if st.button("Validate Switch Impacts", key="run_switch_validation"):
            st.session_state['switch_validation_results'] = run_switch_validation()
        results = st.session_state.get('switch_validation_results')
        if results:
            for item in results:
                if item.get('passed'):
                    st.success(f"{item.get('check')}: PASS — {item.get('detail')}")
                else:
                    st.error(f"{item.get('check')}: FAIL — {item.get('detail')}")

def render_base_sidebar():
    st.sidebar.subheader("What-if Tuning")
    st.sidebar.slider(
        "Revenue adjustment (%)",
        -30.0,
        30.0,
        value=st.session_state.get('whatif_rev_shift', 0.0),
        step=1.0,
        key="whatif_rev_shift",
        help="Applies a percentage shock to the revenue curve before recalculating IRR, NPV, and cash totals.",
    )
    st.sidebar.slider(
        "Expense adjustment (%)",
        -30.0,
        30.0,
        value=st.session_state.get('whatif_expense_shift', 0.0),
        step=1.0,
        key="whatif_expense_shift",
        help="Scales operating expenses by the selected percentage; positive values raise costs, negatives reduce them.",
    )
    st.sidebar.slider(
        "CapEx adjustment (%)",
        -30.0,
        30.0,
        value=st.session_state.get('whatif_capex_shift', 0.0),
        step=1.0,
        key="whatif_capex_shift",
        help="Adjusts the initial capital outlay before recomputing equity contributions and payback metrics.",
    )
    st.sidebar.slider(
        "Discount rate shift (bps)",
        -300,
        300,
        value=st.session_state.get('whatif_discount_shift_bps', 0),
        step=25,
        key="whatif_discount_shift_bps",
        help="Adds or subtracts basis points to the discount rate used in tuned NPV calculations (100 bps = 1%).",
    )
    if st.sidebar.button("Reset tuning", key="reset_whatif_tuning"):
        for state_key, default_value in (
            ("whatif_rev_shift", 0.0),
            ("whatif_expense_shift", 0.0),
            ("whatif_capex_shift", 0.0),
            ("whatif_discount_shift_bps", 0),
        ):
            st.session_state.pop(state_key, None)
            st.session_state[state_key] = default_value
        trigger_rerun()

    st.sidebar.subheader("Forecast Explorer")
    available_scenarios = list(st.session_state.get('scenarios', {}).keys())
    if not available_scenarios:
        st.sidebar.info("No scenarios available. Run a simulation first.")
    else:
        selected_scenario = st.session_state.get('forecast_scenario_select')
        if selected_scenario not in available_scenarios:
            selected_scenario = available_scenarios[0]
            st.session_state['forecast_scenario_select'] = selected_scenario
        st.sidebar.selectbox(
            "Scenario",
            available_scenarios,
            index=available_scenarios.index(st.session_state['forecast_scenario_select']),
            key="forecast_scenario_select",
        )
        metric_options = ["Total Revenue", "Total Expenses", "Cash Flow"]
        default_metric = st.session_state.get('forecast_metric_select', metric_options[0])
        if default_metric not in metric_options:
            default_metric = metric_options[0]
            st.session_state['forecast_metric_select'] = default_metric
        st.sidebar.selectbox(
            "Metric",
            metric_options,
            index=metric_options.index(default_metric),
            key="forecast_metric_select",
        )
        st.sidebar.slider(
            "Forecast Horizon (years)",
            1,
            10,
            value=int(st.session_state.get('forecast_horizon', 5)),
            key="forecast_horizon",
        )
        if st.sidebar.button("Generate Forecast", key="run_forecast_button"):
            st.session_state['forecast_trigger'] = True


def render_scenario_sidebar(msw_default, tires_default, power_price_default, inflation_default, capex_default):
    st.sidebar.subheader("Scenario Controls")
    st.sidebar.markdown("Tweak variables and run simulations.")
    msw_val = st.sidebar.slider(
        "MSW TPD",
        400,
        800,
        value=int(st.session_state.get('msw_tpd', msw_default)),
        key='msw_tpd',
        help="Average municipal solid waste processed per day; higher tonnage increases RDF and power output."
    )
    tires_val = st.sidebar.slider(
        "Tires TPD",
        50,
        200,
        value=int(st.session_state.get('tires_tpd', tires_default)),
        key='tires_tpd',
        help="Daily tire feedstock. Impacts TDF yield and supplemental power production."
    )
    power_price_val = st.sidebar.slider(
        "Power Price ($/kWh)",
        0.05,
        0.15,
        value=float(st.session_state.get('power_price', power_price_default)),
        step=0.01,
        key='power_price',
        help="Average offtake price for electricity sales. Revenue scales linearly with this value."
    )
    inflation_val = st.sidebar.slider(
        "Inflation Rate",
        0.01,
        0.05,
        value=float(st.session_state.get('inflation', inflation_default)),
        step=0.005,
        key='inflation',
        help="Annual inflation assumption applied to revenues/expenses (baseline model uses 2%)."
    )
    capex_val = st.sidebar.slider(
        "CapEx Multiplier",
        0.8,
        1.2,
        value=float(st.session_state.get('capex_mult', capex_default)),
        step=0.05,
        key='capex_mult',
        help="Multiplier on the base CapEx. Values over 1.0 represent overruns; below 1.0 represents savings."
    )
    scenario_name_val = st.sidebar.text_input(
        "Scenario Name",
        value=st.session_state.get('scenario_name_input', ''),
        key='scenario_name_input',
        help="Label used when saving a scenario snapshot for comparison charts and reports.",
    )
    action_cols = st.sidebar.columns(2)
    with action_cols[0]:
        save_clicked = st.button(
            "Save Scenario",
            type="primary",
            key="save_scenario_button",
            use_container_width=True,
        )
    with action_cols[1]:
        run_clicked = st.button(
            "Run Simulation",
            key="run_simulation_button",
            use_container_width=True,
        )
    return (
        int(msw_val),
        int(tires_val),
        float(power_price_val),
        float(inflation_val),
        float(capex_val),
        scenario_name_val,
        save_clicked,
        run_clicked,
    )

sidebar_tabs = ["Base Model", "Scenario Explorer"]
default_sidebar_tab = st.session_state.get('sidebar_tab_radio', sidebar_tabs[0])
sidebar_selection = st.sidebar.radio(
    "Navigation",
    sidebar_tabs,
    index=sidebar_tabs.index(default_sidebar_tab),
    key="sidebar_tab_radio",
)
tab_sequence = sidebar_tabs if sidebar_selection == "Base Model" else list(reversed(sidebar_tabs))
tab_handles = st.tabs(tab_sequence)
if sidebar_selection == "Base Model":
    tab_base, tab_compare = tab_handles
else:
    tab_compare, tab_base = tab_handles

_params_di = st.session_state.get('data_inputs_params', {}) or {}
def _pick_param(params, patterns, fallback):
    for p in patterns:
        for k, v in params.items():
            if p in k:
                try:
                    return float(v)
                except Exception:
                    continue
    return fallback

msw_default = int(st.session_state.get('msw_tpd', _pick_param(_params_di, ["msw_tpd", "msw_tons", "msw"], 550)))
tires_default = int(st.session_state.get('tires_tpd', _pick_param(_params_di, ["tires_tpd", "tire_tpd", "tires"], 100)))
power_price_default = float(st.session_state.get('power_price', _pick_param(_params_di, ["power_price", "price_per_kwh", "kwh_price"], 0.07)))
inflation_default = float(st.session_state.get('inflation', _pick_param(_params_di, ["inflation", "inflation_rate"], 0.02)))
capex_default = float(st.session_state.get('capex_mult', _pick_param(_params_di, ["capex_mult", "capex_multiplier"], 1.0)))

base_inputs_static = st.session_state.setdefault(
    'base_inputs_static',
    {
        'msw_tpd': float(msw_default),
        'tires_tpd': float(tires_default),
        'power_price': float(power_price_default),
        'inflation': float(inflation_default),
        'capex_mult': float(capex_default),
    },
)
st.session_state.setdefault('param_change_history', [])
st.session_state.setdefault('last_param_key', None)

base_rdf_yield, base_tdf_yield, base_syngas_yield, base_power_mwe = simulate_mass_balance(
    base_inputs_static['msw_tpd'],
    base_inputs_static['tires_tpd'],
)
base_scale_factor = base_power_mwe / 46.0 if 46.0 != 0 else 1.0

base_scenario_raw = st.session_state.setdefault(
    '_base_scenario_raw',
    {
        'revenue': base_revenue,
        'expenses': base_expenses,
        'cf': np.array(base_cash_flow_series, dtype=float),
        'cash_flows_full': np.array(base_cash_flows_full, dtype=float),
        'irr': base_project_irr,
        'npv': base_npv,
        'capex': base_capex,
        'msw_tpd': base_inputs_static['msw_tpd'],
        'tires_tpd': base_inputs_static['tires_tpd'],
        'power_price': base_inputs_static['power_price'],
        'inflation': base_inputs_static['inflation'],
        'capex_mult': base_inputs_static['capex_mult'],
        'equity_irr': base_equity_irr,
        'rdf_yield': base_rdf_yield,
        'tdf_yield': base_tdf_yield,
        'syngas_yield': base_syngas_yield,
        'scale': base_scale_factor,
        'power_mwe': base_power_mwe,
    },
)

st.session_state.setdefault('base_switches_static', {
    'graphene': bool(st.session_state.get('switch_graphene', False)),
    'rng': bool(st.session_state.get('switch_rng', False)),
    'tires_only': bool(st.session_state.get('switch_tires_only', False)),
})

st.session_state.setdefault('scenario_name_input', '')
st.sidebar.title("Control Panel")

render_switches_sidebar()
render_validation_sidebar()


def _clone_scenario_payload(payload: dict) -> dict:
    clone = {}
    for key, value in (payload or {}).items():
        if isinstance(value, np.ndarray):
            clone[key] = value.copy()
        elif isinstance(value, (list, tuple)):
            clone[key] = list(value)
        else:
            clone[key] = value
    return clone


scenarios_store = st.session_state.setdefault('scenarios', {})
if 'Base' not in scenarios_store:
    scenarios_store['Base'] = _clone_scenario_payload(base_scenario_raw)
st.session_state.setdefault('selected_scenarios', ['Base'])
if not st.session_state['selected_scenarios'] and 'Base' in scenarios_store:
    st.session_state['selected_scenarios'] = ['Base']

if sidebar_selection == "Base Model":
    render_base_sidebar()
    msw_tpd = int(st.session_state.get('msw_tpd', msw_default))
    tires_tpd = int(st.session_state.get('tires_tpd', tires_default))
    power_price = float(st.session_state.get('power_price', power_price_default))
    inflation = float(st.session_state.get('inflation', inflation_default))
    capex_mult = float(st.session_state.get('capex_mult', capex_default))
    scenario_name = st.session_state.get('scenario_name_input', '')
    save_scenario_clicked = False
    run_simulation_clicked = False
else:
    (
        msw_tpd,
        tires_tpd,
        power_price,
        inflation,
        capex_mult,
        scenario_name,
        save_scenario_clicked,
        run_simulation_clicked,
    ) = render_scenario_sidebar(msw_default, tires_default, power_price_default, inflation_default, capex_default)


def _match_length(array_like, target_len):
    arr = np.array(array_like, dtype=float)
    if arr.size >= target_len:
        return arr[:target_len]
    if arr.size == 0:
        return np.full(target_len, float('nan'))
    padding = np.full(target_len - arr.size, float('nan'))
    return np.concatenate([arr, padding])


def _compute_margin(numerator, denominator):
    margin = np.full_like(denominator, np.nan, dtype=float)
    valid = (denominator != 0) & ~np.isnan(denominator)
    margin[valid] = numerator[valid] / denominator[valid]
    if np.isnan(margin).all():
        return np.zeros_like(denominator, dtype=float)
    running = 0.0
    fallback = np.nanmean(margin[valid]) if np.any(valid) else 0.0
    for idx in range(margin.size):
        if np.isnan(margin[idx]):
            margin[idx] = running if not np.isnan(running) else fallback
        else:
            running = margin[idx]
    margin[np.isnan(margin)] = fallback if not np.isnan(fallback) else 0.0
    return margin


def _fmt_currency(value):
    try:
        if value is None or np.isnan(value):
            return None
        return f"${value:,.0f}"
    except Exception:
        return None


def _fmt_number(value, fmt="{:.0f}"):
    try:
        if value is None or np.isnan(value):
            return None
        return fmt.format(value)
    except Exception:
        return None


def _fmt_percent(value):
    try:
        if value is None or np.isnan(value):
            return None
        return f"{value:.1f}%"
    except Exception:
        return None


def _ensure_list(values):
    if values is None:
        return []
    try:
        arr = np.array(values, dtype=float)
        return arr.tolist()
    except Exception:
        return []


def _normalize_snapshot(snapshot: dict | None) -> dict | None:
    if not snapshot:
        return None
    normalized = {}
    for key in ('revenue', 'expenses', 'cf', 'cash_flows_full'):
        normalized[key] = _ensure_list(snapshot.get(key, []))
    for key in (
        'irr', 'npv', 'capex', 'msw_tpd', 'tires_tpd', 'power_price',
        'inflation', 'capex_mult', 'equity_irr', 'rdf_yield', 'tdf_yield',
        'syngas_yield', 'power_mwe', 'scale'
    ):
        if key in snapshot:
            try:
                normalized[key] = float(snapshot.get(key))
            except Exception:
                normalized[key] = snapshot.get(key)
    return normalized


def _format_pts(value: float) -> str:
    if value is None or np.isnan(value):
        return "±0.00 pts"
    sign = "+" if value > 0 else "-" if value < 0 else "±"
    return f"{sign}{abs(value):.2f} pts"


def _convert_basic_markdown_to_html(text: str) -> str:
    if text is None:
        return ""
    remaining = str(text)
    html_fragments: list[str] = []
    while True:
        start = remaining.find("**")
        if start == -1:
            html_fragments.append(html.escape(remaining))
            break
        html_fragments.append(html.escape(remaining[:start]))
        remaining = remaining[start + 2 :]
        end = remaining.find("**")
        if end == -1:
            html_fragments.append(html.escape("**" + remaining))
            break
        bold_text = remaining[:end]
        html_fragments.append(f"<strong>{html.escape(bold_text)}</strong>")
        remaining = remaining[end + 2 :]
    combined = "".join(html_fragments)
    return combined.replace("$", "&#36;")


def _render_bullet_list(lines, css_class: str = "base-overview-list"):
    if not lines:
        return False
    items = []
    for raw_line in lines:
        if raw_line is None:
            continue
        line = str(raw_line).strip()
        if not line:
            continue
        if line.startswith("- "):
            line = line[2:].lstrip()
        items.append(f"<li>{_convert_basic_markdown_to_html(line)}</li>")
    if not items:
        return False
    html_list = "".join(items)
    st.markdown(f"<ul class='{css_class}'>{html_list}</ul>", unsafe_allow_html=True)
    return True


def _escape_for_markdown(text: str | None) -> str:
    if text is None:
        return ""
    escaped = str(text)
    escaped = escaped.replace("\\", "\\\\")
    escaped = escaped.replace("$", "\\$")
    escaped = escaped.replace("_", "\\_")
    return escaped


def render_data_inferences(
    context_label: str,
    target_snapshot: dict | None,
    baseline_snapshot: dict | None,
    comparison_entries: list[dict] | None = None
):
    st.header(f"Data Inferences & Insights — {context_label}")

    normalized_target = _normalize_snapshot(target_snapshot)
    if not normalized_target:
        st.info("No scenario data available yet. Run a simulation or select a scenario to generate insights.")
        return

    normalized_baseline = _normalize_snapshot(baseline_snapshot) or normalized_target

    target_capex = float(normalized_target.get('capex', base_capex)) if normalized_target else base_capex
    baseline_capex_value = float(normalized_baseline.get('capex', base_capex)) if normalized_baseline else base_capex

    try:
        metrics_target = compute_scenario_metrics(normalized_target, target_capex)
    except Exception:
        st.info("Unable to compute insights for the current data snapshot.")
        return

    try:
        metrics_baseline = compute_scenario_metrics(normalized_baseline, baseline_capex_value)
    except Exception:
        metrics_baseline = None

    is_base_context = context_label.lower().startswith('base')
    base_snapshot = normalized_baseline if metrics_baseline else normalized_target

    comparison_entries = comparison_entries or []
    comparison_details = []
    seen_labels: set[str] = set()
    for entry in comparison_entries:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get('label') or "Scenario")
        if label in seen_labels:
            continue
        normalized_entry = _normalize_snapshot(entry.get('snapshot'))
        if not normalized_entry:
            continue
        is_base_entry = bool(entry.get('is_base')) or label.strip().lower() == "base"
        try:
            capex_entry = float(normalized_entry.get('capex', base_capex))
        except Exception:
            capex_entry = base_capex
        try:
            metrics_entry = compute_scenario_metrics(normalized_entry, capex_entry)
        except Exception:
            metrics_entry = None
        comparison_details.append({
            'label': label,
            'normalized': normalized_entry,
            'metrics': metrics_entry,
            'is_base': is_base_entry,
        })
        seen_labels.add(label)

    baseline_entry = next((item for item in comparison_details if item['is_base']), None)
    if baseline_entry is None and metrics_baseline:
        baseline_entry = {
            'label': "Base",
            'normalized': normalized_baseline,
            'metrics': metrics_baseline,
            'is_base': True,
        }
        comparison_details.insert(0, baseline_entry)

    base_metrics_for_compare = baseline_entry['metrics'] if baseline_entry else metrics_baseline
    non_base_comparisons = [
        item for item in comparison_details if not item.get('is_base')
    ]
    comparison_mode = bool(non_base_comparisons) and context_label.lower().startswith('scenario')

    def to_float(value):
        try:
            f = float(value)
            return f if np.isfinite(f) else np.nan
        except Exception:
            return np.nan

    msw_val = to_float(normalized_target.get('msw_tpd'))
    tires_val = to_float(normalized_target.get('tires_tpd'))
    rdf_val = to_float(normalized_target.get('rdf_yield'))
    syngas_val = to_float(normalized_target.get('syngas_yield'))
    power_val = to_float(normalized_target.get('power_mwe'))
    scale_val = to_float(normalized_target.get('scale'))

    msw_base = to_float(base_snapshot.get('msw_tpd'))
    tires_base = to_float(base_snapshot.get('tires_tpd'))

    overview_bits = []
    if np.isfinite(msw_val):
        if is_base_context:
            overview_bits.append(f"The base case processes about **{msw_val:.0f} tons/day** of MSW through Stage 1.")
        elif not np.isfinite(msw_base) or abs(msw_val - msw_base) < 1:
            overview_bits.append(f"Current sliders push **{msw_val:.0f} tons/day** of MSW through Stage 1.")
        else:
            delta = msw_val - msw_base
            direction = "higher" if delta > 0 else "lower"
            overview_bits.append(
                f"MSW feed averages **{msw_val:.0f} TPD**, about **{abs(delta):.0f} TPD** {direction} than the base case."
            )
    if np.isfinite(tires_val):
        if is_base_context:
            overview_bits.append(f"Baseline tire co-feed holds near **{tires_val:.0f} TPD**.")
        elif not np.isfinite(tires_base) or abs(tires_val - tires_base) < 0.5:
            overview_bits.append(f"Tire co-feed rates sit near **{tires_val:.0f} TPD**.")
        else:
            delta = tires_val - tires_base
            direction = "higher" if delta > 0 else "lower"
            overview_bits.append(
                f"Tire throughput runs **{abs(delta):.0f} TPD** {direction} than base at **{tires_val:.0f} TPD**."
            )
    if np.isfinite(rdf_val) and np.isfinite(syngas_val):
        if is_base_context:
            overview_bits.append(
                f"Base conversion yields produce roughly **{rdf_val:.2f} TPD** of RDF and **{syngas_val:.2f} TPD** of syngas."
            )
        else:
            overview_bits.append(
                f"Conversion yields deliver roughly **{rdf_val:.2f} TPD** of RDF and **{syngas_val:.2f} TPD** of syngas."
            )
    elif np.isfinite(rdf_val):
        overview_bits.append(f"RDF yield stabilizes near **{rdf_val:.2f} TPD**.")
    if np.isfinite(scale_val):
        power_phrase = f" (~{power_val:.1f} MW)" if np.isfinite(power_val) else ""
        if is_base_context:
            overview_bits.append(f"Baseline net power output sits at **{scale_val:.2f}×** the design basis{power_phrase}.")
        else:
            overview_bits.append(f"Overall scale factor sits at **{scale_val:.2f}×** the design basis{power_phrase}.")

    if comparison_mode and base_metrics_for_compare:
        summary_lines = []
        base_revenue_cmp = base_metrics_for_compare.get('total_revenue') if base_metrics_for_compare else float('nan')
        base_expense_cmp = base_metrics_for_compare.get('total_expenses') if base_metrics_for_compare else float('nan')
        base_capex_cmp = base_metrics_for_compare.get('capex') if base_metrics_for_compare else float('nan')
        for item in non_base_comparisons:
            metrics_cmp = item.get('metrics') or {}
            label = item.get('label', "Scenario")
            rev_value = metrics_cmp.get('total_revenue', float('nan'))
            exp_value = metrics_cmp.get('total_expenses', float('nan'))
            capex_value = metrics_cmp.get('capex', float('nan'))
            rev_delta = rev_value - base_revenue_cmp if np.isfinite(base_revenue_cmp) else float('nan')
            exp_delta = exp_value - base_expense_cmp if np.isfinite(base_expense_cmp) else float('nan')
            capex_delta = capex_value - base_capex_cmp if np.isfinite(base_capex_cmp) else float('nan')

            def _segment(label_text, value, delta):
                value_text = format_metric_value(value)
                if np.isfinite(delta):
                    delta_text = format_currency_delta(delta)
                    return f"{label_text} {value_text} ({delta_text} vs base)"
                return f"{label_text} {value_text}"

            summary_lines.append(
                f"{label}: " +
                "; ".join(
                    filter(
                        None,
                        [
                            _segment("revenue", rev_value, rev_delta),
                            _segment("expenses", exp_value, exp_delta),
                            _segment("CapEx", capex_value, capex_delta),
                        ],
                    )
                )
            )
        if not _render_bullet_list(summary_lines):
            _render_bullet_list(overview_bits)
    else:
        _render_bullet_list(overview_bits)

    irr_value = metrics_target.get('irr')
    irr_base = metrics_baseline.get('irr') if metrics_baseline else np.nan
    npv_value = metrics_target.get('npv')
    npv_base = metrics_baseline.get('npv') if metrics_baseline else np.nan
    revenue_value = metrics_target.get('total_revenue')
    revenue_base = metrics_baseline.get('total_revenue') if metrics_baseline else np.nan
    coc_value = metrics_target.get('cash_on_cash')
    coc_base = metrics_baseline.get('cash_on_cash') if metrics_baseline else np.nan

    st.subheader("Key Takeaways")
    takeaway_lines = []
    if comparison_mode and base_metrics_for_compare:
        base_irr_cmp = base_metrics_for_compare.get('irr', float('nan')) if base_metrics_for_compare else float('nan')
        base_npv_cmp = base_metrics_for_compare.get('npv', float('nan')) if base_metrics_for_compare else float('nan')
        base_revenue_cmp = base_metrics_for_compare.get('total_revenue', float('nan')) if base_metrics_for_compare else float('nan')
        base_coc_cmp = base_metrics_for_compare.get('cash_on_cash', float('nan')) if base_metrics_for_compare else float('nan')
        for item in non_base_comparisons:
            metrics_cmp = item.get('metrics') or {}
            label = item.get('label', "Scenario")
            pieces = []
            irr_cmp = metrics_cmp.get('irr', float('nan'))
            if np.isfinite(irr_cmp) and np.isfinite(base_irr_cmp):
                pieces.append(f"IRR {irr_cmp:.2f}% ({_format_pts(irr_cmp - base_irr_cmp)} vs base)")
            elif np.isfinite(irr_cmp):
                pieces.append(f"IRR {irr_cmp:.2f}%")
            npv_cmp = metrics_cmp.get('npv', float('nan'))
            if np.isfinite(npv_cmp) and np.isfinite(base_npv_cmp):
                pieces.append(f"NPV {format_metric_value(npv_cmp)} ({format_currency_delta(npv_cmp - base_npv_cmp)} vs base)")
            elif np.isfinite(npv_cmp):
                pieces.append(f"NPV {format_metric_value(npv_cmp)}")
            revenue_cmp = metrics_cmp.get('total_revenue', float('nan'))
            if np.isfinite(revenue_cmp) and np.isfinite(base_revenue_cmp):
                pieces.append(f"Revenue shift {format_currency_delta(revenue_cmp - base_revenue_cmp)}")
            cash_on_cash_cmp = metrics_cmp.get('cash_on_cash', float('nan'))
            if np.isfinite(cash_on_cash_cmp) and np.isfinite(base_coc_cmp):
                delta_coc = cash_on_cash_cmp - base_coc_cmp
                pieces.append(f"Cash-on-cash {cash_on_cash_cmp:.2f}x ({format_ratio_delta(delta_coc)} vs base)")
            if pieces:
                takeaway_lines.append(f"{label}: " + "; ".join(pieces))
    else:
        if is_base_context:
            if np.isfinite(irr_value):
                takeaway_lines.append(f"Base IRR stands at **{irr_value:.2f}%** with NPV {format_metric_value(npv_value)}.")
            if np.isfinite(revenue_value):
                takeaway_lines.append(f"Lifetime revenue totals {format_metric_value(revenue_value)} against expenses {format_metric_value(metrics_target.get('total_expenses'))}.")
            if np.isfinite(coc_value):
                takeaway_lines.append(f"Cash-on-cash settles near **{coc_value:.2f}x**, supporting the baseline financing case.")
        else:
            if np.isfinite(irr_value) and np.isfinite(irr_base):
                delta = irr_value - irr_base
                takeaway_lines.append(
                    f"IRR now runs **{irr_value:.2f}%** ({_format_pts(delta)} vs. base)."
                )
            elif np.isfinite(irr_value):
                takeaway_lines.append(f"IRR clocks in at **{irr_value:.2f}%** for this scenario.")
            if np.isfinite(npv_value) and np.isfinite(npv_base):
                delta = npv_value - npv_base
                direction = "up" if delta > 0 else "down" if delta < 0 else "flat"
                takeaway_lines.append(
                    f"Scenario NPV is {format_metric_value(npv_value)} ({format_currency_delta(delta)} vs. base, trending {direction})."
                )
            if np.isfinite(revenue_value) and np.isfinite(revenue_base):
                delta = revenue_value - revenue_base
                takeaway_lines.append(
                    f"Total revenue shift is {format_currency_delta(delta)} compared with the Excel baseline."
                )
            if np.isfinite(coc_value) and np.isfinite(coc_base):
                delta = coc_value - coc_base
                takeaway_lines.append(
                    f"Cash-on-cash now tracks **{coc_value:.2f}x** ({format_ratio_delta(delta)} vs. base)."
                )

    mc_summary = st.session_state.get('monte_carlo_summary_text')
    if not is_base_context and mc_summary:
        takeaway_lines.append(f"Monte Carlo recap: {mc_summary}")

    if not _render_bullet_list(takeaway_lines):
        st.markdown("No additional takeaways available yet; run the scenario or populate more data.")

    recommendation_lines = []
    if comparison_mode and base_metrics_for_compare:
        base_irr_cmp = base_metrics_for_compare.get('irr', float('nan')) if base_metrics_for_compare else float('nan')
        base_coc_cmp = base_metrics_for_compare.get('cash_on_cash', float('nan')) if base_metrics_for_compare else float('nan')
        base_revenue_cmp = base_metrics_for_compare.get('total_revenue', float('nan')) if base_metrics_for_compare else float('nan')
        base_capex_cmp = base_metrics_for_compare.get('capex', float('nan')) if base_metrics_for_compare else float('nan')
        for item in non_base_comparisons:
            metrics_cmp = item.get('metrics') or {}
            label = item.get('label', "Scenario")
            actions = []
            irr_cmp = metrics_cmp.get('irr', float('nan'))
            if np.isfinite(irr_cmp) and np.isfinite(base_irr_cmp):
                delta = irr_cmp - base_irr_cmp
                if delta > 0.25:
                    actions.append("Lock the IRR upside by stress-testing throughput and pricing resilience.")
                elif delta < -0.25:
                    actions.append("IRR erosion flags potential cost or CapEx overruns—re-cut budgets before advancing.")
            coc_cmp = metrics_cmp.get('cash_on_cash', float('nan'))
            if np.isfinite(coc_cmp) and np.isfinite(base_coc_cmp):
                delta_coc = coc_cmp - base_coc_cmp
                if delta_coc < -0.1:
                    actions.append("Cash-on-cash softens; revisit equity structure or ramp curve.")
            revenue_cmp = metrics_cmp.get('total_revenue', float('nan'))
            if np.isfinite(revenue_cmp) and np.isfinite(base_revenue_cmp):
                rev_delta = revenue_cmp - base_revenue_cmp
                if abs(rev_delta) >= 1_000_000:
                    direction = "upside" if rev_delta > 0 else "downside"
                    actions.append(f"Revenue {direction} of {format_metric_value(abs(rev_delta))}; reconcile offtake and throughput inputs.")
            capex_cmp = metrics_cmp.get('capex', float('nan'))
            if np.isfinite(capex_cmp) and np.isfinite(base_capex_cmp):
                capex_delta = capex_cmp - base_capex_cmp
                if abs(capex_delta) >= 1_000_000:
                    if capex_delta > 0:
                        actions.append(f"CapEx overruns by ~{format_metric_value(abs(capex_delta))}; evaluate scope or contingency.")
                    else:
                        actions.append(f"CapEx trims by ~{format_metric_value(abs(capex_delta))}; confirm procurement can sustain savings.")
            if actions:
                recommendation_lines.append(f"{label}: " + " ".join(actions))
        if not recommendation_lines and takeaway_lines:
            recommendation_lines.append("Scenario deltas remain close to base; continue monitoring before committing to changes.")
    else:
        if not is_base_context and np.isfinite(irr_value) and np.isfinite(irr_base):
            delta = irr_value - irr_base
            if delta > 0.25:
                recommendation_lines.append("Consider locking in higher throughput and power pricing to preserve the IRR uplift versus base.")
            elif delta < -0.25:
                recommendation_lines.append("Investigate cost or capex overruns to recover lost IRR before advancing this case.")
        if np.isfinite(coc_value) and (np.isnan(coc_base) or coc_value < 1.2):
            recommendation_lines.append("Evaluate equity structure or ramp assumptions to strengthen cash-on-cash returns.")
        if not recommendation_lines and takeaway_lines:
            recommendation_lines.append("Use these insights as a baseline, then iterate the scenario controls or Monte Carlo settings for deeper testing.")

    st.subheader("Recommended Next Moves")
    if not _render_bullet_list(recommendation_lines):
        st.markdown("No immediate actions surfaced. Adjust inputs or run Monte Carlo to generate further guidance.")

def render_base_tab():
    st.header("Base Model Highlights")
    base_revenue_summary = summary_series_aligned.get('revenue', np.array([]))
    if base_revenue_summary.size == 0:
        st.info("Unable to read the 'Summary' tab highlights. Please verify the sheet structure and reload the app.")
    else:
        sim_snapshot = st.session_state.get('sim')
        if not sim_snapshot:
            sim_snapshot = st.session_state.get('scenarios', {}).get('Base')
        if not sim_snapshot:
            sim_snapshot = base_scenario_raw

        year_axis = [_ for _ in (int(str(y)) for y in years)]
        target_len = min(len(year_axis), base_revenue_summary.size)
        year_axis = year_axis[:target_len]
        base_revenue_summary = base_revenue_summary[:target_len]
        base_ebitda_summary = summary_series_aligned.get('ebitda', np.array([]))[:target_len]
        base_net_income_summary = summary_series_aligned.get('net_income', np.array([]))[:target_len]
        base_project_cf_summary = summary_series_aligned.get('project_cash_flow', np.array([]))[:target_len]

        active_revenue = _match_length(sim_snapshot.get('revenue', base_revenue_summary), target_len)
        active_expenses = _match_length(sim_snapshot.get('expenses', base_expenses), target_len)
        ebitda_margin = _compute_margin(base_ebitda_summary, np.where(base_revenue_summary == 0, np.nan, base_revenue_summary))
        net_margin = _compute_margin(base_net_income_summary, np.where(base_revenue_summary == 0, np.nan, base_revenue_summary))
        active_ebitda = active_revenue * ebitda_margin
        active_net_income = active_revenue * net_margin
        active_cf = active_revenue - active_expenses
        active_operating_expense = active_revenue - active_ebitda
        active_other_costs = active_ebitda - active_net_income

        cash_flows_full = np.array(sim_snapshot.get('cash_flows_full', []), dtype=float)
        if cash_flows_full.size >= target_len + 1:
            active_project_cf = cash_flows_full[1:target_len + 1]
        else:
            ratio = np.divide(
                active_revenue,
                base_revenue_summary,
                out=np.ones_like(active_revenue),
                where=~np.isclose(base_revenue_summary, 0)
            )
            active_project_cf = base_project_cf_summary * ratio

        base_totals = {
            'revenue': np.nansum(base_revenue_summary),
            'ebitda': np.nansum(base_ebitda_summary),
            'net_income': np.nansum(base_net_income_summary),
            'project_cf': np.nansum(base_project_cf_summary),
        }
        active_totals = {
            'revenue': np.nansum(active_revenue),
            'ebitda': np.nansum(active_ebitda),
            'net_income': np.nansum(active_net_income),
            'project_cf': np.nansum(active_project_cf),
        }

        project_irr_current = float(sim_snapshot.get('irr', float('nan')))
        equity_irr_current = sim_snapshot.get('equity_irr')
        npv_current = float(sim_snapshot.get('npv', float('nan')))
        project_irr_base = summary_scalar_highlights.get('total_project_irr_pct', float('nan'))
        equity_irr_base = summary_scalar_highlights.get('total_equity_irr_pct', float('nan'))
        npv_base = summary_scalar_highlights.get('net_present_value', float('nan'))
        cash_base = summary_scalar_highlights.get('total_project_cash_flow', float('nan'))
        base_capex_value = float(sim_snapshot.get('capex', base_capex))
        revenue_shift_pct = float(st.session_state.get('whatif_rev_shift', 0.0))
        expense_shift_pct = float(st.session_state.get('whatif_expense_shift', 0.0))
        capex_shift_pct = float(st.session_state.get('whatif_capex_shift', 0.0))
        discount_shift_bps = float(st.session_state.get('whatif_discount_shift_bps', 0))

        with st.expander("What-if Tuning", expanded=False):
            st.caption("Adjust the tuning sliders from the sidebar to apply quick sensitivities.")
            _render_bullet_list(
                [
                    f"Revenue adjustment: **{revenue_shift_pct:+.0f}%**",
                    f"Expense adjustment: **{expense_shift_pct:+.0f}%**",
                    f"CapEx adjustment: **{capex_shift_pct:+.0f}%**",
                    f"Discount rate shift: **{discount_shift_bps:+.0f} bps**",
                ]
            )
            st.caption("Use the 'Reset tuning' button in the sidebar to revert to baseline values.")

        revenue_multiplier = 1 + float(revenue_shift_pct) / 100.0
        expense_multiplier = 1 + float(expense_shift_pct) / 100.0
        capex_multiplier_tuning = 1 + float(capex_shift_pct) / 100.0
        tuned_discount_rate = max(base_discount + float(discount_shift_bps) / 10000.0, 0.0001)
        tuning_applied = any(
            [
                abs(float(revenue_shift_pct)) > 1e-6,
                abs(float(expense_shift_pct)) > 1e-6,
                abs(float(capex_shift_pct)) > 1e-6,
                abs(int(discount_shift_bps)) > 0,
            ]
        )

        tuned_revenue = active_revenue * revenue_multiplier
        tuned_expenses = np.nan_to_num(active_expenses, nan=0.0) * revenue_multiplier * expense_multiplier

        tuned_operating_expense = np.nan_to_num(active_operating_expense, nan=0.0) * revenue_multiplier * expense_multiplier
        tuned_ebitda = tuned_revenue - tuned_operating_expense
        other_costs = np.nan_to_num(active_other_costs, nan=0.0)
        tuned_net_income = tuned_ebitda - other_costs

        ebitda_ratio = np.divide(
            tuned_ebitda,
            np.where(np.isclose(active_ebitda, 0), np.nan, active_ebitda),
        )
        ebitda_ratio = np.nan_to_num(ebitda_ratio, nan=1.0, posinf=1.0, neginf=1.0)

        tuned_project_cf = np.nan_to_num(active_project_cf, nan=0.0) * ebitda_ratio

        tuned_capex_value = base_capex_value * capex_multiplier_tuning
        if cash_flows_full.size > 0:
            base_operating_cf_full = cash_flows_full[1:]
            if base_operating_cf_full.size:
                ratio_full = np.ones_like(base_operating_cf_full)
                overlap = min(base_operating_cf_full.size, ebitda_ratio.size)
                ratio_full[:overlap] = ebitda_ratio[:overlap]
                if overlap < base_operating_cf_full.size:
                    fill_value = ebitda_ratio[overlap - 1] if overlap > 0 else 1.0
                    ratio_full[overlap:] = fill_value
                tuned_operating_cf_full = base_operating_cf_full * ratio_full
            else:
                tuned_operating_cf_full = np.array([], dtype=float)
            tuned_cash_flows_full = np.hstack(
                (np.array([tuned_capex_value], dtype=float), tuned_operating_cf_full)
            )
        else:
            tuned_cash_flows_full = np.hstack(
                (np.array([tuned_capex_value], dtype=float), np.asarray(tuned_project_cf, dtype=float))
            )

        tuned_cash_flows_full = np.where(np.isnan(tuned_cash_flows_full), 0.0, tuned_cash_flows_full)

        project_irr_tuned = (
            npf.irr(tuned_cash_flows_full) * 100
            if tuned_cash_flows_full.size > 1 and np.any(tuned_cash_flows_full[1:] > 0)
            else float('nan')
        )
        if tuned_cash_flows_full.size > 1:
            npv_tuned = npf.npv(tuned_discount_rate, tuned_cash_flows_full[1:]) + tuned_cash_flows_full[0]
        else:
            npv_tuned = tuned_cash_flows_full[0]

        tuned_totals = {
            'revenue': np.nansum(tuned_revenue),
            'ebitda': np.nansum(tuned_ebitda),
            'net_income': np.nansum(tuned_net_income),
            'project_cf': np.nansum(tuned_project_cf),
        }

        totals_display = dict(tuned_totals)

        base_equity_total = summary_scalar_highlights.get('total_equity_contributed')
        base_ltv_ratio = summary_scalar_highlights.get('loan_to_value')

        base_total_investment = float('nan')
        if np.isfinite(base_equity_total) and np.isfinite(base_ltv_ratio) and (1 - base_ltv_ratio) > 1e-6:
            base_total_investment = base_equity_total / max(1 - base_ltv_ratio, 1e-6)
        elif np.isfinite(base_capex):
            base_total_investment = abs(base_capex)
        elif np.isfinite(base_capex_value):
            base_total_investment = abs(base_capex_value)
        elif np.isfinite(base_equity_total):
            base_total_investment = base_equity_total

        base_debt_total = float('nan')
        if np.isfinite(base_total_investment) and np.isfinite(base_ltv_ratio):
            base_debt_total = base_total_investment * base_ltv_ratio
        elif np.isfinite(base_total_investment) and np.isfinite(base_equity_total):
            base_debt_total = max(base_total_investment - base_equity_total, 0.0)

        tuned_total_investment = abs(tuned_capex_value) if np.isfinite(tuned_capex_value) else float('nan')
        tuned_debt_total = float('nan')
        if np.isfinite(tuned_total_investment) and tuned_total_investment > 0 and np.isfinite(base_debt_total):
            tuned_debt_total = base_debt_total
            if (
                np.isfinite(base_total_investment)
                and base_total_investment > 0
                and tuned_total_investment < base_total_investment - 1e-6
            ):
                scale = tuned_total_investment / base_total_investment
                tuned_debt_total = base_debt_total * scale
            tuned_debt_total = min(tuned_debt_total, tuned_total_investment)
            tuned_equity_contrib = max(tuned_total_investment - tuned_debt_total, 0.0)
            tuned_ltv_ratio = tuned_debt_total / tuned_total_investment if tuned_total_investment > 0 else float('nan')
        else:
            tuned_equity_contrib = base_equity_total if np.isfinite(base_equity_total) else float('nan')
            tuned_ltv_ratio = base_ltv_ratio if np.isfinite(base_ltv_ratio) else float('nan')

        if not np.isfinite(tuned_equity_contrib):
            tuned_equity_contrib = base_equity_total if np.isfinite(base_equity_total) else float('nan')
        if np.isfinite(tuned_ltv_ratio):
            tuned_ltv_ratio = float(np.clip(tuned_ltv_ratio, 0.0, 1.5))
        else:
            tuned_ltv_ratio = base_ltv_ratio if np.isfinite(base_ltv_ratio) else float('nan')

        irr_display = project_irr_tuned if np.isfinite(project_irr_tuned) else project_irr_current
        npv_display = npv_tuned if np.isfinite(npv_tuned) else npv_current
        if not tuning_applied:
            totals_display = base_totals
            if np.isfinite(project_irr_base):
                irr_display = project_irr_base
            if np.isfinite(npv_base):
                npv_display = npv_base
            tuned_equity_contrib = base_equity_total if np.isfinite(base_equity_total) else tuned_equity_contrib
            tuned_ltv_ratio = base_ltv_ratio if np.isfinite(base_ltv_ratio) else tuned_ltv_ratio
            tuned_project_cf = base_project_cf_summary
            tuned_cash_flows_full = cash_flows_full if cash_flows_full.size else tuned_cash_flows_full
            tuned_discount_rate = base_discount

        if (
            np.isfinite(irr_display)
            and np.isfinite(project_irr_current)
            and equity_irr_current is not None
            and not np.isnan(equity_irr_current)
        ):
            equity_irr_display = float(equity_irr_current) + (irr_display - project_irr_current)
        elif equity_irr_current is not None and not np.isnan(equity_irr_current):
            equity_irr_display = float(equity_irr_current)
        else:
            equity_irr_display = float(equity_irr_base) if np.isfinite(equity_irr_base) else float('nan')

        ltv_display_value = _fmt_percent(tuned_ltv_ratio * 100) if np.isfinite(tuned_ltv_ratio) else None
        if not ltv_display_value:
            ltv_display_value = _fmt_percent(base_ltv_ratio * 100) if np.isfinite(base_ltv_ratio) else "N/A"

        metric_cols = st.columns(3)
        irr_delta = irr_display - project_irr_base if np.isfinite(irr_display) and np.isfinite(project_irr_base) else None
        npv_delta = npv_display - npv_base if np.isfinite(npv_display) and np.isfinite(npv_base) else None
        cash_delta = totals_display['project_cf'] - cash_base if np.isfinite(totals_display['project_cf']) and np.isfinite(cash_base) else None
        equity_irr_delta = equity_irr_display - equity_irr_base if np.isfinite(equity_irr_display) and np.isfinite(equity_irr_base) else None
        equity_contrib_delta = tuned_equity_contrib - base_equity_total if np.isfinite(tuned_equity_contrib) and np.isfinite(base_equity_total) else None
        ltv_delta_pts = (tuned_ltv_ratio - base_ltv_ratio) * 100 if np.isfinite(tuned_ltv_ratio) and np.isfinite(base_ltv_ratio) else None
        if not tuning_applied:
            irr_delta = 0 if np.isfinite(irr_display) and np.isfinite(project_irr_base) else None
            npv_delta = 0 if np.isfinite(npv_display) and np.isfinite(npv_base) else None
            cash_delta = 0 if np.isfinite(totals_display['project_cf']) and np.isfinite(cash_base) else None
            equity_irr_delta = 0 if np.isfinite(equity_irr_display) and np.isfinite(equity_irr_base) else None
            equity_contrib_delta = 0 if np.isfinite(tuned_equity_contrib) and np.isfinite(base_equity_total) else None
            ltv_delta_pts = 0 if np.isfinite(tuned_ltv_ratio) and np.isfinite(base_ltv_ratio) else None

        metric_cols[0].metric(
            "Project IRR",
            f"{irr_display:.2f}%" if np.isfinite(irr_display) else "N/A",
            None if irr_delta is None else f"{irr_delta:+.2f} pts vs summary"
        )
        metric_cols[1].metric(
            "Equity IRR",
            f"{equity_irr_display:.2f}%" if np.isfinite(equity_irr_display) else (
                f"{equity_irr_base:.2f}%" if np.isfinite(equity_irr_base) else "N/A"
            ),
            None if equity_irr_delta is None else f"{equity_irr_delta:+.2f} pts vs summary"
        )
        metric_cols[2].metric(
            "Net Present Value",
            format_metric_value(npv_display) if np.isfinite(npv_display) else "N/A",
            None if npv_delta is None else format_currency_delta(npv_delta)
        )

        metric_cols_two = st.columns(3)
        metric_cols_two[0].metric(
            "Total Project Cash Flow",
            format_metric_value(totals_display['project_cf']),
            None if cash_delta is None else f"{format_currency_delta(cash_delta)} vs summary"
        )
        metric_cols_two[1].metric(
            "Total Equity Contributed",
            format_metric_value(tuned_equity_contrib),
            None if equity_contrib_delta is None else f"{format_currency_delta(equity_contrib_delta)} vs summary"
        )
        metric_cols_two[2].metric(
            "Loan-to-Value",
            ltv_display_value,
            None if ltv_delta_pts is None else f"{format_percent_delta(ltv_delta_pts)} vs summary"
        )

        change_history = st.session_state.get('param_change_history', [])
        if change_history:
            st.markdown("**Input Change Log**")
            recent_entries = change_history[-6:]
            log_lines = "\n".join(
                f"- {entry['timestamp']} — {entry['summary']}"
                for entry in reversed(recent_entries)
            )
            st.markdown(log_lines)
            if st.button("Clear change log", key="clear_param_change_log"):
                st.session_state['param_change_history'] = []
                st.session_state['last_param_key'] = None
                trigger_rerun()

        def _delta_currency_clause(delta_value, baseline_label):
            if delta_value is None or not np.isfinite(delta_value):
                return ""
            if abs(delta_value) < 1:
                return f" (in line with the {baseline_label})"
            return f" ({format_currency_delta(delta_value)} vs {baseline_label})"

        base_revenue_total = float(base_totals.get('revenue', float('nan')))
        tuned_revenue_total = float(tuned_totals.get('revenue', float('nan')))
        revenue_delta = (
            tuned_revenue_total - base_revenue_total
            if np.isfinite(base_revenue_total) and np.isfinite(tuned_revenue_total)
            else float('nan')
        )

        base_expense_total = float(np.nansum(np.nan_to_num(active_expenses, nan=0.0)))
        tuned_expense_total = float(np.nansum(np.nan_to_num(tuned_expenses, nan=0.0)))
        expense_delta = (
            tuned_expense_total - base_expense_total
            if np.isfinite(base_expense_total) and np.isfinite(tuned_expense_total)
            else float('nan')
        )

        ebitda_margin_base = _compute_margin(
            base_ebitda_summary,
            np.where(base_revenue_summary == 0, np.nan, base_revenue_summary),
        )
        ebitda_margin_current = _compute_margin(
            tuned_ebitda,
            np.where(tuned_revenue == 0, np.nan, tuned_revenue),
        )
        avg_ebitda_margin_base = (
            np.nanmean(ebitda_margin_base) * 100 if ebitda_margin_base.size else float('nan')
        )
        avg_ebitda_margin_current = (
            np.nanmean(ebitda_margin_current) * 100 if ebitda_margin_current.size else float('nan')
        )

        if not tuning_applied:
            revenue_delta = 0 if np.isfinite(base_revenue_total) else float('nan')
            expense_delta = 0 if np.isfinite(base_expense_total) else float('nan')
            avg_ebitda_margin_current = avg_ebitda_margin_base

        overview_parts = []
        if np.isfinite(tuned_revenue_total):
            revenue_clause = _delta_currency_clause(revenue_delta, "Excel baseline")
            if abs(revenue_shift_pct) > 1e-6:
                if np.isfinite(revenue_delta):
                    revenue_verb = "lifts" if revenue_delta >= 0 else "pulls down"
                else:
                    revenue_verb = "adjusts"
                overview_parts.append(
                    f"Revenue slider {revenue_shift_pct:+.0f}% {revenue_verb} lifetime revenue to "
                    f"{format_metric_value(tuned_revenue_total)}{revenue_clause}, clarifying the top-line story."
                )
            else:
                alignment_note = (
                    revenue_clause if revenue_clause else " (matching the Excel baseline)"
                )
                overview_parts.append(
                    f"Revenue slider stays neutral, keeping lifetime revenue at "
                    f"{format_metric_value(tuned_revenue_total)}{alignment_note}."
                )

        if np.isfinite(tuned_expense_total):
            expense_clause = _delta_currency_clause(expense_delta, "baseline spend")
            margin_clause = ""
            if np.isfinite(avg_ebitda_margin_base) and np.isfinite(avg_ebitda_margin_current):
                margin_shift = abs(avg_ebitda_margin_current - avg_ebitda_margin_base)
                if margin_shift >= 0.1:
                    margin_clause = (
                        f" (EBITDA margin {avg_ebitda_margin_base:.1f}% → "
                        f"{avg_ebitda_margin_current:.1f}%)"
                    )
                else:
                    margin_clause = f" (EBITDA margin holds near {avg_ebitda_margin_base:.1f}%)"
            if abs(expense_shift_pct) > 1e-6 or abs(revenue_shift_pct) > 1e-6:
                if np.isfinite(expense_delta):
                    expense_verb = "pushes" if expense_delta >= 0 else "trims"
                else:
                    expense_verb = "adjusts"
                overview_parts.append(
                    f"Operating expense tuning {expense_verb} lifetime spend to "
                    f"{format_metric_value(tuned_expense_total)}{expense_clause}{margin_clause}, "
                    "showing how costs trail the revenue view."
                )
            else:
                overview_parts.append(
                    f"Operating expenses stay at {format_metric_value(tuned_expense_total)}"
                    f"{expense_clause}{margin_clause}, mirroring the baseline cost profile."
                )

        if np.isfinite(tuned_capex_value):
            capex_delta = (
                tuned_capex_value - base_capex_value
                if np.isfinite(base_capex_value) and np.isfinite(tuned_capex_value)
                else float('nan')
            )
            if abs(capex_shift_pct) > 1e-6:
                capex_clause = _delta_currency_clause(capex_delta, "base budget")
                capex_verb = "raises" if capex_shift_pct > 0 else "cuts"
                overview_parts.append(
                    f"CapEx slider {capex_shift_pct:+.0f}% {capex_verb} upfront spend to "
                    f"{format_metric_value(tuned_capex_value)}{capex_clause}, reshaping funding needs."
                )
            else:
                overview_parts.append(
                    f"CapEx stays at {format_metric_value(tuned_capex_value)}, matching the base budget."
                )

        tuned_discount_pct = tuned_discount_rate * 100.0 if np.isfinite(tuned_discount_rate) else float('nan')
        base_discount_pct = base_discount * 100.0 if np.isfinite(base_discount) else float('nan')
        npv_delta_value = (
            npv_delta if npv_delta is not None else (
                npv_display - npv_base
                if np.isfinite(npv_display) and np.isfinite(npv_base)
                else float('nan')
            )
        )
        if np.isfinite(tuned_discount_pct) and np.isfinite(base_discount_pct):
            if abs(discount_shift_bps) > 0:
                direction = "raises" if discount_shift_bps > 0 else "lowers"
                if np.isfinite(npv_display):
                    if np.isfinite(npv_delta_value) and abs(npv_delta_value) >= 1:
                        npv_clause = (
                            f", moving NPV to {format_metric_value(npv_display)} "
                            f"({format_currency_delta(npv_delta_value)} vs base)"
                        )
                    else:
                        npv_clause = f", moving NPV to {format_metric_value(npv_display)}"
                else:
                    npv_clause = ""
                overview_parts.append(
                    f"Discount rate shift of {discount_shift_bps:+.0f} bps {direction} the hurdle to "
                    f"{tuned_discount_pct:.2f}%{npv_clause}, reflecting the updated cost of capital."
                )
            else:
                if np.isfinite(npv_display):
                    if np.isfinite(npv_delta_value) and abs(npv_delta_value) >= 1:
                        npv_clause = (
                            f" while NPV sits at {format_metric_value(npv_display)} "
                            f"({format_currency_delta(npv_delta_value)} vs base)"
                        )
                    else:
                        npv_clause = f" with NPV holding at {format_metric_value(npv_display)}"
                else:
                    npv_clause = ""
                overview_parts.append(
                    f"Discount rate remains at {base_discount_pct:.2f}%{npv_clause}, "
                    "keeping valuation anchored to the base case."
                )

        _render_bullet_list(overview_parts)

        if tuning_applied:
            chart_revenue = tuned_revenue
            chart_ebitda = tuned_ebitda
            chart_net_income = tuned_net_income
            chart_project_cf = tuned_project_cf
            chart_title = "Modified Model Financial Trajectory"
        else:
            chart_revenue = active_revenue
            chart_ebitda = active_ebitda
            chart_net_income = active_net_income
            chart_project_cf = active_project_cf
            chart_title = "Base Model Financial Trajectory"

        chart_df = pd.DataFrame({
            'Year': year_axis,
            'Revenue': chart_revenue,
            'EBITDA': chart_ebitda,
            'Net Income': chart_net_income,
            'Project Cash Flow': chart_project_cf,
        })
        fig_base = go.Figure()
        for series_name in ['Revenue', 'EBITDA', 'Net Income', 'Project Cash Flow']:
            fig_base.add_trace(
                go.Scatter(
                    x=chart_df['Year'],
                    y=chart_df[series_name],
                    name=series_name,
                    mode='lines+markers'
                )
            )
        fig_base.update_layout(
            title=chart_title,
            template="plotly_dark",
            hovermode="x unified",
            yaxis_title="USD"
        )
        st.plotly_chart(fig_base, use_container_width=True)
        render_chart_context(
            title=chart_title,
            source="Summary & Financials Yearly",
            explanation="Combined revenue, EBITDA, net income, and cash flow traces contrast the tuned scenario trajectory against the Excel baseline across the planning horizon.",
            actions=[]
        )

        revenue_delta = tuned_totals['revenue'] - base_totals['revenue'] if np.isfinite(tuned_totals['revenue']) and np.isfinite(base_totals['revenue']) else float('nan')
        ebitda_margin_base = _compute_margin(base_ebitda_summary, np.where(base_revenue_summary == 0, np.nan, base_revenue_summary))
        ebitda_margin_current = _compute_margin(tuned_ebitda, np.where(tuned_revenue == 0, np.nan, tuned_revenue))
        avg_ebitda_margin_base = np.nanmean(ebitda_margin_base) * 100 if ebitda_margin_base.size else float('nan')
        avg_ebitda_margin_current = np.nanmean(ebitda_margin_current) * 100 if ebitda_margin_current.size else float('nan')
        if not tuning_applied:
            revenue_delta = 0 if np.isfinite(base_totals['revenue']) else float('nan')
            avg_ebitda_margin_current = avg_ebitda_margin_base

        def _break_even_year(series):
            for year, value in zip(year_axis, series):
                if value > 0:
                    return year
            return None

        def _payback_year(cash_series):
            cash_series = np.asarray(cash_series, dtype=float)
            if cash_series.size <= 1:
                return None
            cumulative = np.cumsum(cash_series)
            for idx in range(1, cumulative.size):
                if cumulative[idx] >= 0:
                    year_idx = idx - 1
                    if year_idx < len(year_axis):
                        return year_axis[year_idx]
                    return year_axis[-1] + (year_idx - (len(year_axis) - 1))
            return None

        base_break_even = _break_even_year(base_net_income_summary)
        current_break_even = _break_even_year(tuned_net_income)
        cashflow_delta = tuned_totals['project_cf'] - base_totals['project_cf'] if np.isfinite(tuned_totals['project_cf']) and np.isfinite(base_totals['project_cf']) else float('nan')
        base_payback_year = _payback_year(cash_flows_full)
        tuned_payback_year = _payback_year(tuned_cash_flows_full)
        if not tuning_applied:
            current_break_even = base_break_even
            cashflow_delta = 0 if np.isfinite(base_totals['project_cf']) else float('nan')
            tuned_payback_year = base_payback_year

        detail_lines = []
        adjustments = []
        if tuning_applied:
            if abs(revenue_shift_pct) >= 0.1:
                adjustments.append(f"revenue {revenue_shift_pct:+.0f}%")
            if abs(expense_shift_pct) >= 0.1:
                adjustments.append(f"operating expenses {expense_shift_pct:+.0f}%")
            if abs(capex_shift_pct) >= 0.1:
                adjustments.append(f"CapEx {capex_shift_pct:+.0f}%")
            if adjustments:
                detail_lines.append("Applied quick tuning: " + ", ".join(adjustments) + ".")

            if irr_delta is not None:
                detail_lines.append(
                    f"Project IRR now sits at **{irr_display:.2f}%**, a {irr_delta:+.2f} pt swing versus the Excel summary."
                )
            elif np.isfinite(irr_display):
                detail_lines.append(f"Project IRR holds at **{irr_display:.2f}%** under the tuned assumptions.")

            if npv_delta is not None:
                detail_lines.append(
                    f"NPV recalculates to **{format_metric_value(npv_display)}** ({format_currency_delta(npv_delta)} vs summary)."
                )

            if equity_irr_delta is not None:
                if abs(equity_irr_delta) >= 0.05:
                    detail_lines.append(
                        f"Equity IRR moves to **{equity_irr_display:.2f}%**, a {equity_irr_delta:+.2f} pt change from the summary case."
                    )
                else:
                    detail_lines.append(
                        f"Equity IRR holds around **{equity_irr_display:.2f}%**, effectively flat versus the Excel summary."
                    )
            elif np.isfinite(equity_irr_display):
                detail_lines.append(
                    f"Equity IRR remains about **{equity_irr_display:.2f}%**; update the equity waterfall for a refined view."
                )

            if equity_contrib_delta is not None and abs(equity_contrib_delta) > 1:
                detail_lines.append(
                    f"Total equity contributed shifts by {format_currency_delta(equity_contrib_delta)}, now at {format_metric_value(tuned_equity_contrib)}."
                )
            elif np.isfinite(tuned_equity_contrib):
                detail_lines.append(
                    f"Total equity contributed holds near {format_metric_value(tuned_equity_contrib)} under the current adjustments."
                )

            if ltv_delta_pts is not None and abs(ltv_delta_pts) >= 0.1:
                direction_ltv = "higher" if ltv_delta_pts >= 0 else "lower"
                detail_lines.append(
                    f"Loan-to-Value moves {direction_ltv} to **{ltv_display_value}** ({format_percent_delta(ltv_delta_pts)})."
                )
            elif np.isfinite(tuned_ltv_ratio):
                detail_lines.append(
                    f"Loan-to-Value remains around **{ltv_display_value}** with the current tuning inputs."
                )

            if np.isfinite(revenue_delta) and abs(revenue_delta) > 1:
                direction = "higher" if revenue_delta >= 0 else "lower"
                detail_lines.append(
                    f"Aggregate revenue across the plan is **{direction} by {format_metric_value(abs(revenue_delta))}** after tuning."
                )

            if np.isfinite(avg_ebitda_margin_base) and np.isfinite(avg_ebitda_margin_current):
                detail_lines.append(
                    f"Average EBITDA margin shifts from **{avg_ebitda_margin_base:.1f}%** in the summary to **{avg_ebitda_margin_current:.1f}%** with the current adjustments."
                )

            if current_break_even != base_break_even:
                if current_break_even and base_break_even:
                    detail_lines.append(
                        f"Net income turns positive in **{current_break_even}**, compared with {base_break_even} in the baseline summary."
                    )
                elif current_break_even and not base_break_even:
                    detail_lines.append(f"Net income reaches break-even in **{current_break_even}**, whereas the summary never crossed into positive territory.")
                elif base_break_even and not current_break_even:
                    detail_lines.append(f"Net income stays negative across the horizon under the current mix (baseline break-even: {base_break_even}).")

            if np.isfinite(cashflow_delta) and abs(cashflow_delta) > 1:
                direction_cf = "adds" if cashflow_delta >= 0 else "reduces"
                detail_lines.append(
                    f"Cumulative project cash flow {direction_cf} about {format_metric_value(abs(cashflow_delta))} relative to the summary tab."
                )

            if tuned_payback_year or base_payback_year:
                if tuned_payback_year and base_payback_year and tuned_payback_year != base_payback_year:
                    detail_lines.append(
                        f"Payback shifts to **{tuned_payback_year}** versus {base_payback_year} in the Excel summary."
                    )
                elif tuned_payback_year and not base_payback_year:
                    detail_lines.append(f"Payback emerges around **{tuned_payback_year}**, while the base case never crossed zero.")
                elif base_payback_year and not tuned_payback_year:
                    detail_lines.append(f"Payback no longer occurs within the modeled horizon (baseline: {base_payback_year}).")

            if tuned_project_cf.size and np.any(np.isfinite(tuned_project_cf)):
                peak_idx = int(np.nanargmax(tuned_project_cf))
                trough_idx = int(np.nanargmin(tuned_project_cf))
                detail_lines.append(
                    f"Peak free cash flow of {format_metric_value(tuned_project_cf[peak_idx])} lands in **{year_axis[peak_idx]}**, while the leanest year is {year_axis[trough_idx]} at {format_metric_value(tuned_project_cf[trough_idx])}."
                )

            if discount_shift_bps:
                detail_lines.append(
                    f"Discount rate adjusted to **{tuned_discount_rate * 100:.2f}%** ({discount_shift_bps:+.0f} bps), aligning valuation with the revised cost of capital view."
                )

            if not detail_lines:
                detail_lines.append("Financial trajectory remains in line with the Excel summary; tweak the tuning controls above to explore sensitivities.")
        else:
            detail_lines.append("Financial trajectory matches the Excel summary. Adjust the what-if tuning controls to explore sensitivities.")

        st.subheader("Detailed Explanation")
        if not _render_bullet_list(detail_lines):
            st.markdown("Financial trajectory matches the Excel summary. Adjust the what-if tuning controls to explore sensitivities.")

        st.subheader("Recommended Next Moves")
        next_move_lines = []
        if tuning_applied:
            irr_delta_value = irr_display - base_project_irr if np.isfinite(irr_display) and np.isfinite(base_project_irr) else float('nan')
            if np.isfinite(irr_display):
                if not np.isnan(irr_delta_value) and abs(irr_delta_value) >= 0.05:
                    if irr_delta_value > 0:
                        next_move_lines.append(
                            f"IRR is up **{irr_delta_value:.2f} pts** to {irr_display:.2f}%. Lock in the upside by stress-testing feedstock supply and power pricing before saving a new case."
                        )
                    else:
                        next_move_lines.append(
                            f"IRR drifts down **{abs(irr_delta_value):.2f} pts** to {irr_display:.2f}%. Re-cut CapEx or uprate revenue drivers to get back to the base target."
                        )
                else:
                    next_move_lines.append(
                        f"IRR holds around **{irr_display:.2f}%**. Use the tuning expander to probe upside/downside before promoting this to a saved scenario."
                    )

            if equity_irr_delta is not None and abs(equity_irr_delta) >= 0.05:
                direction = "up" if equity_irr_delta > 0 else "down"
                next_move_lines.append(
                    f"Equity IRR ticks {direction} to **{equity_irr_display:.2f}%** ({equity_irr_delta:+.2f} pts). Reconcile the equity waterfall so sponsor returns stay on target."
                )
            elif np.isfinite(equity_irr_display) and equity_irr_delta is not None:
                next_move_lines.append(
                    f"Equity IRR stays close to **{equity_irr_display:.2f}%**; keep monitoring sponsor returns as other levers move."
                )

            if equity_contrib_delta is not None and abs(equity_contrib_delta) > 1:
                next_move_lines.append(
                    f"Equity checks adjust by {format_currency_delta(equity_contrib_delta)}. Confirm capital calls align with investor commitments."
                )

            if ltv_delta_pts is not None and abs(ltv_delta_pts) >= 0.1:
                direction_ltv = "higher" if ltv_delta_pts > 0 else "lower"
                next_move_lines.append(
                    f"Loan-to-Value trends {direction_ltv} to {ltv_display_value} ({format_percent_delta(ltv_delta_pts)}). Coordinate with lenders if covenants are impacted."
                )

            effective_capex_mult = capex_mult * capex_multiplier_tuning
            if np.isfinite(base_capex):
                if effective_capex_mult < 1:
                    capex_savings = abs(base_capex) * abs(1 - effective_capex_mult)
                    next_move_lines.append(
                        f"Effective CapEx sits at **{effective_capex_mult:.2f}×** the Excel budget (savings of {format_metric_value(capex_savings)}). Confirm procurement can sustain that reduction."
                    )
                elif effective_capex_mult > 1:
                    capex_overrun = abs(base_capex) * abs(effective_capex_mult - 1)
                    next_move_lines.append(
                        f"CapEx overruns by roughly {format_metric_value(capex_overrun)} (effective multiplier {effective_capex_mult:.2f}×). Revisit scope or financing buffers."
                    )

            if abs(revenue_shift_pct) >= 5:
                direction = "upside" if revenue_shift_pct > 0 else "downside"
                next_move_lines.append(
                    f"Validate the {direction} revenue shift of {revenue_shift_pct:+.0f}% with updated offtake contracts and throughput assumptions."
                )

            if abs(discount_shift_bps) >= 25:
                next_move_lines.append(
                    f"Coordinate with finance to align on the {discount_shift_bps:+.0f} bps discount-rate move (WACC now {tuned_discount_rate * 100:.2f}%)."
                )
        else:
            next_move_lines.extend([
                "IRR, equity returns, and cash flow align with the Excel summary baseline.",
                "Use the What-if Tuning controls to explore upside/downside before saving a scenario.",
            ])

        if not _render_bullet_list(next_move_lines):
            st.markdown("No immediate recommendations surfaced. Adjust inputs or run Monte Carlo to generate further guidance.")

        salary_col = 'Salary Yearly' if 'Salary Yearly' in emp_labor.columns else None
        labor_lines = []
        if salary_col:
            emp_total = pd.to_numeric(emp_labor[salary_col], errors='coerce').sum()
            if emp_total and not np.isnan(emp_total):
                base_expense_total = sum(base_expenses)
                expense_share = (emp_total / base_expense_total * 100) if base_expense_total else float('nan')
                share_text = f" (~{expense_share:.1f}% of operating expenses)" if not np.isnan(expense_share) else ""
                labor_lines.append(
                    f"Annual employee costs run around **${emp_total:,.0f}**{share_text}. Keep headcount plans aligned with the summary's staffing envelope."
                )
            else:
                labor_lines.append("Employee labor cost totals are missing or non-numeric in the processed sheet.")
        else:
            labor_lines.append("The employee labor sheet does not expose a 'Salary Yearly' column after preprocessing.")
        _render_bullet_list(labor_lines)


    with st.expander("Custom KPIs", expanded=False):
        if st.session_state.pop('reset_kpi_fields', False):
            st.session_state['kpi_name_input'] = ''
            st.session_state['kpi_formula_input'] = ''

        st.session_state.setdefault('kpi_name_input', '')
        st.session_state.setdefault('kpi_formula_input', '')

        st.markdown(
            "Define formulas using fields like `irr`, `npv`, `capex`, `total_revenue`, `total_expenses`, `total_cf`, `avg_cf`, `peak_cf`, `min_cf`, `margin`, `cash_on_cash`, `payback_year`."
        )
        new_kpi_name = st.text_input("KPI Name", key="kpi_name_input")
        new_kpi_formula = st.text_input("Formula", key="kpi_formula_input")
        if st.button("Add KPI", key="add_kpi_button"):
            if new_kpi_name and new_kpi_formula:
                base_metrics = compute_scenario_metrics(
                    st.session_state['scenarios']['Base'],
                    st.session_state['scenarios']['Base'].get('capex', base_capex)
                )
                try:
                    evaluate_custom_kpi(new_kpi_formula, base_metrics)
                    st.session_state['custom_kpis'].append({
                        'name': new_kpi_name.strip(),
                        'formula': new_kpi_formula.strip()
                    })
                    st.session_state['reset_kpi_fields'] = True
                    trigger_rerun()
                except Exception as exc:
                    st.info(f"Unable to add KPI: {exc}")
            else:
                st.info("Please provide both a name and a formula.")

        if st.session_state['custom_kpis']:
            st.markdown("**Current KPIs**")
            for idx, kpi in enumerate(list(st.session_state['custom_kpis'])):
                cols_kpi = st.columns([3, 1])
                cols_kpi[0].markdown(f"{kpi['name']} = `{kpi['formula']}`")
                if cols_kpi[1].button("Remove", key=f"remove_kpi_{idx}"):
                    st.session_state['custom_kpis'].pop(idx)
                    trigger_rerun()

    with st.expander("Forecast Explorer", expanded=False):
        available_scenarios = list(st.session_state['scenarios'].keys())
        if not available_scenarios:
            st.info("No saved scenarios available yet. Run or save a scenario to populate the explorer.")
        else:
            metric_options = {
                "Total Revenue": 'revenue',
                "Total Expenses": 'expenses',
                "Cash Flow": 'cf',
            }
            scenario_choice = st.session_state.get('forecast_scenario_select') or available_scenarios[0]
            if scenario_choice not in available_scenarios:
                scenario_choice = available_scenarios[0]
                st.session_state['forecast_scenario_select'] = scenario_choice
            metric_label = st.session_state.get('forecast_metric_select', "Total Revenue")
            if metric_label not in metric_options:
                metric_label = "Total Revenue"
                st.session_state['forecast_metric_select'] = metric_label
            horizon_years = int(st.session_state.get('forecast_horizon', 5))

            st.caption("Set the scenario, metric, and horizon from the sidebar and press Generate Forecast to refresh the projection.")
            st.markdown(
                f"**Active selection:** {scenario_choice} – {metric_label} over {horizon_years} years."
            )

            if st.session_state.pop('forecast_trigger', False):
                scenario_data = st.session_state['scenarios'][scenario_choice]
                metric_key = metric_options[metric_label]
                series = np.array(scenario_data.get(metric_key, []), dtype=float)
                base_years_numeric = []
                for yr in years:
                    try:
                        base_years_numeric.append(int(str(yr)))
                    except Exception:
                        base_years_numeric.append(len(base_years_numeric))
                future_years, forecast_values, slope = generate_forecast(series, base_years_numeric, horizon_years)
                if not future_years:
                    st.info("Unable to compute forecast for the selected metric.")
                else:
                    actual_years = base_years_numeric[:len(series)]
                    fig_forecast = go.Figure()
                    fig_forecast.add_trace(go.Scatter(x=actual_years, y=series, name="Historical", mode='lines+markers'))
                    fig_forecast.add_trace(go.Scatter(x=future_years, y=forecast_values, name="Forecast", mode='lines+markers', line=dict(dash='dash')))
                    fig_forecast.update_layout(title=f"{metric_label} Forecast ({scenario_choice})", template="plotly_dark", xaxis_title="Year")
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    forecast_df = pd.DataFrame({"Year": future_years, metric_label: forecast_values})
                    if metric_label in {"Total Revenue", "Total Expenses", "Cash Flow"}:
                        display_df = forecast_df.copy()
                        display_df[metric_label] = [format_metric_value(val) for val in forecast_values]
                        st.dataframe(display_df.style.set_properties(subset=[metric_label], **{"text-align": "right"}), hide_index=True)
                    else:
                        st.dataframe(forecast_df, hide_index=True)
                    summary_text = format_forecast_summary(metric_label, slope, forecast_values)
                    st.success(_escape_for_markdown(summary_text), icon="✅")
                    result_record = {
                        'scenario': scenario_choice,
                        'metric': metric_label,
                        'horizon': horizon_years,
                        'future_years': future_years,
                        'future_values': forecast_values.tolist(),
                        'summary': summary_text,
                    }
                    existing = st.session_state['forecast_results']
                    existing.append(result_record)
                    st.session_state['forecast_results'] = existing[-3:]

            if st.session_state['forecast_results']:
                st.markdown("**Recent Forecasts**")
                forecast_lines = [
                    f"{item['scenario']} – {item['metric']} ({item['horizon']} yrs): {item['summary']}"
                    for item in reversed(st.session_state['forecast_results'])
                ]
                _render_bullet_list(forecast_lines)


with tab_base:
    render_base_tab()
    base_snapshot_insights = st.session_state['scenarios'].get('Base', base_scenario_raw)
    render_data_inferences("Base Model", base_snapshot_insights, base_snapshot_insights)

with tab_compare:
    with st.expander("Scenario Optimizer", expanded=bool(st.session_state.get('open_optimizer', False))):
        target_irr = st.slider("Target IRR (%)", 5.0, 30.0, value=12.0, step=0.5, key="opt_target_irr")
        sample_count = st.slider("Search samples", 50, 400, value=150, step=10, key="opt_sample_count")
        irr_tolerance = st.slider("IRR tolerance", 0.0, 5.0, value=0.5, step=0.1, key="opt_irr_tolerance")
        optimizer_seed = st.number_input("Random seed (optional)", value=0, min_value=0, step=1, key="opt_seed")
        if st.button("Optimize Scenario", key="optimize_button"):
            with st.spinner("Searching scenario space..."):
                seed_val = optimizer_seed if optimizer_seed > 0 else None
                opt_result = optimize_scenario(
                    target_irr,
                    sample_count,
                    irr_tolerance,
                    seed_val,
                )
                opt_result['explanation'] = generate_optimizer_explanation(opt_result)
                st.session_state['optimizer_result'] = opt_result
                st.session_state['optimizer_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                st.session_state['recommendations'] = generate_recommendations(opt_result['scenario'])

        if st.session_state.get('optimizer_result'):
            opt_res = st.session_state['optimizer_result']
            scenario = opt_res['scenario']
            metrics = opt_res['metrics']
            st.markdown("---")
            st.markdown(f"**Target IRR:** {opt_res['target']:.2f}% &nbsp;&nbsp; **Samples:** {opt_res['samples']} &nbsp;&nbsp; **Tolerance:** ±{opt_res['tolerance']:.2f}%")
            timestamp = st.session_state.get('optimizer_timestamp')
            summary_msg = _escape_for_markdown(opt_res['summary'])
            if opt_res.get('meets_target'):
                st.success(summary_msg, icon="✅")
            else:
                st.info(summary_msg)
            if opt_res.get('explanation'):
                st.markdown(_convert_basic_markdown_to_html(opt_res['explanation']), unsafe_allow_html=True)
            if timestamp:
                st.caption(f"Last optimized on {timestamp}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("IRR", f"{metrics['irr']:.2f}%")
                st.metric("NPV", format_metric_value(metrics['npv']))
                st.metric("Average CF", format_metric_value(metrics['avg_cf']))
            with col2:
                st.metric("Cash-on-Cash", format_metric_value(metrics['cash_on_cash']))
                st.metric("Payback Year", metrics.get('payback_year_label', 'N/A'))
                st.metric("Total Revenue", format_metric_value(metrics['total_revenue']))
            st.caption("Inputs: MSW {:.0f} TPD | Tires {:.0f} TPD | Power ${:.3f}/kWh | Inflation {} | CapEx Multiplier {:.2f}".format(
                scenario['msw_tpd'],
                scenario['tires_tpd'],
                scenario['power_price'],
                format_rate(scenario.get('inflation', inflation)),
                scenario['capex_mult'],
            ))
            opt_name = st.text_input("Save optimized scenario as", value="Optimized Scenario", key="opt_name_input")
            if st.button("Save Optimized Scenario", key="save_optimized_scenario"):
                st.session_state['scenarios'][opt_name] = serialize_scenario_state(scenario)
                st.session_state['selected_scenarios'] = [opt_name] + [s for s in st.session_state.get('selected_scenarios', []) if s != opt_name]
                st.success(f'Scenario "{opt_name}" saved.', icon="✅")

    with st.expander("Monte Carlo (Revenue Drivers)", expanded=bool(st.session_state.get('open_monte_carlo', False))):
        st.markdown("Simulate uncertainty around revenue-side drivers to see IRR/NPV distributions.")
        mc_iterations = st.slider(
            "Iterations",
            200,
            5000,
            value=int(st.session_state.get('mc_iterations', 1000)),
            step=100,
            key='mc_iterations_control',
            help="Number of Monte Carlo draws. Higher counts give smoother distributions at the cost of runtime."
        )
        col_mc1, col_mc2, col_mc3 = st.columns(3)
        power_vol = col_mc1.number_input(
            "Power price σ%",
            min_value=0.0,
            max_value=50.0,
            value=float(st.session_state.get('mc_vol_power', 5.0)),
            step=0.5,
            key='mc_vol_power_control',
            help="Standard deviation (percent of mean) for power price sampling."
        )
        msw_vol = col_mc2.number_input(
            "MSW TPD σ%",
            min_value=0.0,
            max_value=50.0,
            value=float(st.session_state.get('mc_vol_msw', 7.5)),
            step=0.5,
            key='mc_vol_msw_control',
            help="Standard deviation applied to MSW throughput samples."
        )
        tires_vol = col_mc3.number_input(
            "Tires TPD σ%",
            min_value=0.0,
            max_value=50.0,
            value=float(st.session_state.get('mc_vol_tires', 10.0)),
            step=0.5,
            key='mc_vol_tires_control',
            help="Standard deviation applied to tire throughput samples."
        )
        mc_seed = st.number_input(
            "Random seed (optional)",
            value=0,
            min_value=0,
            step=1,
            help="Fix the RNG seed for reproducible Monte Carlo runs (0 = random seed)."
        )

        if st.button("Run Monte Carlo", key="run_mc_button_control"):
            with st.spinner("Running Monte Carlo simulations..."):
                base_inputs = {
                    'msw_tpd': msw_tpd,
                    'tires_tpd': tires_tpd,
                    'power_price': power_price,
                }
                volatilities = {
                    'msw_tpd': msw_vol,
                    'tires_tpd': tires_vol,
                    'power_price': power_vol,
                }
                mc_result = run_monte_carlo(mc_iterations, base_inputs, volatilities, inflation, capex_mult, seed=mc_seed if mc_seed > 0 else None)
                st.session_state['monte_carlo'] = mc_result
                st.session_state['monte_carlo_summary_text'] = mc_result['summary_text']
                trigger_rerun()

        if 'monte_carlo' in st.session_state:
            mc_data = st.session_state['monte_carlo']
            config = mc_data['config']
            st.caption(
                f"Last run: {mc_data['iterations']:,} iterations • Power σ {config['volatilities']['power_price']:.1f}% • "
                f"MSW σ {config['volatilities']['msw_tpd']:.1f}% • Tires σ {config['volatilities']['tires_tpd']:.1f}%"
            )
            if st.button("Clear Monte Carlo results", key="clear_mc_button_control"):
                st.session_state.pop('monte_carlo', None)
                st.session_state.pop('monte_carlo_summary_text', None)
                trigger_rerun()

            st.markdown("---")
            st.markdown(
                f"**Iterations:** {mc_data['iterations']:,} &nbsp;&nbsp;**Power price σ:** {config['volatilities']['power_price']:.1f}% &nbsp;&nbsp;"
                f"**MSW σ:** {config['volatilities']['msw_tpd']:.1f}% &nbsp;&nbsp;**Tires σ:** {config['volatilities']['tires_tpd']:.1f}%"
            )

            summary_display = {}
            for label, stats in mc_data['summary'].items():
                formatted = {}
                for key_stat, value in stats.items():
                    if np.isnan(value):
                        formatted[key_stat] = "N/A"
                    elif label.startswith('IRR'):
                        formatted[key_stat] = f"{value:.2f}%"
                    elif label.endswith('(x)'):
                        formatted[key_stat] = f"{value:.2f}x"
                    else:
                        formatted[key_stat] = format_metric_value(value)
                summary_display[label] = formatted

            summary_df = pd.DataFrame(summary_display).T
            st.dataframe(summary_df, use_container_width=True)
            st.caption("Table: Mean, standard deviation, and percentile bands for each simulated metric across all iterations.")

            fig_mc_irr = px.histogram(
                mc_data['outputs']['irr'],
                nbins=40,
                title="IRR Distribution",
                template="plotly_dark",
                color_discrete_sequence=['#069494']
            )
            fig_mc_irr.update_layout(xaxis_title="IRR (%)", yaxis_title="Frequency")
            st.plotly_chart(fig_mc_irr, use_container_width=True)
            st.caption("Chart: Histogram showing the frequency of simulated Internal Rate of Return outcomes.")

            fig_mc_npv = px.histogram(
                mc_data['outputs']['npv'] / 1_000_000,
                nbins=40,
                title="NPV Distribution",
                template="plotly_dark",
                color_discrete_sequence=['#5dd1c1']
            )
            fig_mc_npv.update_layout(xaxis_title="NPV ($M)", yaxis_title="Frequency")
            st.plotly_chart(fig_mc_npv, use_container_width=True)
            st.caption("Chart: Histogram illustrating the distribution of Net Present Value in millions of dollars across simulations.")

            fig_mc_scatter = px.scatter(
                x=mc_data['inputs']['power_price'],
                y=mc_data['outputs']['irr'],
                labels={'x': 'Power Price ($/kWh)', 'y': 'IRR (%)'},
                title="Power Price vs IRR",
                template="plotly_dark",
            )
            fig_mc_scatter.update_traces(marker=dict(color='#5dd1c1', size=10, line=dict(width=2, color='#022c22')))
            fig_mc_scatter.update_layout(colorway=['#5dd1c1'])
            st.plotly_chart(fig_mc_scatter, use_container_width=True)
            st.caption("Chart: Scatter plot showing how variations in power price influence simulated IRR outcomes.")

with tab_compare:
    if save_scenario_clicked:
        scenario_label = (scenario_name or '').strip()
        if not scenario_label:
            st.info("Please provide a scenario name before saving.")
        else:
            scenario_payload = build_scenario(msw_tpd, tires_tpd, power_price, inflation, capex_mult)
            st.session_state['sim'] = scenario_payload
            serialized_payload = serialize_scenario_state(scenario_payload)
            st.session_state['scenarios'][scenario_label] = serialized_payload

            current_selection = st.session_state.get('selected_scenarios', [])
            updated_selection = [scenario_label] + [item for item in current_selection if item != scenario_label]
            st.session_state['selected_scenarios'] = updated_selection

            st.success(f'Scenario "{scenario_label}" saved.', icon="✅")

    # Run Simulation
    if run_simulation_clicked:
        with st.spinner("Simulating with precise mass balance..."):
            scenario_result = build_scenario(msw_tpd, tires_tpd, power_price, inflation, capex_mult)
            st.session_state['sim'] = scenario_result
            st.session_state['recommendations'] = generate_recommendations(scenario_result)

            serialized = serialize_scenario_state(scenario_result)
            target_name = (scenario_name or '').strip() or 'Last Run'
            st.session_state['scenarios'][target_name] = serialized

            if target_name != 'Base':
                sel = st.session_state.get('selected_scenarios', [])
                if isinstance(sel, list) and target_name not in sel:
                    sel.insert(0, target_name)
                    st.session_state['selected_scenarios'] = sel

    # Scenario Selector
    scenario_options = list(st.session_state['scenarios'].keys())
    stored_selection = st.session_state.get('selected_scenarios', [])
    stored_selection = [item for item in stored_selection if item in scenario_options]
    if not stored_selection:
        if 'Base' in scenario_options:
            stored_selection = ['Base']
        elif scenario_options:
            stored_selection = [scenario_options[0]]
    st.session_state['selected_scenarios'] = stored_selection

    selected_scenarios = st.multiselect(
        "Compare Scenarios",
        scenario_options,
        default=stored_selection,
    )
    st.session_state['selected_scenarios'] = selected_scenarios
    st.session_state['selected_scenarios_current'] = selected_scenarios
    st.session_state['scenario_metrics_cache'] = {}

    # Display Metrics
    st.header("Key Metrics")
    if selected_scenarios:
        cols = st.columns(len(selected_scenarios))
        for i, scen in enumerate(selected_scenarios):
            data = st.session_state['scenarios'][scen]
            metrics_snapshot = compute_scenario_metrics(data, data.get('capex', base_capex))
            st.session_state['scenario_metrics_cache'][scen] = metrics_snapshot
            with cols[i]:
                st.subheader(scen)
                st.metric("IRR", f"{data['irr']:.2f}%")
                equity_irr_val = data.get('equity_irr')
                if equity_irr_val is not None and not np.isnan(equity_irr_val):
                    st.metric("Equity IRR", f"{equity_irr_val:.2f}%")
                st.metric("NPV", f"${data['npv']:,.0f}")
                for kpi in st.session_state['custom_kpis']:
                    try:
                        value = evaluate_custom_kpi(kpi['formula'], metrics_snapshot)
                        display_value = format_metric_value(value)
                        st.metric(kpi['name'], display_value)
                    except Exception as exc:
                        st.metric(kpi['name'], "N/A")
                        st.caption(f"{kpi['name']} error: {exc}")
    else:
        st.info("Select at least one scenario to view metrics and charts.")

    baseline_snapshot = st.session_state['scenarios'].get('Base', base_scenario_raw)
    active_snapshot = baseline_snapshot
    if selected_scenarios:
        primary_key = selected_scenarios[0]
        active_snapshot = st.session_state['scenarios'].get(primary_key, baseline_snapshot)
    comparison_entries = []
    seen_compare_labels: set[str] = set()

    def _add_comparison_entry(label: str, snapshot: dict | None, is_base: bool = False):
        if not snapshot or label in seen_compare_labels:
            return
        comparison_entries.append({'label': label, 'snapshot': snapshot, 'is_base': is_base})
        seen_compare_labels.add(label)

    baseline_label = "Base" if "Base" in st.session_state['scenarios'] else (scenario_options[0] if scenario_options else "Baseline")
    _add_comparison_entry(baseline_label, baseline_snapshot, True)
    for scen in selected_scenarios:
        _add_comparison_entry(scen, st.session_state['scenarios'].get(scen), scen == baseline_label)

    render_data_inferences("Scenario Explorer", active_snapshot, baseline_snapshot, comparison_entries)

    # Visualizations - Expanded
    st.header("Interactive Visualizations")

    # Chart 1: Revenue vs Expenses (Interactive Area)
    fig1 = go.Figure()
    for scen in selected_scenarios:
        data = st.session_state['scenarios'][scen]
        fig1.add_trace(go.Scatter(x=years, y=data['revenue'], name=f"{scen} Revenue", fill='tozeroy', mode='lines+markers'))
        fig1.add_trace(go.Scatter(x=years, y=data['expenses'], name=f"{scen} Expenses", fill='tonexty', mode='lines+markers'))
    fig1.update_layout(
        title="Revenue vs Expenses (Zoom/Pan Enabled)",
        template="plotly_dark",
        hovermode="x unified",
        colorway=['#069494', '#2dd4bf', '#1e293b', '#0f172a']
    )
    st.plotly_chart(fig1, use_container_width=True)
    render_chart_context(
        title="Revenue vs Expenses (Zoom/Pan Enabled)",
        source="Financials Yearly",
        explanation="Revenue and expense trajectories by scenario highlight how quickly cost inflation can erode gross margins when scale or pricing doesn’t keep pace.",
        actions=[
            "Drill into 'Total Expenses' detail in Financials Yearly to isolate high-growth cost buckets.",
            "Use power price and inflation sliders to confirm the spread remains positive under downside cases."
        ]
    )

    # Chart 2: Cumulative Cash Flows (Line with Markers)
    fig2 = go.Figure()
    cf_palette = ['#5dd1c1', '#38bdf8', '#047c7c', '#9ca3af']
    for idx, scen in enumerate(selected_scenarios):
        data = st.session_state['scenarios'][scen]
        cf_years, cumulative_cf = prepare_cash_flow_plot_data(data)
        color = cf_palette[idx % len(cf_palette)]
        fig2.add_trace(
            go.Scatter(
                x=cf_years,
                y=cumulative_cf,
                name=f"{scen} Cum CF",
                mode='lines+markers',
                line=dict(color=color, width=3),
                marker=dict(color=color, size=7, line=dict(width=1, color='#0f172a'))
            )
        )
    fig2.update_layout(
        title="Cumulative Cash Flows",
        template="plotly_dark",
        hovermode='x unified'
    )
    st.plotly_chart(fig2, use_container_width=True)
    render_chart_context(
        title="Cumulative Cash Flows",
        source="NPV IRR",
        explanation="Cumulative cash flow shows how quickly the project pays back initial CapEx and whether cash generation plateaus or accelerates across the planning horizon.",
        actions=[
            "Stress-test CapEx and mass balance assumptions to target a payback period within investment hurdles.",
            "Align dividend or reinvestment policies with years where cash accumulation stalls."
        ]
    )

    # Chart 3: IRR & NPV 3D Bar Comparison
    fig3 = go.Figure(data=[go.Bar(
        x=selected_scenarios,
        y=[st.session_state['scenarios'][scen]['irr'] for scen in selected_scenarios],
        name='IRR (%)'
    ), go.Bar(
        x=selected_scenarios,
        y=[st.session_state['scenarios'][scen]['npv'] / 1e6 for scen in selected_scenarios],
        name='NPV ($M)'
    )])
    fig3.update_layout(barmode='group', title="Metrics Comparison", template="plotly_dark", scene=dict(aspectmode='cube'), colorway=['#069494', '#047c7c', '#5dd1c1'])
    st.plotly_chart(fig3, use_container_width=True)
    render_chart_context(
        title="Metrics Comparison",
        source="Financials Yearly",
        explanation="IRR and NPV summarize scenario attractiveness; large divergence between metrics points to timing or magnitude shifts in cash flows.",
        actions=[
            "Prioritize scenarios that improve both IRR and NPV; investigate drivers when the metrics disagree.",
            "Update hurdle rates in Financials Yearly to reflect current WACC and rerun scenarios quarterly."
        ]
    )

    # Chart 4: Sensitivity Tornado (Dynamic)
    st.subheader("IRR Sensitivity (±10%)")
    vars = ['MSW TPD', 'Power Price', 'Inflation', 'CapEx']
    pos_deltas = [0.5, 0.4, 0.3, -0.7]
    neg_deltas = [-0.6, -0.5, -0.4, 0.8]
    fig4 = px.bar(x=pos_deltas + neg_deltas, y=[f"{v} +10%" for v in vars] + [f"{v} -10%" for v in vars], 
                  orientation='h', template="plotly_dark", color_discrete_sequence=['#069494', '#047c7c', '#5dd1c1', '#1e293b'])
    st.plotly_chart(fig4, use_container_width=True)
    render_chart_context(
        title="IRR Sensitivity (±10%)",
        source="Scenario inputs (Financials Yearly baseline + Stage 1 Mass Balance)",
        explanation="Sensitivity bars approximate how ±10% shifts in operational levers impact IRR, framing which variables deserve hedging or contractual protection.",
        actions=[
            "Prioritize hedges or long-term contracts on the variables with the largest downside bars.",
            "Replace placeholder deltas with live elasticities recalculated from Financials Yearly to keep guidance current."
        ]
    )

    # Chart 5: Revenue Breakdown Donut (Interactive)
    # Chart 5: Revenue Breakdown Donut (Interactive)

    # helper to fetch revenue series from scenario or sim
    def get_active_series(series_name: str):
        scenarios = st.session_state.get('scenarios', {})
        # Prefer a selected scenario if available
        keys = selected_scenarios if 'selected_scenarios' in locals() and selected_scenarios else list(scenarios.keys())
        if keys:
            data = scenarios.get(keys[0], {})
            if isinstance(data, dict) and series_name in data:
                return data[series_name]
        # Fallback to 'sim' if present
        sim = st.session_state.get('sim', {})
        if isinstance(sim, dict) and series_name in sim:
            return sim[series_name]
        return None

    # build pie_data from active revenue series
    rev_series = get_active_series('revenue')
    if rev_series is not None:
        total_rev = float(np.nansum(rev_series))
        pie_data = {
            'Power': total_rev * 0.30,
            'Byproducts': total_rev * 0.40,
            'Other': total_rev * 0.30,
        }

        fig5 = px.pie(
            values=list(pie_data.values()),
            names=list(pie_data.keys()),
            title="Revenue Breakdown",
            template="plotly_dark",
            hole=0.4,
            color_discrete_sequence=['#069494', '#047c7c', '#5dd1c1']
        )
        fig5.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig5, use_container_width=True)

        render_chart_context(
            title="Revenue Breakdown",
            source="Financials Yearly (Revenue detail) & scenario assumptions",
            explanation="The revenue mix clarifies dependence on power sales versus byproducts, indicating where commercial diversification is weakest.",
            actions=[
                "Replace static percentage splits with actual byproduct price curves from the Data Sources tab.",
                "Develop contingency plans for any stream exceeding 40% of total revenue to mitigate single-market risk."
            ]
        )
    else:
        st.info("Revenue series not found in the selected scenario. Save or select a scenario with revenue data to populate this chart.")


    # Chart 6: Mass Balance Sankey (Flow Diagram)

    def compute_yields_from_scenario(scn: dict):
        if not scn:
            return None

        msw = float(scn.get('msw_tpd', 0) or 0)
        tires = float(scn.get('tires_tpd', 0) or 0)
        rdf_yield = scn.get('rdf_yield')
        tdf_yield = scn.get('tdf_yield')
        syngas_yield = scn.get('syngas_yield')

        # Backfill yields using the mass-balance helper if they are missing
        if rdf_yield is None or tdf_yield is None or syngas_yield is None:
            calc_rdf, calc_tdf, calc_syngas, _ = simulate_mass_balance(msw, tires)
            if rdf_yield is None:
                rdf_yield = calc_rdf
            if tdf_yield is None:
                tdf_yield = calc_tdf
            if syngas_yield is None:
                syngas_yield = calc_syngas

        # Abort if everything is effectively zero (e.g., Base scenario placeholder)
        if max(msw, tires, float(rdf_yield or 0), float(tdf_yield or 0), float(syngas_yield or 0)) <= 0:
            return None

        return (
            msw,
            tires,
            float(rdf_yield or 0),
            float(tdf_yield or 0),
            float(syngas_yield or 0),
        )


    def get_active_yields():
        scenarios = st.session_state.get('scenarios', {})
        keys = selected_scenarios if 'selected_scenarios' in locals() and selected_scenarios else list(scenarios.keys())
        for key in keys:
            scn = scenarios.get(key, {})
            yields_data = compute_yields_from_scenario(scn)
            if yields_data is not None:
                return yields_data

        sim = st.session_state.get('sim')
        yields_data = compute_yields_from_scenario(sim)
        if yields_data is not None:
            return yields_data

        return None

    yields_pack = get_active_yields()
    if yields_pack is not None:
        msw_tpd, tires_tpd, rdf_yield, tdf_yield, syngas_yield = yields_pack
        fig6 = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15, thickness=20,
                line=dict(color="black", width=0.5),
                label=["MSW Input", "Tires Input", "RDF Yield", "TDF Yield", "Syngas", "Power", "Byproducts"]
            ),
            link=dict(
                source=[0, 1, 2, 3, 4, 4],
                target=[2, 3, 4, 4, 5, 6],
                value=[msw_tpd, tires_tpd, rdf_yield, tdf_yield, syngas_yield, syngas_yield * 0.7],
                color=["#069494", "#047c7c", "#5dd1c1", "#1e293b", "#069494", "#5dd1c1"]
            ),
            textfont=dict(color="white")
        )])
        fig6.update_layout(title="Mass Balance Flow (Interactive Nodes)", template="plotly_dark")
        st.plotly_chart(fig6, use_container_width=True)
        render_chart_context(
            title="Mass Balance Flow (Interactive Nodes)",
            source="Stage 1 Mass Balance",
            explanation="The Sankey trace links feedstock inputs to energy and byproducts, exposing conversion efficiency and losses across the process train.",
            actions=[
                "Benchmark conversion yields against latest pilot data in Stage 2/3 sheets and update coefficients accordingly.",
                "Instrument upstream operations to track MSW moisture spread, which materially shifts the syngas node."
            ]
        )
    else:
        st.info("No scenario yields available for the Sankey diagram yet. Save or select a scenario, or run a simulation, to populate the flow.")


    # Chart 7: Employee Salary Breakdown Pie
    salary_data = get_salary_summary().head(20)
    fig7 = px.pie(
        values=salary_data['Salary Yearly'].values,
        names=salary_data['Name'].values,
        title="Employee Salary Breakdown (Top 20)",
        template="plotly_dark",
        hole=0.3,
        color_discrete_sequence=['#069494', '#047c7c', '#5dd1c1', '#0f172a', '#1e293b']
    )
    st.plotly_chart(fig7, use_container_width=True)
    render_chart_context(
        title="Employee Salary Breakdown (Top 20)",
        source="Plant Salaries",
        explanation="Top-20 salary share highlights concentration of labor spend in leadership and specialized roles, informing workforce planning decisions.",
        actions=[
            "Tie salary growth assumptions to the 'Percent Increase' column in the sheet to reflect negotiated escalators.",
            "Model automation or outsourcing scenarios for the highest-cost roles to bend the labor cost curve."
        ]
    )

    # Chart 8: CapEx Distribution Bar
    capex_country_data = get_capex_summary()
    fig8 = px.bar(
        capex_country_data,
        x='Country',
        y='Value',
        title="CapEx by Country",
        template="plotly_dark",
        color='Country',
        color_discrete_sequence=['#069494', '#047c7c', '#5dd1c1', '#0f172a', '#1e293b']
    )
    st.plotly_chart(fig8, use_container_width=True)
    render_chart_context(
        title="CapEx by Country",
        source="CapEx",
        explanation="Country-level CapEx allocation surfaces geopolitical exposure and logistics dependencies for major equipment packages.",
        actions=[
            "Layer in lead-time and FX columns from the CapEx sheet to capture supply-chain risk.",
            "Engage secondary suppliers for any country exceeding 30% of spend to reduce single-region dependency."
        ]
    )

    # Chart 9: Financial Heatmap (Yearly Metrics)
    cf_for_heat = np.zeros(len(years))
    base_cf_array = np.array(base_cash_flow_series, dtype=float)
    if base_cf_array.size:
        length = min(base_cf_array.size, len(years) - 1)
        cf_for_heat[1:1 + length] = base_cf_array[:length]
    heat_data = pd.DataFrame({'Revenue': base_revenue, 'Expenses': base_expenses, 'CF': cf_for_heat}, index=years)
    fig10 = px.imshow(heat_data.T, text_auto=True, aspect="auto", color_continuous_scale=[
        '#022c22',  # deep teal
        '#035f5f',  # richer teal
        '#069494',  # core teal
        '#5dd1c1',  # pale teal
        '#c4f5ec'   # very light teal
    ], title="Financial Heatmap")
    fig10.update_layout(template="plotly_dark")
    st.plotly_chart(fig10, use_container_width=True)
    render_chart_context(
        title="Financial Heatmap",
        source="Financials Yearly",
        explanation="The heatmap compares revenue, expense, and cash flow intensity year-over-year, spotting inflection points that warrant deeper review.",
        actions=[
            "Annotate major step-changes with initiatives (e.g., plant expansion) documented in the Summary sheet.",
            "Set alerts when expenses exhibit >15% YoY growth to trigger cost-control workflows."
        ]
    )

    # Chart 11: Scatter IRR vs NPV Across Scenarios
    if len(selected_scenarios) > 1:
        irr_vals = [st.session_state['scenarios'][scen]['irr'] for scen in selected_scenarios]
        npv_vals = [st.session_state['scenarios'][scen]['npv'] / 1e6 for scen in selected_scenarios]
        fig11 = px.scatter(x=irr_vals, y=npv_vals, text=selected_scenarios, title="IRR vs NPV Scatter", template="plotly_dark", size=irr_vals)
        st.plotly_chart(fig11, use_container_width=True)
        render_chart_context(
            title="IRR vs NPV Scatter",
            source="Financials Yearly",
            explanation="Plotting IRR against NPV helps distinguish scenarios delivering quick paybacks versus those driving absolute value creation.",
            actions=[
                "Cluster scenarios by capital intensity in Summary to explain outliers on the chart.",
                "Adopt a decision matrix: approve only scenarios in the upper-right quadrant (high IRR, high NPV)."
            ]
        )

    # Chart 12: Cash Flow Box Plot
    cf_data = {scen: st.session_state['scenarios'][scen]['cf'] for scen in selected_scenarios}
    fig12 = px.box(pd.DataFrame(cf_data), title="Cash Flow Distribution by Scenario", template="plotly_dark", color_discrete_sequence=['#069494', '#047c7c', '#5dd1c1', '#1e293b'])
    st.plotly_chart(fig12, use_container_width=True)
    render_chart_context(
        title="Cash Flow Distribution by Scenario",
        source="Financials Yearly",
        explanation="Box plots summarize the volatility and downside risk of annual cash flows for each scenario, quickly highlighting unstable strategies.",
        actions=[
            "Overlay percentile metrics from Summary to quantify worst-case outcomes for governance reporting.",
            "Pair high-volatility scenarios with contingency liquidity plans or covenant buffers."
        ]
    )

def answer_question(question: str) -> str:
    if not OPENAI_CLIENT:
        return "LLM not configured. Add OPENAI_API_KEY to your environment or .streamlit/secrets.toml.\n\nSources: LLM"

    special = _direct_chat_response(question)
    if special:
        return special

    scenarios_store = st.session_state.get('scenarios', {})
    selected = st.session_state.get('selected_scenarios_current', list(scenarios_store.keys())[:1])
    if not selected:
        if 'Base' in scenarios_store:
            selected = ['Base']
        else:
            selected = list(scenarios_store.keys())[:1]
    metrics_cache = st.session_state.get('scenario_metrics_cache', {})

    scenario_contexts = []
    for scen in selected:
        data = scenarios_store.get(scen)
        if not data:
            continue
        metrics = metrics_cache.get(scen)
        if not metrics:
            metrics = compute_scenario_metrics(data, data.get('capex', base_capex))
        cash_on_cash = metrics.get('cash_on_cash')
        coc_display = 'N/A'
        if cash_on_cash is not None:
            try:
                if not np.isnan(cash_on_cash):
                    coc_display = f"{cash_on_cash:.2f}x"
            except TypeError:
                pass
        irr_value = metrics.get('irr')
        try:
            irr_display = f"{irr_value:.2f}%" if irr_value is not None and not np.isnan(irr_value) else 'N/A'
        except TypeError:
            irr_display = 'N/A'
        custom_lines = []
        for kpi in st.session_state.get('custom_kpis', []):
            try:
                kpi_value = evaluate_custom_kpi(kpi['formula'], metrics)
                custom_lines.append(f"{kpi['name']}={format_metric_value(kpi_value)}")
            except Exception:
                continue

        base_line = (
            f"IRR={irr_display} | NPV={format_metric_value(metrics.get('npv'))} | "
            f"Total Revenue={format_metric_value(metrics.get('total_revenue'))} | Total Expenses={format_metric_value(metrics.get('total_expenses'))} | "
            f"Avg CF={format_metric_value(metrics.get('avg_cf'))} | Cash-on-Cash={coc_display} | Payback Year={metrics.get('payback_year_label', 'N/A')}"
        )
        if custom_lines:
            base_line += " | " + " | ".join(custom_lines)
        scenario_contexts.append(f"[Scenario {scen}] {base_line}")

    history_entries = st.session_state.get('chat_history', [])[-6:]
    history_text = "\n".join(f"{role.title()}: {content}" for role, content in history_entries) if history_entries else ""

    index_data = st.session_state.get('rag_index_data')
    retrieved = search_documents(index_data, question, top_k=3)
    sources = [doc['source'] for doc in retrieved]
    unique_sources = list(dict.fromkeys(sources))

    context_segments = []
    if history_text:
        context_segments.append(f"[Conversation]\n{history_text}")
    if scenario_contexts:
        context_segments.extend(scenario_contexts)
        # Attribute scenario metrics back to the underlying sheet
        if 'Financials Yearly' not in unique_sources:
            unique_sources.insert(0, 'Financials Yearly')
    context_segments.append(f"[Controls]\n{CHAT_COMMAND_HELP}")
    mc_summary_text = st.session_state.get('monte_carlo_summary_text')
    if mc_summary_text:
        context_segments.append(f"[Monte Carlo] {mc_summary_text}")
        if 'Monte Carlo (Revenue Drivers)' not in unique_sources:
            unique_sources.append('Monte Carlo (Revenue Drivers)')
    chart_contexts = get_relevant_chart_contexts(question)
    for title, meta in chart_contexts:
        actions_text = "; ".join(meta.get('actions', []))
        chart_segment = [
            f"[Chart: {title}]",
            f"Source: {meta.get('source', 'App')}",
            f"Insight: {meta.get('explanation', '')}",
        ]
        if actions_text:
            chart_segment.append(f"Actions: {actions_text}")
        context_segments.append("\n".join(chart_segment))
        # Attribute charts to their underlying Excel tab where possible
        meta_source = meta.get('source', 'App')
        if meta_source and meta_source not in unique_sources:
            unique_sources.append(meta_source)
    knowledge_segments = get_app_knowledge_segments(question)
    for title, content in knowledge_segments:
        context_segments.append(f"[AppDocs: {title}]\n{content}")
        doc_label = f"App Docs – {title}"
        if doc_label not in unique_sources:
            unique_sources.append(doc_label)
    context_segments.extend([f"[{doc['source']}]\n{doc['text']}" for doc in retrieved])
    context_text = "\n\n".join(context_segments)
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS]

    lower_question = question.lower()
    if 'monte carlo' in lower_question:
        summary = st.session_state.get('monte_carlo_summary_text')
        if summary:
            extra = ""
            mc_store = st.session_state.get('monte_carlo')
            if mc_store:
                irr_stats = mc_store['summary'].get('IRR (%)', {})
                npv_stats = mc_store['summary'].get('NPV ($)', {})
                irr_p10 = irr_stats.get('p10')
                irr_p90 = irr_stats.get('p90')
                npv_p10 = npv_stats.get('p10')
                npv_p90 = npv_stats.get('p90')
                if irr_p10 is not None and not np.isnan(irr_p10) and irr_p90 is not None and not np.isnan(irr_p90):
                    extra += f"IRR runs from P10 {irr_p10:.2f}% to P90 {irr_p90:.2f}%. "
                if npv_p10 is not None and not np.isnan(npv_p10) and npv_p90 is not None and not np.isnan(npv_p90):
                    extra += f"NPV spans {format_metric_value(npv_p10)} to {format_metric_value(npv_p90)}."
            return f"{summary}\n{extra}\n\nSources: Monte Carlo (Revenue Drivers)"

    llm_response = None
    if context_text:
        llm_response = call_llm_with_context(context_text, question)

    if llm_response:
        if is_unknown_response(llm_response):
            return llm_response
        source_label = ", ".join(unique_sources) if unique_sources else "LLM"
        return f"{llm_response}\n\nSources: {source_label}"

    llm_fallback = call_llm_with_context("", question)
    if llm_fallback:
        if is_unknown_response(llm_fallback):
            return llm_fallback
        return f"{llm_fallback}\n\nSources: LLM"

    return "I couldn’t find a relevant answer right now. Please try rephrasing or provide more detail."


if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []


def apply_chat_command(query: str):
    updates = []
    command_specs = [
        (r"(?:set|update|change)\s+msw\s*(?:tpd)?\s*(?:to)?\s*([0-9]+)", 'msw_tpd', int, 400, 800, "MSW TPD"),
        (r"(?:set|update|change)\s+tires?\s*(?:tpd)?\s*(?:to)?\s*([0-9]+)", 'tires_tpd', int, 50, 200, "Tires TPD"),
        (r"(?:set|update|change)\s+power\s+price\s*(?:to)?\s*([0-9]+(?:\.[0-9]+)?)", 'power_price', float, 0.05, 0.15, "Power Price"),
        (r"(?:set|update|change)\s+inflation\s*(?:rate)?\s*(?:to)?\s*([0-9]+(?:\.[0-9]+)?)", 'inflation', float, 0.01, 0.05, "Inflation"),
        (r"(?:set|update|change)\s+capex\s*(?:multiplier)?\s*(?:to)?\s*([0-9]+(?:\.[0-9]+)?)", 'capex_mult', float, 0.8, 1.2, "CapEx Multiplier"),
    ]

    lowered = query.lower()
    for pattern, key, caster, min_val, max_val, label in command_specs:
        match = re.search(pattern, lowered)
        if match:
            try:
                value = caster(match.group(1))
            except ValueError:
                continue
            value = max(min_val, min(max_val, value))
            if key in {'power_price', 'inflation'}:
                value = round(value, 3)
            if key == 'capex_mult':
                value = round(value, 2)
            st.session_state[key] = value
            updates.append(f"{label}: {value}")

    return updates


def _try_optimize_from_query(query: str):
    text = query.lower().strip()
    # Examples captured: "optimize to hit 20% irr", "optimize irr to 18%", "find scenario to reach 22% irr"
    m = re.search(r"(optimi[sz]e|find|search).*?(irr).*?(?:to|at|hit|reach|target)\\s*([0-9]+(?:\\.[0-9]+)?)\\s*%", text)
    if not m:
        m = re.search(r"(irr)\\s*(?:to|at|hit|reach|target)\\s*([0-9]+(?:\\.[0-9]+)?)\\s*%.*?(optimi[sz]e|find|search)?", text)
    if not m:
        return None
    try:
        target = float(m.group(3) if m.lastindex and m.lastindex >= 3 else m.group(2))
    except Exception:
        return None
    samples = 800
    tolerance = 0.25  # percentage points
    result = optimize_scenario(target, samples, tolerance, seed=None)
    st.session_state['optimizer_result'] = result
    st.session_state['optimizer_timestamp'] = pd.Timestamp.utcnow().isoformat()
    best_scn = result.get('scenario', {})
    name = f"Optimized {target:.2f}% IRR"
    st.session_state.setdefault('scenarios', {})[name] = serialize_scenario_state(best_scn)
    st.session_state['selected_scenarios'] = [name] + [s for s in st.session_state.get('selected_scenarios', []) if s != name]
    summary = result.get('summary') or generate_optimizer_explanation(result)
    return f"{summary}\n\nSaved scenario: {name}"


def handle_chat_query(query: str):
    st.session_state['chat_history'].append(("user", query))
    adjustments = apply_chat_command(query)
    optimization = _try_optimize_from_query(query)
    response = optimization if optimization else answer_question(query)
    if adjustments:
        adjustment_text = "Adjustments applied: " + ", ".join(adjustments)
        if "\n\nSources:" in response:
            body, sources_section = response.split("\n\nSources:", 1)
            response = f"{body}\n\n{adjustment_text}\n\nSources:{sources_section}"
        else:
            response = f"{response}\n\n{adjustment_text}"
    st.session_state['chat_history'].append(("assistant", response))


def render_chat_interface(tab_key: str):
    st.subheader("Chat with the Model")
    for role, content in st.session_state['chat_history']:
        with st.chat_message(role):
            st.markdown(content)
    prompt = st.chat_input("Ask about the Excel model", key=f"chat_input_{tab_key}")
    if prompt:
        handle_chat_query(prompt)
        trigger_rerun()


# Export Enhanced Report
def generate_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=PAGE_MARGIN,
        rightMargin=PAGE_MARGIN,
        topMargin=PAGE_MARGIN,
        bottomMargin=PAGE_MARGIN,
    )
    usable_width = doc.width

    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading2']
    subheading_style = styles['Heading3']
    body_style = styles['BodyText']
    small_style = ParagraphStyle('Small', parent=body_style, fontSize=9, leading=12)
    caption_style = ParagraphStyle('Caption', parent=body_style, fontSize=8, textColor=colors.grey)

    def para(text, style=body_style):
        return Paragraph(text, style)

    def compute_export_highlights():
        base_revenue_series = summary_series_aligned.get('revenue', np.array([]))
        if base_revenue_series.size == 0:
            return None

        sim_snapshot = st.session_state.get('sim')
        if not sim_snapshot:
            sim_snapshot = st.session_state.get('scenarios', {}).get('Base')
        if not sim_snapshot:
            sim_snapshot = base_scenario_raw

        try:
            year_axis = [int(str(y)) for y in years]
        except Exception:
            year_axis = list(range(1, len(base_revenue_series) + 1))
        target_len = min(len(year_axis), base_revenue_series.size)
        year_axis = year_axis[:target_len]
        base_revenue_summary = base_revenue_series[:target_len]
        base_ebitda_summary = summary_series_aligned.get('ebitda', np.array([]))[:target_len]
        base_net_income_summary = summary_series_aligned.get('net_income', np.array([]))[:target_len]
        base_project_cf_summary = summary_series_aligned.get('project_cash_flow', np.array([]))[:target_len]

        active_revenue = _match_length(sim_snapshot.get('revenue', base_revenue_summary), target_len)
        active_expenses = _match_length(sim_snapshot.get('expenses', base_expenses), target_len)
        ebitda_margin = _compute_margin(
            base_ebitda_summary,
            np.where(base_revenue_summary == 0, np.nan, base_revenue_summary)
        )
        net_margin = _compute_margin(
            base_net_income_summary,
            np.where(base_revenue_summary == 0, np.nan, base_revenue_summary)
        )
        active_ebitda = active_revenue * ebitda_margin
        active_net_income = active_revenue * net_margin
        active_cf = active_revenue - active_expenses
        active_operating_expense = active_revenue - active_ebitda
        active_other_costs = active_ebitda - active_net_income

        cash_flows_full = np.array(sim_snapshot.get('cash_flows_full', []), dtype=float)
        if cash_flows_full.size >= target_len + 1:
            active_project_cf = cash_flows_full[1:target_len + 1]
        else:
            ratio = np.divide(
                active_revenue,
                base_revenue_summary,
                out=np.ones_like(active_revenue),
                where=~np.isclose(base_revenue_summary, 0)
            )
            active_project_cf = base_project_cf_summary * ratio

        base_totals = {
            'revenue': np.nansum(base_revenue_summary),
            'ebitda': np.nansum(base_ebitda_summary),
            'net_income': np.nansum(base_net_income_summary),
            'project_cf': np.nansum(base_project_cf_summary),
        }
        active_totals = {
            'revenue': np.nansum(active_revenue),
            'ebitda': np.nansum(active_ebitda),
            'net_income': np.nansum(active_net_income),
            'project_cf': np.nansum(active_project_cf),
        }

        project_irr_current = float(sim_snapshot.get('irr', float('nan')))
        equity_irr_current = sim_snapshot.get('equity_irr')
        npv_current = float(sim_snapshot.get('npv', float('nan')))
        project_irr_base = summary_scalar_highlights.get('total_project_irr_pct', float('nan'))
        equity_irr_base = summary_scalar_highlights.get('total_equity_irr_pct', float('nan'))
        npv_base = summary_scalar_highlights.get('net_present_value', float('nan'))
        cash_base = summary_scalar_highlights.get('total_project_cash_flow', float('nan'))
        base_capex_value = float(sim_snapshot.get('capex', base_capex))

        revenue_shift_pct = float(st.session_state.get('whatif_rev_shift', 0.0))
        expense_shift_pct = float(st.session_state.get('whatif_expense_shift', 0.0))
        capex_shift_pct = float(st.session_state.get('whatif_capex_shift', 0.0))
        discount_shift_bps = float(st.session_state.get('whatif_discount_shift_bps', 0))

        revenue_multiplier = 1 + revenue_shift_pct / 100.0
        expense_multiplier = 1 + expense_shift_pct / 100.0
        capex_multiplier_tuning = 1 + capex_shift_pct / 100.0
        tuned_discount_rate = max(base_discount + discount_shift_bps / 10000.0, 0.0001)
        tuning_applied = any(
            abs(val) > 1e-6
            for val in (revenue_shift_pct, expense_shift_pct, capex_shift_pct, discount_shift_bps)
        )

        tuned_revenue = active_revenue * revenue_multiplier
        tuned_expenses = np.nan_to_num(active_expenses, nan=0.0) * revenue_multiplier * expense_multiplier
        tuned_operating_expense = np.nan_to_num(active_operating_expense, nan=0.0) * revenue_multiplier * expense_multiplier
        tuned_ebitda = tuned_revenue - tuned_operating_expense
        other_costs = np.nan_to_num(active_other_costs, nan=0.0)
        tuned_net_income = tuned_ebitda - other_costs

        revenue_ratio = np.ones_like(tuned_revenue)
        revenue_mask = (~np.isclose(active_revenue, 0)) & np.isfinite(active_revenue)
        revenue_ratio[revenue_mask] = tuned_revenue[revenue_mask] / active_revenue[revenue_mask]
        revenue_ratio = np.nan_to_num(revenue_ratio, nan=1.0, posinf=1.0, neginf=1.0)

        ebitda_ratio = np.ones_like(tuned_ebitda)
        ebitda_mask = (~np.isclose(active_ebitda, 0)) & np.isfinite(active_ebitda)
        ebitda_ratio[ebitda_mask] = tuned_ebitda[ebitda_mask] / active_ebitda[ebitda_mask]
        ebitda_ratio = np.nan_to_num(ebitda_ratio, nan=1.0, posinf=1.0, neginf=1.0)

        tuned_project_cf = np.nan_to_num(active_project_cf, nan=0.0)
        if tuned_project_cf.size:
            tuned_project_cf = tuned_project_cf * ebitda_ratio[:tuned_project_cf.size]

        tuned_capex_value = base_capex_value * capex_multiplier_tuning
        if cash_flows_full.size > 0:
            base_operating_cf_full = cash_flows_full[1:]
            if base_operating_cf_full.size:
                ratio_full = np.ones_like(base_operating_cf_full)
                overlap = min(base_operating_cf_full.size, ebitda_ratio.size)
                ratio_full[:overlap] = ebitda_ratio[:overlap]
                if overlap < base_operating_cf_full.size:
                    fill_value = ebitda_ratio[overlap - 1] if overlap > 0 else 1.0
                    ratio_full[overlap:] = fill_value
                tuned_operating_cf_full = base_operating_cf_full * ratio_full
            else:
                tuned_operating_cf_full = np.array([], dtype=float)
            tuned_cash_flows_full = np.hstack(
                (np.array([tuned_capex_value], dtype=float), tuned_operating_cf_full)
            )
        else:
            tuned_cash_flows_full = np.hstack(
                (np.array([tuned_capex_value], dtype=float), np.asarray(tuned_project_cf, dtype=float))
            )

        tuned_cash_flows_full = np.where(np.isnan(tuned_cash_flows_full), 0.0, tuned_cash_flows_full)

        project_irr_tuned = (
            npf.irr(tuned_cash_flows_full) * 100
            if tuned_cash_flows_full.size > 1 and np.any(tuned_cash_flows_full[1:] > 0)
            else float('nan')
        )
        if tuned_cash_flows_full.size > 1:
            npv_tuned = npf.npv(tuned_discount_rate, tuned_cash_flows_full[1:]) + tuned_cash_flows_full[0]
        else:
            npv_tuned = tuned_cash_flows_full[0]

        tuned_totals = {
            'revenue': np.nansum(tuned_revenue),
            'ebitda': np.nansum(tuned_ebitda),
            'net_income': np.nansum(tuned_net_income),
            'project_cf': np.nansum(tuned_project_cf),
        }

        base_equity_total = summary_scalar_highlights.get('total_equity_contributed')
        base_ltv_ratio = summary_scalar_highlights.get('loan_to_value')

        base_total_investment = float('nan')
        if np.isfinite(base_equity_total) and np.isfinite(base_ltv_ratio) and (1 - base_ltv_ratio) > 1e-6:
            base_total_investment = base_equity_total / max(1 - base_ltv_ratio, 1e-6)
        elif np.isfinite(base_capex):
            base_total_investment = abs(base_capex)
        elif np.isfinite(base_capex_value):
            base_total_investment = abs(base_capex_value)
        elif np.isfinite(base_equity_total):
            base_total_investment = base_equity_total

        base_debt_total = float('nan')
        if np.isfinite(base_total_investment) and np.isfinite(base_ltv_ratio):
            base_debt_total = base_total_investment * base_ltv_ratio
        elif np.isfinite(base_total_investment) and np.isfinite(base_equity_total):
            base_debt_total = max(base_total_investment - base_equity_total, 0.0)

        tuned_total_investment = abs(tuned_capex_value) if np.isfinite(tuned_capex_value) else float('nan')
        tuned_debt_total = float('nan')
        if np.isfinite(tuned_total_investment) and tuned_total_investment > 0 and np.isfinite(base_debt_total):
            tuned_debt_total = base_debt_total
            if (
                np.isfinite(base_total_investment)
                and base_total_investment > 0
                and tuned_total_investment < base_total_investment - 1e-6
            ):
                scale = tuned_total_investment / base_total_investment
                tuned_debt_total = base_debt_total * scale
            tuned_debt_total = min(tuned_debt_total, tuned_total_investment)
            tuned_equity_contrib = max(tuned_total_investment - tuned_debt_total, 0.0)
            tuned_ltv_ratio = tuned_debt_total / tuned_total_investment if tuned_total_investment > 0 else float('nan')
        else:
            tuned_equity_contrib = base_equity_total if np.isfinite(base_equity_total) else float('nan')
            tuned_ltv_ratio = base_ltv_ratio if np.isfinite(base_ltv_ratio) else float('nan')

        if not np.isfinite(tuned_equity_contrib):
            tuned_equity_contrib = base_equity_total if np.isfinite(base_equity_total) else float('nan')
        if np.isfinite(tuned_ltv_ratio):
            tuned_ltv_ratio = float(np.clip(tuned_ltv_ratio, 0.0, 1.5))
        else:
            tuned_ltv_ratio = base_ltv_ratio if np.isfinite(base_ltv_ratio) else float('nan')

        irr_display = project_irr_tuned if np.isfinite(project_irr_tuned) else project_irr_current
        npv_display = npv_tuned if np.isfinite(npv_tuned) else npv_current
        totals_display = tuned_totals

        if (
            np.isfinite(irr_display)
            and np.isfinite(project_irr_current)
            and equity_irr_current is not None
            and not np.isnan(equity_irr_current)
        ):
            equity_irr_display = float(equity_irr_current) + (irr_display - project_irr_current)
        elif equity_irr_current is not None and not np.isnan(equity_irr_current):
            equity_irr_display = float(equity_irr_current)
        else:
            equity_irr_display = float(equity_irr_base) if np.isfinite(equity_irr_base) else float('nan')

        irr_delta = irr_display - project_irr_base if np.isfinite(irr_display) and np.isfinite(project_irr_base) else None
        npv_delta = npv_display - npv_base if np.isfinite(npv_display) and np.isfinite(npv_base) else None
        cash_delta = totals_display['project_cf'] - cash_base if np.isfinite(totals_display['project_cf']) and np.isfinite(cash_base) else None
        equity_irr_delta = equity_irr_display - equity_irr_base if np.isfinite(equity_irr_display) and np.isfinite(equity_irr_base) else None
        equity_contrib_delta = tuned_equity_contrib - base_equity_total if np.isfinite(tuned_equity_contrib) and np.isfinite(base_equity_total) else None
        ltv_delta_pts = (tuned_ltv_ratio - base_ltv_ratio) * 100 if np.isfinite(tuned_ltv_ratio) and np.isfinite(base_ltv_ratio) else None

        revenue_delta = totals_display['revenue'] - base_totals['revenue'] if np.isfinite(totals_display['revenue']) and np.isfinite(base_totals['revenue']) else float('nan')
        ebitda_margin_base = _compute_margin(base_ebitda_summary, np.where(base_revenue_summary == 0, np.nan, base_revenue_summary))
        ebitda_margin_current = _compute_margin(tuned_ebitda, np.where(tuned_revenue == 0, np.nan, tuned_revenue))
        avg_ebitda_margin_base = np.nanmean(ebitda_margin_base) * 100 if ebitda_margin_base.size else float('nan')
        avg_ebitda_margin_current = np.nanmean(ebitda_margin_current) * 100 if ebitda_margin_current.size else float('nan')

        def _break_even_year(series):
            for year, value in zip(year_axis, series):
                if value > 0:
                    return year
            return None

        def _payback_year(cash_series):
            cash_series = np.asarray(cash_series, dtype=float)
            if cash_series.size <= 1:
                return None
            cumulative = np.cumsum(cash_series)
            for idx in range(1, cumulative.size):
                if cumulative[idx] >= 0:
                    year_idx = idx - 1
                    if year_idx < len(year_axis):
                        return year_axis[year_idx]
                    return year_axis[-1] + (year_idx - (len(year_axis) - 1))
            return None

        base_break_even = _break_even_year(base_net_income_summary)
        current_break_even = _break_even_year(tuned_net_income)
        base_payback_year = _payback_year(cash_flows_full)
        tuned_payback_year = _payback_year(tuned_cash_flows_full)

        detail_lines = []
        adjustments = []
        if abs(revenue_shift_pct) >= 0.1:
            adjustments.append(f"Revenue {revenue_shift_pct:+.0f}%")
        if abs(expense_shift_pct) >= 0.1:
            adjustments.append(f"Operating expenses {expense_shift_pct:+.0f}%")
        if abs(capex_shift_pct) >= 0.1:
            adjustments.append(f"CapEx {capex_shift_pct:+.0f}%")
        if abs(discount_shift_bps) >= 1:
            adjustments.append(f"Discount rate {discount_shift_bps:+.0f} bps")

        if irr_delta is not None:
            detail_lines.append(
                f"Project IRR is <b>{irr_display:.2f}%</b>, a {format_percent_delta(irr_delta)} shift versus the Excel summary."
            )
        elif np.isfinite(irr_display):
            detail_lines.append(f"Project IRR remains at <b>{irr_display:.2f}%</b> under the tuned assumptions.")

        if npv_delta is not None:
            detail_lines.append(
                f"NPV recalculates to <b>{format_metric_value(npv_display)}</b> ({format_currency_delta(npv_delta)} vs summary)."
            )

        if equity_irr_delta is not None:
            if abs(equity_irr_delta) >= 0.05:
                detail_lines.append(
                    f"Equity IRR moves to <b>{equity_irr_display:.2f}%</b>, a {equity_irr_delta:+.2f} pt change from the base case."
                )
            else:
                detail_lines.append(
                    f"Equity IRR holds near <b>{equity_irr_display:.2f}%</b>, effectively flat versus the summary."
                )
        elif np.isfinite(equity_irr_display):
            detail_lines.append(
                f"Equity IRR remains about <b>{equity_irr_display:.2f}%</b>; update the equity waterfall for a refined view."
            )

        if np.isfinite(revenue_delta) and abs(revenue_delta) > 1:
            direction = "higher" if revenue_delta >= 0 else "lower"
            detail_lines.append(
                f"Aggregate revenue is {direction} by <b>{format_metric_value(abs(revenue_delta))}</b> relative to the summary tab."
            )

        if np.isfinite(avg_ebitda_margin_base) and np.isfinite(avg_ebitda_margin_current):
            detail_lines.append(
                f"Average EBITDA margin shifts from <b>{avg_ebitda_margin_base:.1f}%</b> to <b>{avg_ebitda_margin_current:.1f}%</b> under the current inputs."
            )

        if current_break_even != base_break_even:
            if current_break_even and base_break_even:
                detail_lines.append(
                    f"Net income turns positive in <b>{current_break_even}</b>, compared with {base_break_even} in the baseline summary."
                )
            elif current_break_even and not base_break_even:
                detail_lines.append(
                    f"Net income reaches break-even in <b>{current_break_even}</b>, while the summary never crossed zero."
                )
            elif base_break_even and not current_break_even:
                detail_lines.append(
                    f"Net income stays negative across the horizon (baseline break-even: {base_break_even})."
                )

        if cash_delta is not None and np.isfinite(cash_delta) and abs(cash_delta) > 1:
            direction_cf = "adds" if cash_delta >= 0 else "reduces"
            detail_lines.append(
                f"Cumulative project cash flow {direction_cf} about <b>{format_metric_value(abs(cash_delta))}</b> relative to the summary."
            )

        if equity_contrib_delta is not None and abs(equity_contrib_delta) > 1:
            detail_lines.append(
                f"Total equity contributed adjusts by {format_currency_delta(equity_contrib_delta)}, now {format_metric_value(tuned_equity_contrib)}."
            )

        if ltv_delta_pts is not None and abs(ltv_delta_pts) >= 0.1:
            detail_lines.append(
                f"Loan-to-Value shifts to <b>{_fmt_percent(tuned_ltv_ratio * 100) or 'N/A'}</b> ({format_percent_delta(ltv_delta_pts)})."
            )
        elif np.isfinite(tuned_ltv_ratio):
            detail_lines.append(
                f"Loan-to-Value remains around <b>{_fmt_percent(tuned_ltv_ratio * 100) or 'N/A'}</b> with the current tuning."
            )

        if tuned_project_cf.size and np.any(np.isfinite(tuned_project_cf)):
            peak_idx = int(np.nanargmax(tuned_project_cf))
            trough_idx = int(np.nanargmin(tuned_project_cf))
            detail_lines.append(
                f"Peak free cash flow of {format_metric_value(tuned_project_cf[peak_idx])} appears in {year_axis[peak_idx]}, while the leanest year is {year_axis[trough_idx]} at {format_metric_value(tuned_project_cf[trough_idx])}."
            )

        if discount_shift_bps:
            detail_lines.append(
                f"Discount rate is now <b>{tuned_discount_rate * 100:.2f}%</b> ({discount_shift_bps:+.0f} bps vs baseline)."
            )

        payback_note = ""
        if tuned_payback_year or base_payback_year:
            if tuned_payback_year and base_payback_year and tuned_payback_year != base_payback_year:
                payback_note = f"Payback shifts to {tuned_payback_year} vs {base_payback_year} originally."
            elif tuned_payback_year and not base_payback_year:
                payback_note = f"Payback emerges around {tuned_payback_year}, whereas the base case never reached breakeven."
            elif base_payback_year and not tuned_payback_year:
                payback_note = f"Payback no longer occurs within the modeled horizon (baseline: {base_payback_year})."
            if payback_note:
                detail_lines.append(payback_note)

        if not detail_lines:
            detail_lines.append("Financial trajectory remains in line with the Excel summary; adjust the what-if controls to explore sensitivities.")

        irr_delta_value = irr_display - project_irr_base if np.isfinite(irr_display) and np.isfinite(project_irr_base) else float('nan')
        capex_base_mult = float(sim_snapshot.get('capex_mult', st.session_state.get('capex_mult', capex_mult)))
        effective_capex_mult = capex_base_mult * capex_multiplier_tuning

        recommendations = []
        if np.isfinite(irr_display):
            if not np.isnan(irr_delta_value) and abs(irr_delta_value) >= 0.05:
                if irr_delta_value > 0:
                    recommendations.append(
                        f"IRR is up {irr_delta_value:+.2f} pts to {irr_display:.2f}%. Stress-test feedstock and pricing before locking this case."
                    )
                else:
                    recommendations.append(
                        f"IRR slips {irr_delta_value:+.2f} pts to {irr_display:.2f}%. Re-cut CapEx or uprate revenues to get back on target."
                    )
            else:
                recommendations.append(
                    f"IRR holds around {irr_display:.2f}%. Use tuning scenarios to probe upside/downside before saving."
                )

        if equity_irr_delta is not None and abs(equity_irr_delta) >= 0.05:
            direction = "up" if equity_irr_delta > 0 else "down"
            recommendations.append(
                f"Equity IRR ticks {direction} to {equity_irr_display:.2f}% ({equity_irr_delta:+.2f} pts); reconcile the equity waterfall with sponsors."
            )
        elif np.isfinite(equity_irr_display) and equity_irr_delta is not None:
            recommendations.append(
                f"Equity IRR stays near {equity_irr_display:.2f}%; keep monitoring sponsor returns as inputs move."
            )

        if equity_contrib_delta is not None and abs(equity_contrib_delta) > 1:
            recommendations.append(
                f"Equity checks adjust by {format_currency_delta(equity_contrib_delta)}; confirm capital calls with investors."
            )

        if ltv_delta_pts is not None and abs(ltv_delta_pts) >= 0.1:
            direction_ltv = "higher" if ltv_delta_pts > 0 else "lower"
            recommendations.append(
                f"Loan-to-Value trends {direction_ltv} to {(_fmt_percent(tuned_ltv_ratio * 100) or 'N/A')} ({format_percent_delta(ltv_delta_pts)}). Coordinate with lenders if covenants tighten."
            )

        if np.isfinite(base_capex):
            if effective_capex_mult < 1:
                capex_savings = abs(base_capex) * abs(1 - effective_capex_mult)
                recommendations.append(
                    f"Effective CapEx is {effective_capex_mult:.2f}× baseline (savings {format_metric_value(capex_savings)}). Validate procurement can deliver."
                )
            elif effective_capex_mult > 1:
                capex_overrun = abs(base_capex) * abs(effective_capex_mult - 1)
                recommendations.append(
                    f"CapEx overruns by {format_metric_value(capex_overrun)} (multiplier {effective_capex_mult:.2f}×). Revisit scope or financing buffers."
                )

        if abs(revenue_shift_pct) >= 5:
            direction = "upside" if revenue_shift_pct > 0 else "downside"
            recommendations.append(
                f"Validate the {direction} revenue shift of {revenue_shift_pct:+.0f}% with refreshed offtake assumptions."
            )

        if abs(discount_shift_bps) >= 25:
            recommendations.append(
                f"Discount rate now {tuned_discount_rate * 100:.2f}% ({discount_shift_bps:+.0f} bps); align with finance on WACC assumptions."
            )

        if not recommendations:
            recommendations.append("No immediate follow-ups detected. Capture this case or tweak assumptions for additional sensitivity sweeps.")

        msw_text = _fmt_number(sim_snapshot.get('msw_tpd', msw_tpd))
        rdf_text = _fmt_number(sim_snapshot.get('rdf_yield'), "{:.2f}")
        syngas_text = _fmt_number(sim_snapshot.get('syngas_yield'), "{:.2f}")
        scale_text = _fmt_number(sim_snapshot.get('scale'), "{:.2f}")
        power_text = _fmt_number(sim_snapshot.get('power_mwe'), "{:.1f}")
        overview_parts = []
        if msw_text:
            overview_parts.append(f"Current sliders push about {msw_text} tons/day of MSW through Stage 1.")
        if rdf_text and syngas_text:
            overview_parts.append(f"That yields roughly {rdf_text} tons/day of RDF and {syngas_text} tons/day of syngas.")
        if scale_text:
            detail_power = f" (~{power_text} MW)" if power_text else ""
            overview_parts.append(f"Net power output scales to {scale_text}× the design basis{detail_power}.")

        tuning_summary = ""
        if adjustments:
            tuning_summary = "What-if tuning applied: " + ", ".join(adjustments) + "."
        elif tuning_applied:
            tuning_summary = "What-if tuning active with minor adjustments applied."
        else:
            tuning_summary = "No what-if adjustments active; values mirror the Excel summary."

        metrics_rows = [
            (
                "Project IRR",
                f"{irr_display:.2f}%" if np.isfinite(irr_display) else "N/A",
                format_percent_delta(irr_delta) if irr_delta is not None else "—",
            ),
            (
                "Equity IRR",
                f"{equity_irr_display:.2f}%" if np.isfinite(equity_irr_display) else "N/A",
                format_percent_delta(equity_irr_delta) if equity_irr_delta is not None else "—",
            ),
            (
                "NPV",
                format_metric_value(npv_display) if np.isfinite(npv_display) else "N/A",
                format_currency_delta(npv_delta) if npv_delta is not None else "—",
            ),
            (
                "Total Project Cash Flow",
                format_metric_value(totals_display['project_cf']),
                format_currency_delta(cash_delta) if cash_delta is not None else "—",
            ),
            (
                "Total Equity Contributed",
                format_metric_value(tuned_equity_contrib),
                format_currency_delta(equity_contrib_delta) if equity_contrib_delta is not None else "—",
            ),
            (
                "Loan-to-Value",
                _fmt_percent(tuned_ltv_ratio * 100) if np.isfinite(tuned_ltv_ratio) else (
                    _fmt_percent(base_ltv_ratio * 100) if np.isfinite(base_ltv_ratio) else "N/A"
                ),
                format_percent_delta(ltv_delta_pts) if ltv_delta_pts is not None else "—",
            ),
            (
                "Aggregate Revenue",
                format_metric_value(totals_display['revenue']),
                format_currency_delta(revenue_delta) if np.isfinite(revenue_delta) and abs(revenue_delta) > 1 else "—",
            ),
        ]

        return {
            'tuning_applied': tuning_applied,
            'tuning_summary': tuning_summary,
            'overview_text': " ".join(overview_parts) if overview_parts else "",
            'metrics_rows': metrics_rows,
            'detail_lines': detail_lines,
            'recommendations': recommendations,
        }
    story = [
        Paragraph("CREC Financial Digital Twin Report", title_style),
        Spacer(1, 6),
        Paragraph(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", small_style),
        Spacer(1, 12),
    ]

    scenario_keys = selected_scenarios if selected_scenarios else list(st.session_state['scenarios'].keys())
    unique_keys = []
    for key in scenario_keys:
        if key not in unique_keys:
            unique_keys.append(key)

    base_key = 'Base' if 'Base' in st.session_state['scenarios'] else (unique_keys[0] if unique_keys else None)
    base_data = None
    base_metrics = None
    base_inputs_line = None
    if base_key:
        base_data = st.session_state['scenarios'][base_key]
        base_metrics = compute_scenario_metrics(base_data, base_data.get('capex', base_capex))
        base_inputs_line = (
            f"MSW {base_data.get('msw_tpd', msw_tpd):.0f} TPD | Tires {base_data.get('tires_tpd', tires_tpd):.0f} TPD | "
            f"Power ${base_data.get('power_price', power_price):.3f}/kWh | Inflation {format_rate(base_data.get('inflation', inflation))} | "
            f"CapEx Multiplier {base_data.get('capex_mult', capex_mult):.2f}"
        )

    if base_metrics:
        story.append(Paragraph("Base Scenario Overview", heading_style))
        inputs_text = f"<b>Scenario:</b> {base_key} &nbsp;&nbsp; <b>Inputs:</b> {base_inputs_line}" if base_inputs_line else f"<b>Scenario:</b> {base_key}"
        story.append(para(inputs_text, small_style))
        base_table_data = [
            [para('<b>Metric</b>', small_style), para('<b>Value</b>', small_style)],
            [para('IRR'), para(safe_percent(base_metrics.get('irr')))],
            [para('NPV'), para(format_metric_value(base_metrics.get('npv')))],
            [para('Payback Year'), para(base_metrics.get('payback_year_label', 'N/A'))],
            [para('Total Revenue'), para(format_metric_value(base_metrics.get('total_revenue')))],
            [para('Total Expenses'), para(format_metric_value(base_metrics.get('total_expenses')))],
            [para('Average Cash Flow'), para(format_metric_value(base_metrics.get('avg_cf')))],
            [para('Cash-on-Cash'), para(format_metric_value(base_metrics.get('cash_on_cash')))],
        ]
        base_table = Table(base_table_data, hAlign='LEFT', colWidths=[0.4 * usable_width, 0.5 * usable_width])
        base_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f2933')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        story.append(base_table)
        story.append(para("Use these baseline values as the reference point for scenario deltas.", caption_style))
        story.append(Spacer(1, 12))

    base_equity_irr_value = base_data.get('equity_irr') if base_key and base_metrics else None

    highlights_snapshot = compute_export_highlights()
    if highlights_snapshot:
        story.append(Paragraph("Base Model Highlights", heading_style))
        story.append(para(highlights_snapshot['tuning_summary'], small_style))
        if highlights_snapshot['overview_text']:
            story.append(para(highlights_snapshot['overview_text'], body_style))
        story.append(Spacer(1, 6))
        table_data = [
            [para('<b>Metric</b>', small_style), para('<b>Value</b>', small_style), para('<b>Δ vs Summary</b>', small_style)]
        ]
        for metric, value, delta in highlights_snapshot['metrics_rows']:
            table_data.append([
                para(metric, small_style),
                para(value, small_style),
                para(delta, small_style),
            ])
        highlights_table = Table(
            table_data,
            hAlign='LEFT',
            colWidths=[0.28 * usable_width, 0.36 * usable_width, 0.36 * usable_width]
        )
        highlights_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1f2933')),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        story.append(highlights_table)
        story.append(Spacer(1, 8))

        if highlights_snapshot['detail_lines']:
            story.append(Paragraph("Detailed Insights", subheading_style))
            detail_items = [
                ListItem(Paragraph(line, body_style), leftIndent=12)
                for line in highlights_snapshot['detail_lines']
            ]
            story.append(ListFlowable(detail_items, bulletType='bullet', start='•', leftPadding=12))
            story.append(Spacer(1, 6))

        if highlights_snapshot['recommendations']:
            story.append(Paragraph("Recommended Next Moves", subheading_style))
            rec_items = [
                ListItem(Paragraph(line, body_style), leftIndent=12)
                for line in highlights_snapshot['recommendations']
            ]
            story.append(ListFlowable(rec_items, bulletType='bullet', start='•', leftPadding=12))
            story.append(Spacer(1, 12))

    if unique_keys:
        story.append(Paragraph("Scenario Comparison Snapshot", heading_style))
        comp_data = [[
            para('<b>Scenario</b>', small_style),
            para('<b>IRR</b>', small_style),
            para('<b>Equity IRR</b>', small_style),
            para('<b>ΔIRR vs Base</b>', small_style),
            para('<b>NPV</b>', small_style),
            para('<b>ΔNPV vs Base</b>', small_style),
            para('<b>Avg Cash Flow</b>', small_style),
            para('<b>Cash-on-Cash</b>', small_style),
            para('<b>Payback</b>', small_style),
        ]]
        for scen in unique_keys:
            data = st.session_state['scenarios'].get(scen)
            if not data:
                continue
            metrics = compute_scenario_metrics(data, data.get('capex', base_capex))
            irr_delta = metrics.get('irr') - base_metrics['irr'] if base_metrics and scen != base_key else None
            npv_delta = metrics.get('npv') - base_metrics['npv'] if base_metrics and scen != base_key else None
            avg_cf_delta = metrics.get('avg_cf') - base_metrics['avg_cf'] if base_metrics and scen != base_key else None
            coc_delta = metrics.get('cash_on_cash') - base_metrics['cash_on_cash'] if base_metrics and scen != base_key else None
            equity_irr_val = data.get('equity_irr')
            equity_irr_text = f"{equity_irr_val:.2f}%" if equity_irr_val is not None and not np.isnan(equity_irr_val) else "N/A"
            comp_data.append([
                para(scen, small_style),
                para(safe_percent(metrics.get('irr'))),
                para(equity_irr_text),
                para(format_percent_delta(irr_delta) if irr_delta is not None else '—', small_style),
                para(format_metric_value(metrics.get('npv'))),
                para(format_currency_delta(npv_delta) if npv_delta is not None else '—', small_style),
                para(f"{format_metric_value(metrics.get('avg_cf'))} ({format_currency_delta(avg_cf_delta) if avg_cf_delta is not None else '—'})", small_style),
                para(format_metric_value(metrics.get('cash_on_cash'))),
                para(metrics.get('payback_year_label', 'N/A')),
            ])
        col_specs = [0.14, 0.09, 0.11, 0.11, 0.12, 0.12, 0.13, 0.1, 0.08]
        comp_table = Table(
            comp_data,
            hAlign='LEFT',
            colWidths=[spec * usable_width for spec in col_specs]
        )
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1f2933')),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        story.append(comp_table)
        story.append(Spacer(1, 12))
    else:
        story.append(para("No scenarios selected. Run a simulation in the app before exporting the report.", body_style))

    for scen in unique_keys:
        data = st.session_state['scenarios'].get(scen)
        if not data:
            continue
        metrics = compute_scenario_metrics(data, data.get('capex', base_capex))
        story.append(Paragraph(f"Scenario: {scen}", heading_style))
        inputs_line = (
            f"MSW {data.get('msw_tpd', msw_tpd):.0f} TPD | Tires {data.get('tires_tpd', tires_tpd):.0f} TPD | "
            f"Power ${data.get('power_price', power_price):.3f}/kWh | Inflation {format_rate(data.get('inflation', inflation))} | "
            f"CapEx Multiplier {data.get('capex_mult', capex_mult):.2f}"
        )
        story.append(para(f"<b>Inputs:</b> {inputs_line}", small_style))

        equity_irr_val = data.get('equity_irr')
        equity_irr_text = f"{equity_irr_val:.2f}%" if equity_irr_val is not None and not np.isnan(equity_irr_val) else "N/A"
        coc_val = metrics.get('cash_on_cash')
        coc_text = f"{coc_val:.2f}x" if coc_val is not None and not np.isnan(coc_val) else "N/A"

        metric_table_data = [
            [para('<b>Category</b>', small_style), para('<b>Details</b>', small_style)],
            [para('Core Metrics'), para(f"IRR {safe_percent(metrics.get('irr'))} | Equity IRR {equity_irr_text} | NPV {format_metric_value(metrics.get('npv'))} | Payback {metrics.get('payback_year_label', 'N/A')}")],
            [para('Totals'), para(f"Revenue {format_metric_value(metrics.get('total_revenue'))} | Expenses {format_metric_value(metrics.get('total_expenses'))}")],
            [para('Cash Flow'), para(f"Average {format_metric_value(metrics.get('avg_cf'))} | Peak {format_metric_value(metrics.get('peak_cf'))} | Min {format_metric_value(metrics.get('min_cf'))} | Cash-on-Cash {coc_text}")],
        ]
        if base_metrics and scen != base_key:
            irr_delta = metrics.get('irr') - base_metrics['irr']
            npv_delta = metrics.get('npv') - base_metrics['npv']
            avg_cf_delta = metrics.get('avg_cf') - base_metrics['avg_cf']
            coc_delta = metrics.get('cash_on_cash') - base_metrics['cash_on_cash']
            equity_irr_delta_row = None
            if base_equity_irr_value is not None and equity_irr_val is not None and not np.isnan(equity_irr_val):
                equity_irr_delta_row = equity_irr_val - base_equity_irr_value
            metric_table_data.append([
                para('Δ vs Base'),
                para(
                    f"IRR {format_percent_delta(irr_delta)} | Equity IRR {format_percent_delta(equity_irr_delta_row) if equity_irr_delta_row is not None else '—'} | "
                    f"NPV {format_currency_delta(npv_delta)} | Avg CF {format_currency_delta(avg_cf_delta)} | Cash-on-Cash {format_ratio_delta(coc_delta)}"
                )
            ])

        metric_table = Table(
            metric_table_data,
            hAlign='LEFT',
            colWidths=[0.2 * usable_width, 0.72 * usable_width]
        )
        metric_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1f2933')),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        story.append(metric_table)

        if st.session_state['custom_kpis']:
            kpi_items = []
            for kpi in st.session_state['custom_kpis']:
                try:
                    kpi_value = evaluate_custom_kpi(kpi['formula'], metrics)
                    text = f"<b>{kpi['name']}:</b> {format_metric_value(kpi_value)}"
                except Exception as exc:
                    text = f"<b>{kpi['name']}:</b> Unable to evaluate ({exc})"
                kpi_items.append(ListItem(Paragraph(text, small_style), bulletColor=colors.HexColor('#1f2933')))
            if kpi_items:
                story.append(Paragraph("Custom KPIs", subheading_style))
                story.append(ListFlowable(kpi_items, bulletType='bullet', start='•', leftPadding=12))

        insights = generate_insights(data)
        insight_items = [ListItem(Paragraph(text, small_style), bulletColor=colors.HexColor('#1f2933')) for text in insights]
        if insight_items:
            story.append(Paragraph("Key Insights", subheading_style))
            story.append(ListFlowable(insight_items, bulletType='bullet', start='•', leftPadding=12))

        story.append(Spacer(1, 12))

    # ─────────────────────────────────────────────────────────────────────────
    # EMBEDDED CHARTS (PNG images rendered from Plotly figures)
    # ─────────────────────────────────────────────────────────────────────────
    # App teal color palette (consistent with theme.css and in-app charts)
    PDF_TEAL_PALETTE = [
        '#069494',  # core teal (primary)
        '#5dd1c1',  # pale teal (secondary)
        '#035f5f',  # richer teal (accent)
        '#022c22',  # deep teal (dark accent)
        '#c4f5ec',  # very light teal
        '#0a7c7c',  # mid teal
    ]
    PDF_CHART_BG = '#fafafa'  # light background for print readability
    PDF_GRID_COLOR = 'rgba(6, 148, 148, 0.15)'
    PDF_TITLE_COLOR = '#035f5f'

    def _apply_teal_theme(fig):
        """Apply the app's teal color theme to a Plotly figure."""
        fig.update_layout(
            paper_bgcolor=PDF_CHART_BG,
            plot_bgcolor=PDF_CHART_BG,
            font=dict(family='Helvetica, Arial, sans-serif', color='#1f2933'),
            title=dict(font=dict(color=PDF_TITLE_COLOR, size=16)),
            colorway=PDF_TEAL_PALETTE,
            margin=dict(l=50, r=40, t=55, b=45),
        )
        fig.update_xaxes(
            gridcolor=PDF_GRID_COLOR,
            linecolor='#069494',
            tickfont=dict(color='#1f2933'),
        )
        fig.update_yaxes(
            gridcolor=PDF_GRID_COLOR,
            linecolor='#069494',
            tickfont=dict(color='#1f2933'),
        )
        return fig

    def _fig_to_image(fig, width=520, height=320):
        """Convert a Plotly figure to a ReportLab Image flowable."""
        try:
            img_bytes = fig.to_image(format='png', width=width, height=height, scale=2)
            from reportlab.platypus import Image as RLImage
            img_buffer = io.BytesIO(img_bytes)
            return RLImage(img_buffer, width=width, height=height)
        except Exception:
            return None

    chart_section_added = False

    # Chart 1: Revenue vs Expenses
    try:
        rev_exp_data = {'Year': [], 'Revenue': [], 'Expenses': [], 'Scenario': []}
        for scen in unique_keys[:3]:
            data = st.session_state['scenarios'].get(scen)
            if not data:
                continue
            rev = np.array(data.get('revenue', []), dtype=float)
            exp = np.array(data.get('expenses', []), dtype=float)
            n = min(len(years), rev.size, exp.size)
            for i in range(n):
                rev_exp_data['Year'].append(years[i])
                rev_exp_data['Revenue'].append(rev[i])
                rev_exp_data['Expenses'].append(exp[i])
                rev_exp_data['Scenario'].append(scen)
        if rev_exp_data['Year']:
            import plotly.express as px
            fig_rev = px.line(
                pd.DataFrame(rev_exp_data),
                x='Year', y='Revenue', color='Scenario',
                title='Revenue vs Expenses by Scenario',
            )
            fig_rev.add_scatter(
                x=rev_exp_data['Year'], y=rev_exp_data['Expenses'],
                mode='lines', name='Expenses', line=dict(dash='dot', color='#c4f5ec')
            )
            _apply_teal_theme(fig_rev)
            img = _fig_to_image(fig_rev)
            if img:
                if not chart_section_added:
                    story.append(Paragraph("Key Charts", heading_style))
                    chart_section_added = True
                story.append(Paragraph("Revenue vs Expenses", subheading_style))
                story.append(img)
                story.append(para(
                    "This chart compares annual revenue against operating expenses for each scenario. "
                    "A widening gap signals improving margins; convergence warns of margin compression.",
                    caption_style
                ))
                story.append(Spacer(1, 10))
    except Exception:
        pass

    # Chart 2: Cumulative Cash Flow
    try:
        cf_plot_data = {'Year': [], 'Cumulative CF': [], 'Scenario': []}
        for scen in unique_keys[:3]:
            data = st.session_state['scenarios'].get(scen)
            if not data:
                continue
            cf_full = np.array(data.get('cash_flows_full', []), dtype=float)
            if cf_full.size == 0:
                continue
            cum = np.cumsum(cf_full)
            yr_labels = [years[0] - 1] + list(years[:cf_full.size - 1]) if cf_full.size <= len(years) + 1 else list(range(cf_full.size))
            for i, val in enumerate(cum):
                cf_plot_data['Year'].append(yr_labels[i] if i < len(yr_labels) else yr_labels[-1] + i)
                cf_plot_data['Cumulative CF'].append(val)
                cf_plot_data['Scenario'].append(scen)
        if cf_plot_data['Year']:
            fig_cf = px.line(
                pd.DataFrame(cf_plot_data),
                x='Year', y='Cumulative CF', color='Scenario',
                title='Cumulative Cash Flow',
            )
            fig_cf.add_hline(y=0, line_dash='dash', line_color='#035f5f', line_width=2)
            _apply_teal_theme(fig_cf)
            img = _fig_to_image(fig_cf)
            if img:
                if not chart_section_added:
                    story.append(Paragraph("Key Charts", heading_style))
                    chart_section_added = True
                story.append(Paragraph("Cumulative Cash Flow", subheading_style))
                story.append(img)
                story.append(para(
                    "Cumulative cash flow shows when the project crosses payback (zero line). "
                    "Steeper post-payback slopes indicate stronger free-cash generation.",
                    caption_style
                ))
                story.append(Spacer(1, 10))
    except Exception:
        pass

    # Chart 3: IRR Comparison Bar
    try:
        irr_data = {'Scenario': [], 'IRR': []}
        for scen in unique_keys[:6]:
            data = st.session_state['scenarios'].get(scen)
            if not data:
                continue
            irr_val = data.get('irr')
            if irr_val is not None and np.isfinite(irr_val):
                irr_data['Scenario'].append(scen)
                irr_data['IRR'].append(irr_val)
        if irr_data['Scenario']:
            fig_irr = px.bar(
                pd.DataFrame(irr_data),
                x='Scenario', y='IRR',
                title='IRR Comparison',
                color='IRR',
                color_continuous_scale=[
                    [0, '#022c22'],
                    [0.25, '#035f5f'],
                    [0.5, '#069494'],
                    [0.75, '#5dd1c1'],
                    [1, '#c4f5ec'],
                ],
            )
            _apply_teal_theme(fig_irr)
            fig_irr.update_layout(showlegend=False, coloraxis_showscale=False)
            img = _fig_to_image(fig_irr, width=480, height=280)
            if img:
                if not chart_section_added:
                    story.append(Paragraph("Key Charts", heading_style))
                    chart_section_added = True
                story.append(Paragraph("IRR Comparison", subheading_style))
                story.append(img)
                story.append(para(
                    "Bar heights represent each scenario's internal rate of return. "
                    "Compare against your hurdle rate to identify viable configurations.",
                    caption_style
                ))
                story.append(Spacer(1, 10))
    except Exception:
        pass

    # Chart 4: NPV Comparison Bar
    try:
        npv_data = {'Scenario': [], 'NPV': []}
        for scen in unique_keys[:6]:
            data = st.session_state['scenarios'].get(scen)
            if not data:
                continue
            npv_val = data.get('npv')
            if npv_val is not None and np.isfinite(npv_val):
                npv_data['Scenario'].append(scen)
                npv_data['NPV'].append(npv_val)
        if npv_data['Scenario']:
            fig_npv = px.bar(
                pd.DataFrame(npv_data),
                x='Scenario', y='NPV',
                title='NPV Comparison',
                color='NPV',
                color_continuous_scale=[
                    [0, '#022c22'],
                    [0.25, '#035f5f'],
                    [0.5, '#069494'],
                    [0.75, '#5dd1c1'],
                    [1, '#c4f5ec'],
                ],
            )
            _apply_teal_theme(fig_npv)
            fig_npv.update_layout(showlegend=False, coloraxis_showscale=False)
            img = _fig_to_image(fig_npv, width=480, height=280)
            if img:
                if not chart_section_added:
                    story.append(Paragraph("Key Charts", heading_style))
                    chart_section_added = True
                story.append(Paragraph("NPV Comparison", subheading_style))
                story.append(img)
                story.append(para(
                    "Net Present Value measures the dollar value created above the discount rate. "
                    "Higher bars indicate more attractive investments.",
                    caption_style
                ))
                story.append(Spacer(1, 10))
    except Exception:
        pass

    if chart_section_added:
        story.append(Spacer(1, 6))

    chart_texts = [
        "Revenue vs Expenses – Compare top-line performance against cost escalation year over year.",
        "Cumulative Cash Flow – Observe payback timing and post-payback ramp-up.",
        "IRR & NPV Comparison – Stack scenarios against hurdle rates and value targets.",
        "Sensitivity Tornado – Rank drivers by IRR impact; prioritize hedges/contracts accordingly.",
        "Mass-Balance Sankey – Trace feedstock conversion to outputs to identify efficiency losses.",
        "Salary & CapEx Charts – Spot concentration of labor spend and supplier/geography exposure.",
        "Power Production & Heat Map – Assess operational stability and year-by-year performance intensity.",
    ]
    story.append(Paragraph("Interpreting the Interactive Dashboards", heading_style))
    chart_items = [ListItem(Paragraph(txt, small_style), bulletColor=colors.HexColor('#1f2933')) for txt in chart_texts]
    story.append(ListFlowable(chart_items, bulletType='bullet', start='•', leftPadding=12))
    story.append(Spacer(1, 12))

    story.append(Paragraph("AI Scenario Suggestions", heading_style))
    recs = st.session_state.get('recommendations', [])
    if recs:
        rec_items = []
        for rec in recs:
            rec_items.append(ListItem(Paragraph(f"{rec['label']}: {rec['summary']}", small_style), bulletColor=colors.HexColor('#1f2933')))
        story.append(ListFlowable(rec_items, bulletType='bullet', start='•', leftPadding=12))
    else:
        story.append(para("No recommendations available yet. Run a simulation to generate suggestions.", body_style))
    story.append(Spacer(1, 12))

    if st.session_state.get('optimizer_result'):
        opt_result = st.session_state['optimizer_result']
        opt_metrics = opt_result['metrics']
        opt_scenario = opt_result['scenario']
        story.append(Paragraph("Scenario Optimizer", heading_style))
        story.append(para(
            f"Target IRR {opt_result['target']:.2f}% | Samples {opt_result['samples']} | Tolerance ±{opt_result['tolerance']:.2f}%",
            small_style,
        ))
        story.append(para(opt_result['summary'], body_style))
        if opt_result.get('explanation'):
            story.append(para(opt_result['explanation'], body_style))
        optimizer_table = Table(
            [
                [para('<b>Metric</b>', small_style), para('<b>Value</b>', small_style)],
                [para('IRR'), para(f"{opt_metrics['irr']:.2f}%")],
                [para('NPV'), para(format_metric_value(opt_metrics['npv']))],
                [para('Average Cash Flow'), para(format_metric_value(opt_metrics['avg_cf']))],
                [para('Cash-on-Cash'), para(format_metric_value(opt_metrics['cash_on_cash']))],
                [para('Inputs'), para(
                    f"MSW {opt_scenario['msw_tpd']:.0f} TPD | Tires {opt_scenario['tires_tpd']:.0f} TPD | "
                    f"Power ${opt_scenario['power_price']:.3f}/kWh | CapEx x{opt_scenario['capex_mult']:.2f}"
                )],
            ],
            hAlign='LEFT',
            colWidths=[140, 360],
        )
        optimizer_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1f2933')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        story.append(optimizer_table)
        story.append(Spacer(1, 12))

    if st.session_state.get('forecast_results'):
        story.append(Paragraph("Forecast Explorer Summaries", heading_style))
        for forecast in st.session_state['forecast_results']:
            story.append(Paragraph(
                f"Scenario {forecast['scenario']} – {forecast['metric']} ({forecast['horizon']} year horizon)",
                subheading_style,
            ))
            story.append(para(forecast['summary'], body_style))
            forecast_rows = [[para('<b>Year</b>', small_style), para('<b>Projected</b>', small_style)]]
            for yr, val in zip(forecast['future_years'], forecast['future_values']):
                forecast_rows.append([para(str(yr), small_style), para(format_metric_value(val))])
            forecast_table = Table(forecast_rows, hAlign='LEFT', colWidths=[80, 120])
            forecast_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1f2933')),
                ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ]))
            story.append(forecast_table)
            story.append(Spacer(1, 8))

    if 'monte_carlo' in st.session_state:
        mc_data = st.session_state['monte_carlo']
        summary = mc_data['summary']
        config = mc_data['config']
        story.append(Paragraph("Monte Carlo Summary – Revenue Drivers", heading_style))
        story.append(para(
            f"Iterations {mc_data['iterations']:,} | Power σ {config['volatilities']['power_price']:.1f}% | MSW σ {config['volatilities']['msw_tpd']:.1f}% | Tires σ {config['volatilities']['tires_tpd']:.1f}%",
            small_style,
        ))

        mc_table_data = [[para('<b>Metric</b>', small_style), para('<b>Mean</b>', small_style), para('<b>P10</b>', small_style), para('<b>P90</b>', small_style)]]
        for metric_label, stats in summary.items():
            if metric_label.startswith('IRR'):
                mean = safe_percent(stats.get('mean'))
                p10 = safe_percent(stats.get('p10'))
                p90 = safe_percent(stats.get('p90'))
            elif metric_label.endswith('(x)'):
                def format_ratio(value):
                    return f"{value:.2f}x" if value is not None and not np.isnan(value) else 'N/A'
                mean = format_ratio(stats.get('mean'))
                p10 = format_ratio(stats.get('p10'))
                p90 = format_ratio(stats.get('p90'))
            else:
                mean = format_metric_value(stats.get('mean'))
                p10 = format_metric_value(stats.get('p10'))
                p90 = format_metric_value(stats.get('p90'))
            mc_table_data.append([
                para(metric_label, small_style),
                para(mean, small_style),
                para(p10, small_style),
                para(p90, small_style),
            ])
        mc_table = Table(mc_table_data, hAlign='LEFT', colWidths=[150, 70, 70, 70])
        mc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1f2933')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        story.append(mc_table)

        mc_summary_text = st.session_state.get('monte_carlo_summary_text')
        if mc_summary_text:
            story.append(para(f"Summary: {mc_summary_text}", body_style))

        irr_stats = summary.get('IRR (%)', {})
        if irr_stats.get('p10') is not None and not np.isnan(irr_stats.get('p10', float('nan'))):
            irr_p10 = irr_stats.get('p10')
            irr_p90 = irr_stats.get('p90')
            story.append(para(
                f"Interpretation: Approximately half the simulations fall between {irr_p10:.2f}% and {irr_p90:.2f}% IRR. Use these bands to set risk-adjusted targets and contingency plans.",
                body_style,
            ))
        story.append(para("Refer to the interactive app for histograms and scatter plots that visualize these distributions.", caption_style))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Notes", heading_style))
    notes = [
        "Use the Streamlit app to interact with charts, update assumptions, and re-run Monte Carlo simulations.",
        "Custom KPIs and insights derive directly from the uploaded Excel workbook; refresh the workbook to update figures.",
        "Chat assistant responses cite workbook tabs, scenario metrics, and Monte Carlo outputs for auditability.",
    ]
    note_items = [ListItem(Paragraph(note, small_style), bulletColor=colors.HexColor('#1f2933')) for note in notes]
    story.append(ListFlowable(note_items, bulletType='bullet', start='•', leftPadding=12))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

with tab_base:
    render_chat_interface("base")

with tab_compare:
    render_chat_interface("scenario")
    st.divider()
    pdf_bytes = generate_pdf()
    st.download_button(
        label="Export PDF Report",
        data=pdf_bytes,
        file_name="crec_report.pdf",
        mime="application/pdf",
    )
