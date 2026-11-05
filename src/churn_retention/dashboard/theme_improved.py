"""Improved theme with better visual comfort and modern design."""

import streamlit as st

# ── Light theme CSS variables ────────────────────────────────────────────────
_LIGHT_VARS = """
:root {
  /* Softer, more accessible color palette */
  --ink: #1a202c;           /* Softer dark gray instead of harsh black-blue */
  --muted: #718096;         /* Softer gray for muted text */
  --line: #e2e8f0;         /* Lighter border color */
  --accent: #4f46e5;       /* Indigo 600 - softer, more modern accent */
  --background: #f8fafc;   /* Light blue-gray background */
  --surface: #ffffff;      /* White surfaces */
  --success: #059669;      /* Green 600 - softer success color */
  --warning: #d97706;      /* Amber 600 - softer warning */
  --danger: #dc2626;       /* Red 600 - slightly softer danger */
  --info: #2563eb;         /* Blue 600 - for info */
}"""

# ── Dark theme CSS variables ─────────────────────────────────────────────────
_DARK_VARS = """
:root {
  --background: #0f172a;
  --surface: #1e293b;
  --ink: #f1f5f9;
  --muted: #94a3b8;
  --line: #334155;
  --accent: #818cf8;       /* Lighter indigo for dark background */
  --success: #10b981;
  --warning: #f59e0b;
  --danger: #ef4444;
  --info: #60a5fa;
}"""

# ── Dark mode sidebar / header overrides ─────────────────────────────────────
_DARK_OVERRIDES = """
[data-testid="stSidebar"] {
  background: #1e293b !important;
  border-right: 1px solid #334155 !important;
}

[data-testid="stHeader"] {
  background: #0f172a !important;
}

.stApp {
  background: #0f172a !important;
  color: #f1f5f9 !important;
}

/* Headings and markdown */
h1, h2, h3, h4, h5, h6 {
  color: #f8fafc !important;
}

p, label, span, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] span {
  color: #cbd5e1 !important;
}

/* Containers and panels */
[data-testid="stVerticalBlockBorderWrapper"],
[class*="st-key-panel-"],
.summary-grid,
.empty-state {
  background: #1e293b !important;
  border-color: #334155 !important;
}

/* Summary items */
.summary-item {
  border-color: #334155 !important;
}
.summary-value {
  color: #f8fafc !important;
}
.summary-label, .summary-note {
  color: #94a3b8 !important;
}

/* Inputs, textareas, selects */
[data-baseweb="input"], [data-baseweb="base-input"] {
  background-color: #1e293b !important;
  color: #f1f5f9 !important;
  border-color: #334155 !important;
}
input, textarea {
  color: #f1f5f9 !important;
  background-color: transparent !important;
}
[data-baseweb="select"] > div {
  background-color: #1e293b !important;
  color: #f1f5f9 !important;
  border-color: #334155 !important;
}

/* Segmented controls & pills */
[data-testid="stSegmentedControl"] [data-baseweb="button-group"] {
  background-color: #1e293b !important;
  border-color: #334155 !important;
}

/* Secondary buttons */
button[data-testid="baseButton-secondary"] {
  background: #1e293b !important;
  border: 1px solid #334155 !important;
  color: #f1f5f9 !important;
}

/* Metrics */
div[data-testid="stMetricValue"] {
  color: #f8fafc !important;
}
div[data-testid="stMetricLabel"] {
  color: #94a3b8 !important;
}

/* Expanders */
[data-testid="stExpander"] {
  background: #1e293b !important;
  border-color: #334155 !important;
}
[data-testid="stExpander"] summary {
  color: #f1f5f9 !important;
}

/* Dataframe & Tables */
[data-testid="stDataFrame"] {
  border-color: #334155 !important;
}

/* Status badge colours */
.status {
  background: #334155 !important;
  color: #cbd5e1 !important;
  border-color: #475569 !important;
}

/* Top bar fixed in the top right corner */
.st-key-top_bar_container {
  background: #1e293b !important;
  border-color: #334155 !important;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45) !important;
}
"""

IMPROVED_CSS = """<style>
:root {
  /* Softer, more accessible color palette */
  --ink: #1a202c;           /* Softer dark gray instead of harsh black-blue */
  --muted: #718096;         /* Softer gray for muted text */
  --line: #e2e8f0;         /* Lighter border color */
  --accent: #4f46e5;       /* Indigo 600 - softer, more modern accent */
  --background: #f8fafc;   /* Light blue-gray background */
  --surface: #ffffff;      /* White surfaces */
  --success: #059669;      /* Green 600 - softer success color */
  --warning: #d97706;      /* Amber 600 - softer warning */
  --danger: #dc2626;       /* Red 600 - slightly softer danger */
  --info: #2563eb;         /* Blue 600 - for info */
}

html, body, .stApp {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
  color: var(--ink);
  background: var(--background);
  font-variant-numeric: tabular-nums;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.stApp {
  background: var(--background);
}

/* Improved spacing and layout */
[data-testid="stMainBlockContainer"] {
  max-width: 1400px;
  padding: 1.25rem 2.5rem 3rem;
}

[data-testid="stSidebar"] {
  background: var(--surface);
  border-right: 1px solid var(--line);
}

[data-testid="stHeader"] {
  background: var(--background);
}

/* Softer containers and panels */
[data-testid="stVerticalBlockBorderWrapper"],
[class*="st-key-panel-"] {
  background: var(--surface);
  border-radius: 16px;
  border: 1px solid var(--line);
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px 0 rgba(0, 0, 0, 0.03);
}

/* Improved typography */
h1, h2, h3 {
  color: var(--ink);
  letter-spacing: -0.025em;
  font-weight: 600;
}

h1 {
  font-size: clamp(1.75rem, 2.5vw, 2.4rem);
  line-height: 1.2;
  margin: 0 0 0.75rem;
}

h2 {
  font-size: 1.4rem;
  margin: 0 0 0.5rem;
}

h3 {
  font-size: 1.15rem;
  margin: 0 0 0.5rem;
}

p, label, [data-testid="stMarkdownContainer"] {
  font-size: 15px;
  line-height: 1.7;
  color: var(--muted);
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
  color: var(--muted);
  font-size: 14px;
}

/* Improved buttons and inputs */
button {
  min-height: 44px;
  border-radius: 10px;
  font-weight: 500;
}

button[data-testid="baseButton-primary"] {
  background: linear-gradient(135deg, var(--accent) 0%, #3730a3 100%);
  border: none;
}

button[data-testid="baseButton-secondary"] {
  background: var(--surface);
  border: 1px solid var(--line);
  color: var(--ink);
}

:focus-visible {
  outline: 3px solid var(--accent) !important;
  outline-offset: 3px;
  outline-color: rgba(79, 70, 229, 0.3) !important;
}

a {
  color: var(--accent);
  text-decoration: none;
  font-weight: 500;
}

a:hover {
  text-decoration: underline;
  text-underline-offset: 3px;
}

/* Improved page heading */
.page-heading {
  margin: 0 0 2.2rem;
  max-width: 900px;
}

.eyebrow {
  color: var(--accent);
  font-weight: 600;
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.75rem;
}

.page-heading p {
  color: var(--muted);
  font-size: 16px;
  line-height: 1.7;
  margin: 0;
  max-width: 700px;
}

/* Improved summary grid */
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 200px), 1fr));
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 16px;
  margin: 0 0 2rem;
  overflow: hidden;
}

.summary-item {
  padding: 24px 28px;
  min-width: 0;
  border-right: 1px solid var(--line);
}

.summary-item:last-child {
  border-right: none;
}

.summary-label {
  color: var(--muted);
  font-size: 14px;
  font-weight: 500;
  margin-bottom: 12px;
}

.summary-value {
  color: var(--ink);
  font-size: clamp(1.6rem, 2.3vw, 2.1rem);
  font-weight: 700;
  letter-spacing: -0.03em;
  line-height: 1.2;
  overflow-wrap: anywhere;
}

.summary-note {
  color: var(--muted);
  font-size: 13px;
  margin-top: 8px;
  line-height: 1.5;
}

/* Improved status badges */
.status {
  display: inline-flex;
  align-items: center;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 600;
  padding: 6px 12px;
  margin-bottom: 12px;
  background: #f1f5f9;
  color: #475569;
  border: 1px solid #e2e8f0;
}

.status-success {
  background: #d1fae5;
  color: var(--success);
  border-color: #a7f3d0;
}

.status-warning {
  background: #fef3c7;
  color: var(--warning);
  border-color: #fde68a;
}

.status-danger {
  background: #fee2e2;
  color: var(--danger);
  border-color: #fecaca;
}

.status-info {
  background: #dbeafe;
  color: var(--info);
  border-color: #bfdbfe;
}

/* Improved empty states */
.empty-state {
  background: var(--surface);
  border: 2px dashed #cbd5e1;
  border-radius: 16px;
  padding: 48px 32px;
  margin-bottom: 1.5rem;
  text-align: center;
}

.empty-state h3 {
  font-size: 20px;
  margin: 1rem 0 0.75rem;
  color: var(--ink);
}

.empty-state p {
  color: var(--muted);
  max-width: 540px;
  margin: 0 auto;
  line-height: 1.7;
}

.empty-icon {
  color: var(--accent);
  font-size: 40px;
  opacity: 0.8;
}

/* Improved tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
}

.stTabs [data-baseweb="tab"] {
  border-radius: 10px;
  padding: 10px 20px;
  font-weight: 500;
}

.stTabs [aria-selected="true"] {
  background: var(--accent);
  color: white;
}

/* Improved data tables */
[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  border-radius: 12px;
  overflow: hidden;
}

/* Better spacing for columns */
[data-testid="column"] {
  padding: 0 12px;
}

/* Improved expander */
[data-testid="stExpander"] {
  border: 1px solid var(--line);
  border-radius: 12px;
  overflow: hidden;
}

/* Responsive improvements */
@media (max-width: 1100px) {
  [data-testid="stMainBlockContainer"] {
    padding: 3rem 1.8rem 3rem;
  }
  
  .summary-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  
  .summary-item {
    padding: 20px 22px;
    border-right: none;
    border-bottom: 1px solid var(--line);
  }
  
  .summary-item:nth-child(even) {
    border-right: none;
  }
  
  .summary-item:nth-last-child(-n+2) {
    border-bottom: none;
  }
}

@media (max-width: 768px) {
  [data-testid="stMainBlockContainer"] {
    padding: 2.5rem 1.2rem 2.5rem;
  }
  
  .summary-grid {
    grid-template-columns: 1fr;
  }
  
  .summary-item {
    border-right: none;
    border-bottom: 1px solid var(--line);
    padding: 18px 20px;
  }
  
  .summary-item:last-child {
    border-bottom: none;
  }
  
  .page-heading {
    margin-bottom: 1.8rem;
  }
  
  .empty-state {
    padding: 32px 20px;
  }
}

/* Top bar container fixed right at the top corner */
.st-key-top_bar_container {
  position: fixed !important;
  top: 8px !important;
  right: 72px !important;
  z-index: 999999 !important;
  width: 320px !important;
  max-width: 90vw !important;
  background: var(--surface) !important;
  border: 1px solid var(--line) !important;
  border-radius: 20px !important;
  padding: 4px 8px !important;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08) !important;
  margin: 0 !important;
}

.st-key-top_bar_container [data-testid="stHorizontalBlock"] {
  align-items: center !important;
  gap: 6px !important;
  margin: 0 !important;
  padding: 0 !important;
}

.st-key-top_bar_container [data-testid="column"] {
  padding: 0 2px !important;
}

.st-key-top_bar_container button {
  min-height: 32px !important;
  height: 32px !important;
  padding: 0 10px !important;
  font-size: 13px !important;
  border-radius: 14px !important;
  line-height: 1 !important;
  white-space: nowrap !important;
}

.st-key-top_bar_container [data-testid="stSegmentedControl"] {
  margin: 0 !important;
  padding: 0 !important;
}

.st-key-top_bar_container [data-testid="stSegmentedControl"] [data-baseweb="button-group"] {
  height: 32px !important;
  min-height: 32px !important;
  border-radius: 14px !important;
}

.st-key-top_bar_container [data-testid="stSegmentedControl"] button {
  height: 32px !important;
  min-height: 32px !important;
  font-size: 12px !important;
  padding: 0 8px !important;
  line-height: 1 !important;
}
</style>
"""

def apply_improved_theme() -> None:
    """Apply the improved, more visually comfortable theme.

    Injects IMPROVED_CSS (which contains the shared structural rules) and then
    a second ``<style>`` block with either the light or dark CSS variable
    overrides depending on ``st.session_state.get("ui_dark_mode")``.
    """
    # Base structural styles (always applied)
    st.html(IMPROVED_CSS)
    # Dynamic variable overrides based on current mode
    dark = st.session_state.get("ui_dark_mode", False)
    if dark:
        st.html(f"<style>{_DARK_VARS}\n{_DARK_OVERRIDES}</style>")
    else:
        st.html(f"<style>{_LIGHT_VARS}</style>")