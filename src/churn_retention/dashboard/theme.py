"""Offline design tokens and application-owned layout classes."""

import streamlit as st

CSS = """<style>
:root { --ink:#17243b; --muted:#526079; --line:#e0e5ee; --accent:#3949ab; }
html, body, .stApp { font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;
  color:var(--ink); background:#f5f6fa; font-variant-numeric:tabular-nums; }
.stApp { background:#f5f6fa; }
[data-testid="stMainBlockContainer"] { max-width:1440px; padding:4rem 2.8rem 4rem; }
[data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
[data-testid="stHeader"] { background:#f5f6fa; }
[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-radius:12px; }
h1,h2,h3 { color:var(--ink); letter-spacing:-.025em; }
h2 { font-size:1.35rem; } h3 { font-size:1.12rem; }
p, label, [data-testid="stMarkdownContainer"] { font-size:15px; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:var(--muted); }
button, input, select, textarea { font:inherit; }
button { min-height:42px; }
:focus-visible { outline:3px solid #6677db !important; outline-offset:3px; }
a { color:#3345a2; }
.page-heading { margin:0 0 1.8rem; max-width:900px; }
.eyebrow { color:#4757aa; font-weight:650; font-size:13px; margin-bottom:.5rem; }
.page-heading h1 { font-size:clamp(1.65rem,2.5vw,2.2rem); line-height:1.2; margin:0 0 .65rem; }
.page-heading p { color:var(--muted); font-size:15px; line-height:1.65; margin:0; }
.brand { display:flex; align-items:center; gap:12px; padding:.5rem 0 1.3rem; }
.brand-mark { width:38px; height:38px; border-radius:11px; background:#3949ab;
 color:#fff; display:grid; place-items:center; font-size:22px; font-weight:700; }
.brand strong { display:block; font-size:18px; letter-spacing:-.5px; }
.brand small { color:var(--muted); font-size:12px; }
.summary-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,170px),1fr));
 background:#fff; border:1px solid var(--line); border-radius:12px; margin:0 0 1.6rem; }
.summary-item { padding:22px 24px; min-width:0; }
.summary-label { color:#526079; font-size:14px; margin-bottom:10px; }
.summary-value { color:#17243b; font-size:clamp(1.5rem,2.3vw,2rem); font-weight:650;
 letter-spacing:-.04em; line-height:1.2; overflow-wrap:anywhere; }
.summary-note { color:#526079; font-size:12px; margin-top:8px; line-height:1.5; }
.status { display:inline-block; border-radius:6px; font-size:13px; font-weight:600;
 padding:5px 10px; margin-bottom:10px; background:#edf0f7; color:#36425a; }
.status-success { background:#e7f5ec; color:#14643e; }
.status-warning { background:#fff1d9; color:#865009; }
.status-danger { background:#ffebee; color:#a82337; }
.empty-state { background:#fff; border:1px dashed #c7cfdf; border-radius:12px;
 padding:42px 28px; margin-bottom:1.1rem; text-align:center; }
.empty-state h3 { font-size:19px; margin:.8rem 0; }
.empty-state p { color:var(--muted); max-width:540px; margin:auto; line-height:1.7; }
.empty-icon { color:var(--accent); font-size:32px; }
.source-note { padding:12px 0 18px; color:var(--muted); font-size:14px; }
@media(max-width:1100px) { [data-testid="stMainBlockContainer"] { padding:4rem 1.5rem 3rem; } }
@media(max-width:640px) {
 [data-testid="stMainBlockContainer"] { padding:4rem 1rem 3rem; }
 .summary-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
 .summary-item { padding:18px 14px; }
 .page-heading { margin-bottom:1.4rem; }
 .empty-state { padding:28px 18px; }
}
</style>
"""


def apply_theme() -> None:
    st.html(CSS)
