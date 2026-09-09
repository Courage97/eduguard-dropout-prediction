import time
from contextlib import contextmanager

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import (
    load_artefacts, encode_input, get_unscaled,
    predict_student, shap_waterfall_chart,
)
from utils.interventions import get_interventions

st.set_page_config(
    page_title="EduGuard — Batch Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Palette (matches the rest of EduGuard) ──────────────────────────────────
BLUE1  = "#3A4EFB"
BLUE2  = "#33A4FA"
YELLOW = "#E3FF3B"
DARK   = "#252943"
WHITE  = "#FFFFFF"
RED    = "#FF4B4B"
GREEN  = "#00C48C"
AMBER  = "#FFB800"
PURPLE = "#7B2FBE"
BORDER = "#E0E3FF"
CARD   = "#F4F5FF"
MUTED  = "#6B7280"


# ── Small helper so every open <div wrapper> gets closed exactly once ──────
@contextmanager
def padded(extra_style=""):
    st.markdown(f"<div style='{extra_style}'>", unsafe_allow_html=True)
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)


def risk_gauge_chart(prob, clr):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 44, "color": clr, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": BORDER,
                     "tickfont": {"size": 12, "color": "#9CA3AF"}},
            "bar": {"color": clr, "thickness": 0.32},
            "bgcolor": CARD,
            "borderwidth": 0,
        },
    ))
    fig.update_layout(
        height=170,
        margin=dict(l=24, r=24, t=16, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
*,html,body,[class*="css"]{font-family:'Inter',sans-serif!important;box-sizing:border-box;;}
/* Streamlit renders built-in icons (status/expander check, tooltips, toolbar
   buttons) as ligature spans on a Material icon font — e.g. data-testid=
   "stExpanderIconCheck" whose text content is literally "check". The blanket
   Inter override above breaks that ligature and leaves the raw word showing,
   so hand icon-bearing elements back their own font. */
[data-testid*="Icon"]{font-family:'Material Symbols Rounded'!important;}
.main{background:#F7F8FF!important;font-size:21px;}
.block-container{padding:0 56px!important;max-width:100%!important;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none;}

/* Base font-size bump: Streamlit's native widget text (labels, help text,
   multiselect, dataframe, uploader) renders at browser default (~14px)
   unless told otherwise — bump the shared baseline across the page. */
[data-testid="stWidgetLabel"] p{font-size:20px!important;font-weight:600!important;}
[data-testid="stMarkdownContainer"] p{font-size:20px;}
[data-testid="stDataFrame"] *{font-size:19px!important;}
[data-testid="stMultiSelect"] *{font-size:20px!important;}
[data-testid="stSelectbox"] *{font-size:20px!important;}
[data-testid="stFileUploaderDropzone"] *{font-size:20px!important;}
[data-testid="stExpander"] summary p{font-size:20px!important;font-weight:600!important;}
[data-testid="stCaptionContainer"] p{font-size:19px!important;}

/* ── Navbar (shared look across all EduGuard pages) ───────────────────── */
.edu-nav{position:sticky;top:0;z-index:9999;background:rgba(37,41,67,0.92);
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  margin:0 -56px;
  padding:0 56px;display:flex;align-items:center;justify-content:space-between;
  min-height:72px;border-bottom:1px solid rgba(255,255,255,0.07);}
.edu-brand{display:flex;align-items:center;gap:13px;text-decoration:none!important;}
.edu-brand-icon{background:linear-gradient(135deg,#3A4EFB,#33A4FA);width:42px;height:42px;
  border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:26px;
  box-shadow:0 6px 16px rgba(58,78,251,0.35);}
.edu-brand-name{color:#fff;font-size:25px;font-weight:900;letter-spacing:-0.5px;line-height:1.1;}
.edu-brand-sub{color:#E3FF3B;font-size:8px;font-weight:700;letter-spacing:2px;
  text-transform:uppercase;margin-top:2px;}
.edu-navlinks{display:flex;align-items:center;gap:2px;}
.edu-navlinks a{color:rgba(222,224,237,0.72)!important;font-size:15px;font-weight:500;
  padding:9px 18px;border-radius:8px;text-decoration:none!important;transition:color .2s,background .2s;}
.edu-navlinks a:hover{color:#fff!important;background:rgba(255,255,255,0.07);}
.edu-navlinks a.active{color:#E3FF3B!important;background:rgba(227,255,59,0.1);}
.edu-live{display:inline-flex;align-items:center;gap:7px;background:rgba(227,255,59,0.12);
  border:1px solid rgba(227,255,59,0.4);color:#E3FF3B;padding:7px 16px;border-radius:24px;
  font-size:10px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;}
.edu-live-dot{width:6px;height:6px;border-radius:50%;background:#E3FF3B;
  animation:eduPulse 1.8s ease-in-out infinite;}
@keyframes eduPulse{0%,100%{opacity:1;}50%{opacity:0.35;}}

/* ── Page header ────────────────────────────────────────────────────── */
.page-wrap{padding:40px 0 8px;}
.page-header{margin-bottom:22px;}
.page-eyebrow{font-size:18px;font-weight:700;letter-spacing:2.5px;
  text-transform:uppercase;color:#3A4EFB;margin:0 0 10px;}
.page-h1{font-size:45px;font-weight:900;color:#252943!important;letter-spacing:-1px;
  margin:0 0 10px!important;padding:0!important;line-height:1.15;}
.page-sub{font-size:18px;color:#6B7280;margin:0 0 18px;line-height:1.65;max-width:660px;}
.context-badges{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px;}
.context-badge{background:#fff;border:1px solid #E0E3FF;color:#374151;
  padding:7px 15px;border-radius:20px;font-size:18px;font-weight:600;}

/* ── Shared card header (upload steps, charts, table, drill-down) ────── */
[data-testid="stForm"]{background:transparent;border:none;padding:0;}
.domain-card-head{display:flex;align-items:center;gap:10px;margin:14px;
  padding-bottom:12px;border-bottom:1px solid #EEF0FA;}
.domain-card-icon{width:34px;height:34px;border-radius:10px;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;font-size:22px;}
.domain-card-title{font-size:22px;font-weight:800;letter-spacing:0.1px;}

.step-num-badge{display:inline-flex;align-items:center;justify-content:center;
  width:28px;height:28px;border-radius:50%;color:#fff;font-size:18px;
  font-weight:800;flex-shrink:0;vertical-align:middle;}

/* ── Dominant upload dropzone — marker + :has() trick, same technique as
   the reveal-* animation hooks further down: an invisible marker span
   inside the container lets us style exactly that bordered wrapper
   without Streamlit exposing a class hook for st.container(border=True). */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.upload-zone-marker){
  border:2px dashed #B9C0F7!important;border-radius:20px!important;
  background:linear-gradient(180deg,#F8F9FF 0%,#FFFFFF 100%)!important;
  box-shadow:none!important;}
[data-testid="stFileUploaderDropzone"]{min-height:168px!important;
  display:flex!important;flex-direction:column!important;
  align-items:center!important;justify-content:center!important;gap:6px;}
[data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"]{
  font-size:41px!important;margin-bottom:4px;}
[data-testid="stFileUploaderDropzoneInstructions"] *{font-size:20px!important;}

/* ── Column reference card (replaces the old CSV format guide expander) ── */
.colref-card{background:#fff;border:1px solid #E0E3FF;border-radius:18px;
  padding:24px 26px;box-shadow:0 4px 18px rgba(58,78,251,0.06);margin-bottom:6px;}
.colref-domain{margin-bottom:18px;}
.colref-domain:last-child{margin-bottom:0;}
.colref-domain-label{display:flex;align-items:center;gap:8px;font-size:18px;
  font-weight:800;letter-spacing:0.4px;text-transform:uppercase;margin-bottom:10px;}
.colref-chips{display:flex;flex-wrap:wrap;gap:8px;}
.colref-chip{background:#F7F8FF;border:1px solid #E0E3FF;border-radius:8px;
  padding:6px 13px;font-size:18px;font-weight:600;color:#374151;cursor:help;
  transition:border-color .15s ease,background .15s ease;}
.colref-chip:hover{background:#F0F1FF;border-color:#C7CCF5;}

button[data-testid^="stBaseButton-primary"]{
  background:linear-gradient(90deg,#3A4EFB,#33A4FA)!important;
  color:#fff!important;border:none!important;border-radius:12px!important;
  font-weight:700!important;font-size:21px!important;padding:14px 28px!important;
  box-shadow:0 10px 26px rgba(58,78,251,0.25)!important;transition:transform .15s ease;}
button[data-testid^="stBaseButton-primary"] p{color:#fff!important;font-weight:700!important;font-size:21px!important;}
button[data-testid^="stBaseButton-primary"]:hover{transform:translateY(-2px);}
button[data-testid^="stBaseButton-secondary"]{border-radius:10px!important;
  font-weight:600!important;border-color:#E0E3FF!important;}
button[data-testid^="stBaseButton-secondary"] p{font-size:19px!important;}

/* File uploader dropzone — restyle Streamlit's default dashed box to match
   the rest of the card system instead of the plain grey default. */
[data-testid="stFileUploaderDropzone"]{
  background:#F7F8FF!important;border:1.5px dashed #C7CCF5!important;
  border-radius:14px!important;}
/* The dropzone's upload glyph is a ligature-font icon span (stIconMaterial,
   text content "upload"), not an <svg> — recolor via `color`, not `fill`. */
[data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"]{
  color:#3A4EFB!important;}

/* Multiselect chips — brand-blue instead of Streamlit's default theme-red
   (#FF4B4B, which is Streamlit's own primaryColor default and happens to
   match this app's RED constant by coincidence). The chip itself is a
   react-aria span identified by a stable data-tag attribute, not the
   [data-baseweb="tag"] BaseWeb convention this version doesn't render. */
[data-testid="stMultiSelectTagsContainer"] span[data-tag]{
  background:#3A4EFB!important;border-radius:8px!important;}

.section-divider{height:1px;background:#E0E3FF;margin:32px 0 28px;border:none;}
.section-h2{font-size:31px;font-weight:800;color:#252943!important;
  letter-spacing:-0.4px;margin:0 0 6px!important;padding:0!important;}
.section-h3{font-size:27px;font-weight:800;color:#252943!important;
  letter-spacing:-0.3px;margin:0 0 6px!important;padding:0!important;}
.section-caption{font-size:19px;color:#6B7280;margin:0 0 18px;}

/* ── Alert banner ───────────────────────────────────────────────────── */
.alert-card{border-radius:16px;padding:20px 24px;margin:4px 0 8px;
  display:flex;align-items:flex-start;gap:16px;box-shadow:0 4px 18px rgba(0,0,0,0.05);}
.alert-card.is-danger{background:#FFF5F5;border:1px solid #FFD3D3;}
.alert-card.is-ok{background:#F0FFF9;border:1px solid #B8F5E1;}
.alert-icon-circle{width:46px;height:46px;border-radius:12px;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;font-size:26px;}
.alert-card.is-danger .alert-icon-circle{background:#FF4B4B18;}
.alert-card.is-ok .alert-icon-circle{background:#00C48C18;}
.alert-title{font-size:22px;font-weight:800;margin:0 0 4px;}
.alert-card.is-danger .alert-title{color:#FF4B4B;}
.alert-card.is-ok .alert-title{color:#00C48C;}
.alert-body{font-size:20px;color:#374151;margin:0;line-height:1.6;}

/* ── Cohort KPI tiles ───────────────────────────────────────────────── */
.kpi-card{background:#fff;border:1px solid #E0E3FF;border-radius:16px;
  padding:20px 18px;box-shadow:0 4px 18px rgba(58,78,251,0.07);
  transition:transform .2s ease, box-shadow .2s ease;}
.kpi-card:hover{transform:translateY(-4px);box-shadow:0 14px 28px rgba(58,78,251,0.13);}
.kpi-icon{width:38px;height:38px;border-radius:11px;display:flex;align-items:center;
  justify-content:center;font-size:23px;margin-bottom:14px;}
.kpi-num{font-size:37px;font-weight:900;letter-spacing:-1.2px;line-height:1;}
.kpi-label{font-size:18px;color:#6B7280;font-weight:700;margin-top:7px;
  text-transform:uppercase;letter-spacing:0.4px;}
.kpi-sub{font-size:18px;color:#9CA3AF;margin-top:4px;}

/* ── Result hero (drill-down) ───────────────────────────────────────── */
.result-id{font-size:18px;color:#6B7280;font-weight:700;text-transform:uppercase;
  letter-spacing:0.8px;margin:0 0 8px;}
.result-level-badge{display:inline-block;padding:6px 16px;border-radius:20px;
  font-size:19px;font-weight:800;letter-spacing:0.3px;margin-bottom:14px;}
.result-summary{font-size:21px;color:#374151;line-height:1.7;max-width:520px;margin:0;}
.mini-card{background:#fff;border:1px solid #E0E3FF;padding:16px 18px;
  border-radius:14px;text-align:center;margin-bottom:12px;
  box-shadow:0 4px 14px rgba(58,78,251,0.06);}
.mini-card-num{font-size:37px;font-weight:900;letter-spacing:-1px;line-height:1;}
.mini-card-label{font-size:12px;color:#6B7280;font-weight:500;margin-top:5px;}

.tip-banner{background:#F4F5FF;border:1px dashed #C7CCF5;border-radius:12px;
  padding:14px 18px;font-size:19px;color:#3A4EFB;font-weight:600;margin:24px 0 8px;}

/* ── Entrance animation shared by driver + intervention cards ─────────── */
@keyframes eduCardIn{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:translateY(0);}}

/* ── Driver cards (grid-aligned, matches the predictor page) ──────────── */
.driver-card{background:#fff;border:1px solid #E0E3FF;border-radius:16px;
  padding:22px 20px;box-shadow:0 4px 18px rgba(58,78,251,0.07);
  transition:transform .2s ease, box-shadow .2s ease;
  display:flex;flex-direction:column;justify-content:space-between;height:100%;
  animation:eduCardIn .4s ease both;}
.driver-card:hover{transform:translateY(-4px);box-shadow:0 16px 32px rgba(58,78,251,0.14);}
.driver-direction{font-size:17px;font-weight:700;text-transform:uppercase;
  letter-spacing:1px;margin-bottom:8px;}
.driver-name{font-size:21px;font-weight:700;color:#252943;
  margin-bottom:10px;line-height:1.3;min-height:42px;display:flex;align-items:flex-end;}
.driver-shap{font-size:39px;font-weight:900;letter-spacing:-1px;}
.driver-val{font-size:18px;color:#9CA3AF;margin-top:6px;}
.driver-bar-track{height:6px;background:#F0F1FA;border-radius:6px;overflow:hidden;margin-top:14px;}
.driver-bar-fill{height:100%;border-radius:6px;transition:width .7s ease;}

/* ── Intervention accordion (grouped by urgency, collapsible) ─────────── */
.interv-accordion{display:flex;flex-direction:column;gap:14px;}
.interv-group{background:#fff;border:1px solid #E0E3FF;border-radius:16px;
  overflow:hidden;transition:box-shadow .2s ease;}
.interv-group[open]{box-shadow:0 10px 28px rgba(58,78,251,0.09);}
.interv-group-summary{list-style:none;cursor:pointer;padding:18px 22px;
  display:flex;align-items:center;gap:12px;font-weight:800;font-size:20.5px;
  user-select:none;}
.interv-group-summary::-webkit-details-marker{display:none;}
.interv-group-summary::after{content:"›";margin-left:auto;font-size:25px;
  color:#9CA3AF;transition:transform .2s ease;}
.interv-group[open]>.interv-group-summary::after{transform:rotate(90deg);}
.interv-group-badge{font-size:17.5px;font-weight:800;padding:3px 11px;border-radius:20px;}
.interv-group-body{padding:2px 22px 22px;}
.interv-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
  gap:16px;align-items:stretch;}

.interv-urgent,.interv-medium,.interv-monitor{
  display:flex;flex-direction:column;height:100%;border-radius:12px;
  padding:18px 20px;animation:eduCardIn .35s ease both;transition:transform .15s ease;}
.interv-urgent{border-left:4px solid #FF4B4B;background:#FFF5F5;}
.interv-medium{border-left:4px solid #FFB800;background:#FFFBF0;}
.interv-monitor{border-left:4px solid #00C48C;background:#F0FFF9;}
.interv-urgent:hover,.interv-medium:hover,.interv-monitor:hover{transform:translateY(-3px);}
.interv-head{display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap;}
.driver-chip{background:#F4F5FF;color:#3A4EFB;padding:3px 12px;border-radius:20px;
  font-size:17px;font-weight:600;border:1px solid #E0E3FF;}
.interv-party{font-size:18px;color:#9CA3AF;}
.interv-action{font-size:20px;color:#252943;font-weight:500;
  margin:0 0 10px;line-height:1.5;flex-grow:1;}
.rationale-toggle{margin-top:auto;padding-top:10px;border-top:1px solid rgba(0,0,0,0.08);}
.rationale-toggle summary{cursor:pointer;font-size:17.5px;font-weight:700;
  color:#3A4EFB;list-style:none;display:flex;align-items:center;gap:6px;user-select:none;}
.rationale-toggle summary::-webkit-details-marker{display:none;}
.rationale-toggle summary::after{content:"›";margin-left:auto;transition:transform .2s ease;}
.rationale-toggle[open] summary::after{transform:rotate(90deg);}
.rationale-toggle-body{font-size:18px;color:#6B7280;line-height:1.6;padding-top:8px;}

.interv-legend{display:flex;gap:18px;flex-wrap:wrap;margin:14px 0 18px;}
.interv-legend-item{display:flex;align-items:center;gap:7px;font-size:18.5px;
  color:#6B7280;font-weight:500;}
.interv-legend-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0;}

/* ── Footer ─────────────────────────────────────────────────────────── */
.edu-footer{background:#252943;padding:32px 56px 24px;margin:56px -56px 0;}
.edu-footer-credit{color:#7A7F9A;font-size:18px;text-align:center;line-height:1.6;}

/* ── Reveal animation ───────────────────────────────────────────────────
   Same marker + :has() technique as the predictor page: an invisible
   marker span placed as the first element inside a Streamlit-generated
   stVerticalBlock lets us animate exactly that block (and nothing above
   or below it) without Streamlit exposing a class hook of its own. Two
   independent cascades: "a" for the cohort results that appear after a
   CSV upload, "b" for the single-student drill-down that appears after
   its own button click — each restarts its own stagger from 0s. */
@keyframes fadeInUp{from{opacity:0;transform:translateY(18px);}
  to{opacity:1;transform:translateY(0);}}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .reveal-a1){
  animation:fadeInUp .5s cubic-bezier(.22,.8,.32,1) both;}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .reveal-a2){
  animation:fadeInUp .5s cubic-bezier(.22,.8,.32,1) .08s both;}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .reveal-a3){
  animation:fadeInUp .5s cubic-bezier(.22,.8,.32,1) .16s both;}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .reveal-a4){
  animation:fadeInUp .5s cubic-bezier(.22,.8,.32,1) .22s both;}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .reveal-a5){
  animation:fadeInUp .5s cubic-bezier(.22,.8,.32,1) .28s both;}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .reveal-a6){
  animation:fadeInUp .5s cubic-bezier(.22,.8,.32,1) .36s both;}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .reveal-b1){
  animation:fadeInUp .55s cubic-bezier(.22,.8,.32,1) both;}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .reveal-b2){
  animation:fadeInUp .55s cubic-bezier(.22,.8,.32,1) .10s both;}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .reveal-b3){
  animation:fadeInUp .55s cubic-bezier(.22,.8,.32,1) .20s both;}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .reveal-b4){
  animation:fadeInUp .55s cubic-bezier(.22,.8,.32,1) .30s both;}

/* ── Floating back-to-top button ───────────────────────────────────────── */
.back-to-top-btn{position:fixed;bottom:28px;right:28px;width:46px;height:46px;
  border-radius:50%;background:linear-gradient(135deg,#3A4EFB,#33A4FA);
  color:#fff!important;display:flex;align-items:center;justify-content:center;
  font-size:25px;font-weight:900;text-decoration:none!important;
  box-shadow:0 8px 24px rgba(58,78,251,0.35);z-index:9998;
  transition:transform .15s ease, box-shadow .15s ease;}
.back-to-top-btn:hover{transform:translateY(-3px);
  box-shadow:0 12px 30px rgba(58,78,251,0.45);}

@media(max-width:768px){
  .block-container{padding:0 20px!important;}
  .edu-nav{margin:0 -20px;padding:0 20px;}.edu-navlinks{display:none;}
  .edu-footer{margin:56px -20px 0;padding:32px 20px 24px;}
  .page-wrap{padding:28px 0 8px;}
  .page-h1{font-size:33px;}
  .kpi-num{font-size:31px;}
  .interv-grid{grid-template-columns:1fr;}
  .back-to-top-btn{bottom:18px;right:18px;width:42px;height:42px;font-size:23px;}
}
</style>
""", unsafe_allow_html=True)

# ── NAVBAR ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div id="page-top"></div>
<nav class="edu-nav">
  <a class="edu-brand" href="/">
    <div class="edu-brand-icon">🛡️</div>
    <div>
      <div class="edu-brand-name">EduGuard</div>
      <div class="edu-brand-sub">Early Dropout Intelligence</div>
    </div>
  </a>
  <div class="edu-navlinks">
    <a href="/">Home</a>
    <a href="/app/pages/predictor.py">Student Predictor</a>
    <a href="/batch" class="active">Batch Dashboard</a>
    <a href="/performance">Model Performance</a>
    <a href="/about">About</a>
  </div>
  <div class="edu-live"><span class="edu-live-dot"></span>Live</div>
</nav>
<a href="#page-top" class="back-to-top-btn" title="Back to top">↑</a>
""", unsafe_allow_html=True)

# ── LOAD ARTEFACTS ─────────────────────────────────────────────────────────────
A            = load_artefacts()
model        = A["model"]
scaler       = A["scaler"]
feature_cols = A["feature_cols"]
cont_cols    = A["cont_cols"]
explainer    = A["explainer"]
display_names= A["display_names"]
expected_val = A["expected_val"]

# ══════════════════════════════════════════════════════════════════════
# GENERATE 50-STUDENT SS2 SAMPLE CSV
# ══════════════════════════════════════════════════════════════════════
@st.cache_data
def generate_ss2_sample():
    np.random.seed(99)
    n = 50

    genders      = np.random.choice(["Female","Male"], n, p=[0.52,0.48])
    ages         = np.random.choice(range(15,19), n,
                       p=[0.30,0.35,0.25,0.10])
    distances    = np.round(np.where(
                       np.random.rand(n) < 0.40,
                       np.random.exponential(8,n)+5,
                       np.random.exponential(2,n)+0.5), 1)
    distances    = np.clip(distances, 0.5, 35.0)
    incomes      = np.random.choice(["Low","Medium","High"], n,
                       p=[0.55,0.30,0.15])
    fee_map      = {"Low":["Unpaid","Partial","Paid"],
                    "Medium":["Unpaid","Partial","Paid"],
                    "High":["Unpaid","Partial","Paid"]}
    fee_prob     = {"Low":[0.45,0.35,0.20],
                    "Medium":[0.10,0.35,0.55],
                    "High":[0.02,0.10,0.88]}
    fees         = np.array([np.random.choice(
                       fee_map[i], p=fee_prob[i]) for i in incomes])
    parent_edus  = np.random.choice(
                       ["No Formal Education","Primary","Secondary",
                        "OND/NCE","HND/BSc","Postgraduate"], n,
                       p=[0.12,0.22,0.30,0.18,0.13,0.05])
    inv_prob     = {"No Formal Education":[0.55,0.30,0.15],
                    "Primary":[0.38,0.38,0.24],
                    "Secondary":[0.22,0.42,0.36],
                    "OND/NCE":[0.12,0.38,0.50],
                    "HND/BSc":[0.06,0.30,0.64],
                    "Postgraduate":[0.03,0.22,0.75]}
    involvements = np.array([np.random.choice(
                       ["Absent","Passive","Active"],
                       p=inv_prob[e]) for e in parent_edus])

    base_att = np.where(distances>10,
                   np.random.normal(62,15,n),
                   np.random.normal(80,12,n))
    fee_pen  = np.where(fees=="Unpaid",-15,
               np.where(fees=="Partial",-7,0))
    attendance = np.round(np.clip(base_att+fee_pen, 0, 100), 1)

    base_score = (0.45*attendance
                  + np.where(incomes=="High",25,
                    np.where(incomes=="Medium",15,5))
                  + np.random.normal(0,10,n))
    acad_score = np.round(np.clip(base_score, 5, 100), 1)

    fail_prob    = np.clip(1-(acad_score/100)*1.3, 0.02, 0.95)
    failed_subj  = np.array([np.random.binomial(8,p) for p in fail_prob])
    failed_subj  = np.clip(failed_subj, 0, 8)

    cog_score    = np.round(np.clip(
                       acad_score*0.70+np.random.normal(0,12,n)+15,
                       5, 100), 1)
    pace_choices = []
    for s in acad_score:
        if s >= 70:
            pace_choices.append(np.random.choice(
                ["Slow","Average","Fast"], p=[0.05,0.30,0.65]))
        elif s >= 50:
            pace_choices.append(np.random.choice(
                ["Slow","Average","Fast"], p=[0.20,0.55,0.25]))
        else:
            pace_choices.append(np.random.choice(
                ["Slow","Average","Fast"], p=[0.55,0.35,0.10]))
    learning_pace = np.array(pace_choices)

    study_base = np.where(involvements=="Active",
                     np.random.normal(72,12,n),
                 np.where(involvements=="Passive",
                     np.random.normal(55,14,n),
                     np.random.normal(38,15,n)))
    study_habit = np.round(np.clip(study_base, 0, 100), 1)

    practical   = np.round(np.clip(
                      np.random.normal(58+(acad_score-50)*0.25, 14, n),
                      0, 100), 1)

    pe_idx      = np.where(attendance>=75, 0,
                  np.where(attendance>=50, 1, 2))
    pe_choices_map = [
        [0.65,0.25,0.10],
        [0.30,0.45,0.25],
        [0.10,0.30,0.60],
    ]
    pe_part     = np.array([np.random.choice(
                      ["Active","Passive","Absent"],
                      p=pe_choices_map[i]) for i in pe_idx])

    psycho_base = np.where(pe_part=="Active",
                      np.random.normal(72,12,n),
                  np.where(pe_part=="Passive",
                      np.random.normal(55,13,n),
                      np.random.normal(38,14,n)))
    psycho      = np.round(np.clip(
                      psycho_base+np.random.normal(0,6,n), 0, 100), 1)

    df = pd.DataFrame({
        "Student_ID":              [f"SS2/EGA/{i+1:03d}" for i in range(n)],
        "Gender":                  genders,
        "Age":                     ages,
        "Year_Group":              ["SS2"] * n,
        "Distance_to_School_km":   distances,
        "Household_Income_Band":   incomes,
        "Fee_Payment_Status":      fees,
        "Parental_Education_Level":parent_edus,
        "Parental_Involvement":    involvements,
        "Attendance_Rate_Pct":     attendance,
        "Avg_Academic_Score_Pct":  acad_score,
        "Failed_Subjects_Count":   failed_subj,
        "Cognitive_Assessment_Score": cog_score,
        "Learning_Pace":           learning_pace,
        "Study_Habit_Score":       study_habit,
        "Practical_Subject_Score": practical,
        "PE_Participation":        pe_part,
        "Psychomotor_Dev_Score":   psycho,
    })
    return df

sample_df = generate_ss2_sample()

# ══════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="page-wrap">
  <div class="page-header">
    <p class="page-eyebrow">Class Assessment</p>
    <h1 class="page-h1">Batch Risk Dashboard</h1>
    <p class="page-sub">
      Upload a CSV of student records. EduGuard assesses every student
      instantly, returning a ranked risk table, cohort charts, and a
      per-student drill-down with full SHAP explanation and intervention plan.
    </p>
    <div class="context-badges">
      <div class="context-badge">📄 CSV upload</div>
      <div class="context-badge">🧑‍🎓 Any cohort size</div>
      <div class="context-badge">⚡ Instant scoring</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# COLUMN REFERENCE — compact, icon-based, grouped by domain (replaces the
# old CSV Format Guide expander + markdown table)
# ══════════════════════════════════════════════════════════════════════
with padded():
    colref_domains = [
        (BLUE1, "👤", "Demographic", [
            ("Student_ID", "Any text — e.g. SS2/EGA/001"),
            ("Gender", "Male / Female"),
            ("Age", "11 – 20"),
            ("Year_Group", "JSS1 – SS3"),
            ("Distance_to_School_km", "0.5 – 35.0"),
        ]),
        (PURPLE, "💰", "Socioeconomic", [
            ("Household_Income_Band", "Low / Medium / High"),
            ("Fee_Payment_Status", "Unpaid / Partial / Paid"),
            ("Parental_Education_Level", "No Formal Education – Postgraduate"),
            ("Parental_Involvement", "Absent / Passive / Active"),
        ]),
        (BLUE2, "📚", "Academic", [
            ("Attendance_Rate_Pct", "0 – 100"),
            ("Avg_Academic_Score_Pct", "0 – 100"),
            ("Failed_Subjects_Count", "0 – 8"),
        ]),
        (AMBER, "🧠", "Cognitive", [
            ("Cognitive_Assessment_Score", "0 – 100"),
            ("Learning_Pace", "Slow / Average / Fast"),
            ("Study_Habit_Score", "0 – 100"),
        ]),
        (GREEN, "🏃", "Psychomotor", [
            ("Practical_Subject_Score", "0 – 100"),
            ("PE_Participation", "Absent / Passive / Active"),
            ("Psychomotor_Dev_Score", "0 – 100"),
        ]),
    ]
    domain_blocks = []
    for clr, icon, label, cols in colref_domains:
        chips = "".join(
            f"<span class='colref-chip' title='{accepted}'>{name}</span>"
            for name, accepted in cols
        )
        domain_blocks.append(
            f"<div class='colref-domain'>"
            f"<div class='colref-domain-label' style='color:{clr};'>{icon} {label}</div>"
            f"<div class='colref-chips'>{chips}</div>"
            f"</div>"
        )
    with st.container(border=True):
        st.markdown(
            f"<div class='domain-card-head'>"
            f"<div class='domain-card-icon' style='background:{BLUE1}18;'>🗂️</div>"
            f"<div class='domain-card-title' style='color:{BLUE1};'>Column Reference</div>"
            f"</div>"
            f"<p class='section-caption' style='margin-bottom:16px;'>"
            f"17 columns across 5 domains · hover a column for its accepted "
            f"values · download the sample CSV below for exact formatting.</p>"
            f"{''.join(domain_blocks)}",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    dl_col, up_col = st.columns([1, 2])

    with dl_col:
        with st.container(border=True):
            st.markdown(
                f"<div class='domain-card-head'>"
                f"<span class='step-num-badge' style='background:{BLUE1};'>1</span>"
                f"<span class='domain-card-title' style='color:{BLUE1};margin-left:8px;'>"
                f"Download Template</span></div>",
                unsafe_allow_html=True)
            st.download_button(
                "⬇️  Download SS2 Sample CSV (50 students)",
                data=sample_df.to_csv(index=False),
                file_name="EduGuard_SS2_Sample.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(
                "50 SS2 students · realistic risk mix · "
                "generated using UNESCO UIS & World Bank parameters"
            )

    with up_col:
        with st.container(border=True):
            st.markdown(
                f"<span class='upload-zone-marker'></span>"
                f"<div class='domain-card-head'>"
                f"<span class='step-num-badge' style='background:{BLUE2};'>2</span>"
                f"<span class='domain-card-title' style='color:{BLUE2};margin-left:8px;'>"
                f"Upload Your Student CSV</span></div>",
                unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Drag and drop your CSV here, or click to browse",
                type=["csv"],
                label_visibility="collapsed",
            )
            st.caption("CSV only · one row per student · matches the Column Reference above")


# ══════════════════════════════════════════════════════════════════════
# PROCESS BATCH
# ══════════════════════════════════════════════════════════════════════
if uploaded is not None:
    with st.status("Processing uploaded cohort...", expanded=True) as batch_status:
        st.write("📥 Reading CSV file...")
        time.sleep(0.25)
        batch_df    = pd.read_csv(uploaded)
        result_rows = []
        raw_rows    = []
        errors      = []

        st.write(f"🤖 Scoring {len(batch_df)} student profile(s) through the XGBoost ensemble...")
        for idx, row in batch_df.iterrows():
            raw = row.to_dict()
            try:
                enc  = encode_input(raw, feature_cols, cont_cols, scaler)
                prob = model.predict_proba(enc)[0][1]
                lv   = ("HIGH"   if prob >= 0.70
                         else "MEDIUM" if prob >= 0.40
                         else "LOW")
                result_rows.append({
                    "Student_ID":   str(raw.get("Student_ID","Unknown")),
                    "Risk_Score_%": round(prob * 100, 1),
                    "Risk_Level":   lv,
                    "Gender":       str(raw.get("Gender","")),
                    "Year_Group":   str(raw.get("Year_Group","")),
                    "Attendance_%": raw.get("Attendance_Rate_Pct",""),
                    "Academic_%":   raw.get("Avg_Academic_Score_Pct",""),
                    "Fee_Status":   str(raw.get("Fee_Payment_Status","")),
                    "Income_Band":  str(raw.get("Household_Income_Band","")),
                    "Failed_Subj":  raw.get("Failed_Subjects_Count",""),
                })
                raw_rows.append(raw)
            except Exception as e:
                errors.append(f"Row {idx+1} ({raw.get('Student_ID','?')}): {str(e)[:60]}")
        time.sleep(0.25)

        st.write("📊 Building cohort charts and rankings...")
        time.sleep(0.25)
        batch_status.update(
            label=(f"Processed {len(result_rows)} of {len(batch_df)} students"
                   + (f" — {len(errors)} skipped" if errors else "")),
            state="complete", expanded=False,
        )

    if errors:
        with st.expander(f"⚠️ {len(errors)} rows skipped — click to see details"):
            for e in errors:
                st.caption(e)

    if not result_rows:
        st.error("No valid rows could be processed. Check your CSV matches the template.")
        st.stop()

    results_df = pd.DataFrame(result_rows).sort_values(
        "Risk_Score_%", ascending=False
    ).reset_index(drop=True)

    n  = len(results_df)
    nh = int((results_df["Risk_Level"] == "HIGH").sum())
    nm = int((results_df["Risk_Level"] == "MEDIUM").sum())
    nl = int((results_df["Risk_Level"] == "LOW").sum())
    avg_score = results_df["Risk_Score_%"].mean()

    with padded():
        st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)

        # ── ALERT BANNER ───────────────────────────────────────────────
        with st.container():
            if nh > 0:
                st.markdown(
                    f"<span class='reveal-a1'></span>"
                    f"<div class='alert-card is-danger'>"
                    f"<div class='alert-icon-circle'>🚨</div>"
                    f"<div>"
                    f"<p class='alert-title'>{nh} student{'s' if nh>1 else ''} flagged as HIGH RISK</p>"
                    f"<p class='alert-body'>"
                    f"Immediate action is required. Review the ranked table below, "
                    f"use the drill-down tool to get intervention plans for each "
                    f"high-risk student, and alert the school counsellor and VP Academics."
                    f"</p></div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<span class='reveal-a1'></span>"
                    f"<div class='alert-card is-ok'>"
                    f"<div class='alert-icon-circle'>✅</div>"
                    f"<div>"
                    f"<p class='alert-title'>No HIGH RISK students detected</p>"
                    f"<p class='alert-body'>"
                    f"All students fall within the Medium or Low risk range. "
                    f"Continue standard monitoring and review Medium risk students."
                    f"</p></div></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

        # ── COHORT KPI ROW ───────────────────────────────────────────────
        st.markdown(
            "<h2 class='section-h2'>Cohort Overview</h2>"
            "<p class='section-caption'>Where the whole uploaded class stands, at a glance.</p>",
            unsafe_allow_html=True)

        with st.container():
            st.markdown("<span class='reveal-a2'></span>", unsafe_allow_html=True)
            k1, k2, k3, k4, k5 = st.columns(5)
            with k1:
                st.markdown(
                    f"<div class='kpi-card'>"
                    f"<div class='kpi-icon' style='background:{BLUE1}18;color:{BLUE1};'>🎓</div>"
                    f"<div class='kpi-num' style='color:{DARK};'>{n}</div>"
                    f"<div class='kpi-label'>Total Students</div>"
                    f"<div class='kpi-sub'>Full cohort uploaded</div></div>",
                    unsafe_allow_html=True)
            with k2:
                st.markdown(
                    f"<div class='kpi-card'>"
                    f"<div class='kpi-icon' style='background:{RED}18;color:{RED};'>🚨</div>"
                    f"<div class='kpi-num' style='color:{RED};'>{nh}</div>"
                    f"<div class='kpi-label'>High Risk</div>"
                    f"<div class='kpi-sub'>{(nh/n*100 if n else 0):.1f}% of cohort</div></div>",
                    unsafe_allow_html=True)
            with k3:
                st.markdown(
                    f"<div class='kpi-card'>"
                    f"<div class='kpi-icon' style='background:{AMBER}18;color:{AMBER};'>⚠️</div>"
                    f"<div class='kpi-num' style='color:{AMBER};'>{nm}</div>"
                    f"<div class='kpi-label'>Medium Risk</div>"
                    f"<div class='kpi-sub'>{(nm/n*100 if n else 0):.1f}% of cohort</div></div>",
                    unsafe_allow_html=True)
            with k4:
                st.markdown(
                    f"<div class='kpi-card'>"
                    f"<div class='kpi-icon' style='background:{GREEN}18;color:{GREEN};'>✅</div>"
                    f"<div class='kpi-num' style='color:{GREEN};'>{nl}</div>"
                    f"<div class='kpi-label'>Low Risk</div>"
                    f"<div class='kpi-sub'>{(nl/n*100 if n else 0):.1f}% of cohort</div></div>",
                    unsafe_allow_html=True)
            with k5:
                st.markdown(
                    f"<div class='kpi-card'>"
                    f"<div class='kpi-icon' style='background:{PURPLE}18;color:{PURPLE};'>📊</div>"
                    f"<div class='kpi-num' style='color:{PURPLE};'>{avg_score:.1f}%</div>"
                    f"<div class='kpi-label'>Avg Risk Score</div>"
                    f"<div class='kpi-sub'>Across all students</div></div>",
                    unsafe_allow_html=True)

        st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)

        # ── CHARTS ─────────────────────────────────────────────────────
        ch1, ch2, ch3 = st.columns(3)

        with ch1:
            with st.container(border=True):
                st.markdown(
                    f"<span class='reveal-a3'></span>"
                    f"<div class='domain-card-head'>"
                    f"<div class='domain-card-icon' style='background:{RED}18;'>🥧</div>"
                    f"<div class='domain-card-title' style='color:{RED};'>Risk Level Split</div>"
                    f"</div>", unsafe_allow_html=True)
                fig_pie = go.Figure(go.Pie(
                    labels=["High Risk","Medium Risk","Low Risk"],
                    values=[nh, nm, nl],
                    marker_colors=[RED, AMBER, GREEN],
                    hole=0.58,
                    textinfo="label+percent",
                    textfont_size=13,
                    hovertemplate="%{label}: %{value} students<extra></extra>",
                ))
                fig_pie.update_layout(
                    height=280,
                    margin=dict(l=10,r=10,t=10,b=10),
                    showlegend=False,
                    paper_bgcolor=WHITE,
                    font=dict(family="Inter", size=13),
                )
                st.plotly_chart(fig_pie, use_container_width=True,
                                 config={"displayModeBar": False})

        with ch2:
            with st.container(border=True):
                st.markdown(
                    f"<span class='reveal-a4'></span>"
                    f"<div class='domain-card-head'>"
                    f"<div class='domain-card-icon' style='background:{BLUE1}18;'>📈</div>"
                    f"<div class='domain-card-title' style='color:{BLUE1};'>Risk Score Distribution</div>"
                    f"</div>", unsafe_allow_html=True)
                fig_hist = px.histogram(
                    results_df, x="Risk_Score_%",
                    nbins=20,
                    color_discrete_sequence=[BLUE1],
                    labels={"Risk_Score_%": "Dropout Risk Score (%)"},
                )
                fig_hist.add_vline(
                    x=70, line_color=RED, line_dash="dash", line_width=2,
                    annotation_text="High ≥ 70%",
                    annotation_font_color=RED,
                    annotation_position="top right",
                )
                fig_hist.add_vline(
                    x=40, line_color=AMBER, line_dash="dot", line_width=1.5,
                    annotation_text="Medium ≥ 40%",
                    annotation_font_color=AMBER,
                    annotation_position="top right",
                )
                fig_hist.update_layout(
                    height=280,
                    margin=dict(l=10,r=10,t=10,b=10),
                    plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                    font=dict(family="Inter", size=13),
                )
                st.plotly_chart(fig_hist, use_container_width=True,
                                 config={"displayModeBar": False})

        with ch3:
            with st.container(border=True):
                st.markdown(
                    f"<span class='reveal-a5'></span>"
                    f"<div class='domain-card-head'>"
                    f"<div class='domain-card-icon' style='background:{PURPLE}18;'>🏫</div>"
                    f"<div class='domain-card-title' style='color:{PURPLE};'>Avg Risk by Year Group</div>"
                    f"</div>", unsafe_allow_html=True)
                yg_risk = (
                    results_df.groupby("Year_Group")["Risk_Score_%"]
                    .mean().reset_index()
                    .sort_values("Risk_Score_%", ascending=False)
                )
                fig_yg = px.bar(
                    yg_risk, x="Year_Group", y="Risk_Score_%",
                    color="Risk_Score_%",
                    color_continuous_scale=[[0,"#00C48C"],[0.5,"#FFB800"],[1,"#FF4B4B"]],
                    labels={"Risk_Score_%": "Avg Risk Score (%)",
                            "Year_Group": "Year Group"},
                    text="Risk_Score_%",
                )
                fig_yg.update_traces(
                    texttemplate="%{text:.1f}%",
                    textposition="outside",
                )
                fig_yg.update_layout(
                    height=280,
                    margin=dict(l=10,r=10,t=10,b=10),
                    plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                    coloraxis_showscale=False,
                    font=dict(family="Inter", size=13),
                )
                st.plotly_chart(fig_yg, use_container_width=True,
                                 config={"displayModeBar": False})

        st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)

        # ── RANKED TABLE ───────────────────────────────────────────────
        with st.container(border=True):
            st.markdown(
                f"<span class='reveal-a6'></span>"
                f"<div class='domain-card-head'>"
                f"<div class='domain-card-icon' style='background:{BLUE2}18;'>🗂️</div>"
                f"<div class='domain-card-title' style='color:{BLUE2};'>"
                f"Students Ranked by Dropout Risk</div></div>",
                unsafe_allow_html=True)

            f1, f2, f3 = st.columns(3)
            with f1:
                fl = st.multiselect(
                    "Risk Level", ["HIGH","MEDIUM","LOW"],
                    default=["HIGH","MEDIUM","LOW"],
                )
            with f2:
                yg_opts = sorted(results_df["Year_Group"].unique().tolist())
                fy = st.multiselect("Year Group", yg_opts, default=yg_opts)
            with f3:
                gn_opts = sorted(results_df["Gender"].unique().tolist())
                fg = st.multiselect("Gender", gn_opts, default=gn_opts)

            filtered = results_df[
                results_df["Risk_Level"].isin(fl) &
                results_df["Year_Group"].isin(fy) &
                results_df["Gender"].isin(fg)
            ].copy()

            st.caption(
                f"Showing {len(filtered)} of {n} students · "
                f"sorted by risk score (highest first)"
            )

            def colour_risk(val):
                if val == "HIGH":
                    return "background-color:#FFF0F0;color:#FF4B4B;font-weight:700"
                elif val == "MEDIUM":
                    return "background-color:#FFFBF0;color:#FFB800;font-weight:700"
                return "background-color:#F0FFF9;color:#00C48C;font-weight:700"

            # Attendance_%/Academic_%/Failed_Subj come straight from the
            # uploaded CSV (raw.get(..., "")), so a missing column can leave
            # an empty string sitting in an otherwise-numeric column — a bare
            # "{:.1f}" format string crashes on that. These guarded
            # formatters fall back to str() instead, the same defensive
            # pattern used for the single-student page's categorical
            # feat_val. Risk_Score_% is always our own round(prob*100,1)
            # float, never a raw passthrough, so it's safe to format directly.
            def _safe_fmt(v, spec):
                return (format(v, spec)
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                        else str(v))

            styled_table = (
                filtered.style
                .map(colour_risk, subset=["Risk_Level"])
                .format({
                    "Risk_Score_%": "{:.1f}".format,
                    "Attendance_%": lambda v: _safe_fmt(v, ".1f"),
                    "Academic_%":   lambda v: _safe_fmt(v, ".1f"),
                    "Failed_Subj":  lambda v: _safe_fmt(v, ".0f"),
                })
            )
            st.dataframe(styled_table, use_container_width=True,
                         height=460, hide_index=True)

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇️  Download Full Risk Report (CSV)",
                    data=results_df.to_csv(index=False),
                    file_name="EduGuard_Full_Risk_Report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl2:
                high_only = results_df[results_df["Risk_Level"] == "HIGH"]
                st.download_button(
                    f"⬇️  Download HIGH RISK Only — {len(high_only)} students",
                    data=high_only.to_csv(index=False),
                    file_name="EduGuard_HighRisk_Students.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary",
                )

        # ══════════════════════════════════════════════════════════════
        # PER-STUDENT DRILL-DOWN
        # ══════════════════════════════════════════════════════════════
        st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
        st.markdown(
            "<h2 class='section-h2'>🔬 Per-Student Drill-Down</h2>"
            "<p class='section-caption'>Select any student from the uploaded "
            "batch to get their full SHAP explanation and ranked intervention "
            "plan — without re-entering data.</p>",
            unsafe_allow_html=True)

        # Build lookup: student_id → raw dict + result row
        id_list   = [r["Student_ID"] for r in result_rows]
        id_to_raw = {r["Student_ID"]: raw_rows[i]
                     for i, r in enumerate(result_rows)}
        id_to_score = {r["Student_ID"]: r["Risk_Score_%"]
                       for r in result_rows}
        id_to_level = {r["Student_ID"]: r["Risk_Level"]
                       for r in result_rows}

        # Sort selector — high risk students first
        high_ids   = [i for i in id_list if id_to_level[i]=="HIGH"]
        medium_ids = [i for i in id_list if id_to_level[i]=="MEDIUM"]
        low_ids    = [i for i in id_list if id_to_level[i]=="LOW"]
        sorted_ids = high_ids + medium_ids + low_ids

        with st.container(border=True):
            st.markdown(
                f"<div class='domain-card-head'>"
                f"<div class='domain-card-icon' style='background:{PURPLE}18;'>🧑‍🎓</div>"
                f"<div class='domain-card-title' style='color:{PURPLE};'>"
                f"Choose a Student</div></div>",
                unsafe_allow_html=True)
            selected_id = st.selectbox(
                "Select student to analyse",
                options=sorted_ids,
                format_func=lambda sid: (
                    f"🔴 {sid}  ({id_to_score[sid]:.1f}%)"
                    if id_to_level[sid]=="HIGH"
                    else f"🟡 {sid}  ({id_to_score[sid]:.1f}%)"
                    if id_to_level[sid]=="MEDIUM"
                    else f"🟢 {sid}  ({id_to_score[sid]:.1f}%)"
                ),
            )
            run_clicked = st.button(
                "🔍  Run Full Analysis for This Student",
                type="primary", use_container_width=True,
            )

        if run_clicked:
            raw_sel = id_to_raw[selected_id]

            with st.status(f"Analysing {selected_id}...", expanded=True) as status_sel:
                st.write("📥 Encoding student profile...")
                time.sleep(0.25)
                enc_sel      = encode_input(raw_sel, feature_cols, cont_cols, scaler)

                st.write("🤖 Running XGBoost ensemble model...")
                time.sleep(0.25)
                prob_sel, shap_sel = predict_student(model, explainer, enc_sel)
                unscaled_sel = get_unscaled(raw_sel, feature_cols)

                st.write("🔍 Computing SHAP attributions...")
                time.sleep(0.2)

                st.write("📋 Ranking interventions...")
                time.sleep(0.2)
                result_sel   = get_interventions(
                    shap_vals=shap_sel,
                    feature_vals=unscaled_sel,
                    feature_names=list(feature_cols),
                    risk_score=prob_sel,
                    display_names=display_names,
                    top_n_drivers=3,
                    max_interventions_per_driver=3,
                )
                status_sel.update(
                    label=f"Analysis complete — {result_sel['risk_level']} risk identified",
                    state="complete", expanded=False,
                )

            level_sel   = result_sel["risk_level"]
            urgent_sel  = sum(1 for i in result_sel["interventions"]
                              if "URGENT" in i["urgency"])
            clr_sel     = RED if level_sel=="HIGH" \
                          else AMBER if level_sel=="MEDIUM" else GREEN
            emoji_sel   = "🔴" if level_sel=="HIGH" \
                          else "🟡" if level_sel=="MEDIUM" else "🟢"

            st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

            # ── Drill-down hero card ────────────────────────────────
            with st.container(border=True):
                st.markdown(
                    f"<span class='reveal-b1'></span>"
                    f"<div style='height:6px;width:100%;border-radius:6px;"
                    f"background:{clr_sel};margin-bottom:20px;'></div>",
                    unsafe_allow_html=True)
                gcol, mcol, ccol = st.columns([1, 2, 1])
                with gcol:
                    st.plotly_chart(risk_gauge_chart(prob_sel, clr_sel),
                                     use_container_width=True,
                                     config={"displayModeBar": False})
                with mcol:
                    st.markdown(
                        f"<div class='result-id'>Full Analysis — {selected_id}</div>"
                        f"<div class='result-level-badge' "
                        f"style='background:{clr_sel}18;color:{clr_sel};border:1px solid {clr_sel}44;'>"
                        f"{emoji_sel} {level_sel} RISK</div>"
                        f"<p class='result-summary'>{result_sel['summary']}</p>",
                        unsafe_allow_html=True)
                with ccol:
                    st.markdown(
                        f"<div class='mini-card'>"
                        f"<div class='mini-card-num' style='color:{RED};'>{urgent_sel}</div>"
                        f"<div class='mini-card-label'>Urgent actions</div></div>"
                        f"<div class='mini-card'>"
                        f"<div class='mini-card-num' style='color:{BLUE1};'>"
                        f"{len(result_sel['interventions'])}</div>"
                        f"<div class='mini-card-label'>Total interventions</div></div>",
                        unsafe_allow_html=True)

            st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

            # ── SHAP explanation card ────────────────────────────────
            with st.container(border=True):
                st.markdown(
                    f"<span class='reveal-b2'></span>"
                    f"<div class='domain-card-head'>"
                    f"<div class='domain-card-icon' style='background:{BLUE1}18;'>📊</div>"
                    f"<div class='domain-card-title' style='color:{BLUE1};'>"
                    f"Why This Score</div></div>"
                    f"<p class='section-caption'>How each factor pushed "
                    f"{selected_id}'s prediction up or down, from SHAP analysis. "
                    f"<span style='color:{RED};font-weight:700;'>Red</span> raises "
                    f"risk &nbsp;·&nbsp; <span style='color:{GREEN};font-weight:700;'>"
                    f"Green</span> lowers it.</p>",
                    unsafe_allow_html=True)
                fig_drill = shap_waterfall_chart(
                    shap_sel, unscaled_sel, feature_cols, display_names,
                    expected_val, prob_sel,
                    title=f"SHAP Feature Contributions — {selected_id}",
                )
                fig_drill.update_layout(
                    plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                    font=dict(family="Inter", size=13),
                )
                st.plotly_chart(fig_drill, use_container_width=True,
                                 config={"displayModeBar": False})

            st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

            # ── Top drivers card ──────────────────────────────────────
            with st.container(border=True):
                st.markdown(
                    f"<span class='reveal-b3'></span>"
                    f"<div class='domain-card-head'>"
                    f"<div class='domain-card-icon' style='background:{PURPLE}18;'>🔍</div>"
                    f"<div class='domain-card-title' style='color:{PURPLE};'>"
                    f"Top Risk Drivers</div></div>"
                    f"<p class='section-caption'>The 3 factors with the strongest "
                    f"pull on this prediction.</p>",
                    unsafe_allow_html=True)
                top3_sel = result_sel["top_drivers"][:3]
                max_abs_sel = max((abs(d["shap_val"]) for d in top3_sel), default=0) or 1
                dr1, dr2, dr3 = st.columns(3)
                for i, (col, driver) in enumerate(
                    zip([dr1,dr2,dr3], top3_sel)
                ):
                    sign      = "+" if driver["shap_val"] > 0 else ""
                    dclr      = RED if driver["shap_val"] > 0 else GREEN
                    direction = "↑ increases risk" \
                                if driver["shap_val"] > 0 else "↓ reduces risk"
                    # A top driver can be a categorical feature (e.g.
                    # Fee_Payment_Status = "Unpaid"), not just a continuous
                    # score — guard the numeric format so a string value
                    # never crashes the card.
                    fv = driver["feat_val"]
                    fv_display = f"{fv:.1f}" if isinstance(fv, (int, float)) else str(fv)
                    pct_sel = round(abs(driver["shap_val"]) / max_abs_sel * 100, 1)
                    col.markdown(
                        f"<div class='driver-card' style='animation-delay:{i*0.08}s;'>"
                        f"<div>"
                        f"<div class='driver-direction' style='color:{dclr};'>{direction}</div>"
                        f"<div class='driver-name'>{driver['display']}</div>"
                        f"<div class='driver-shap' style='color:{dclr};'>"
                        f"{sign}{driver['shap_val']:.3f}</div>"
                        f"</div>"
                        f"<div>"
                        f"<div class='driver-bar-track'>"
                        f"<div class='driver-bar-fill' style='width:{pct_sel}%;background:{dclr};'></div>"
                        f"</div>"
                        f"<div class='driver-val'>Student value: {fv_display}</div>"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True)

            st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

            # ── Interventions accordion (grouped by urgency, collapsible) ──
            with st.container(border=True):
                st.markdown(
                    f"<span class='reveal-b4'></span>"
                    f"<div class='domain-card-head'>"
                    f"<div class='domain-card-icon' style='background:{BLUE2}18;'>📋</div>"
                    f"<div class='domain-card-title' style='color:{BLUE2};'>"
                    f"Recommended Interventions</div></div>"
                    f"<p class='section-caption'>Grouped by urgency and linked to "
                    f"dominant risk drivers · Based on Ojo &amp; Awonuga (2022) "
                    f"Nigerian school intervention taxonomy. Click a group to "
                    f"expand it, and a card's <strong>Why this helps</strong> "
                    f"to see the rationale.</p>",
                    unsafe_allow_html=True)
                st.markdown(
                    f"<div class='interv-legend'>"
                    f"<div class='interv-legend-item'><span class='interv-legend-dot' "
                    f"style='background:{RED};'></span>Urgent — act this week</div>"
                    f"<div class='interv-legend-item'><span class='interv-legend-dot' "
                    f"style='background:{AMBER};'></span>Medium — act this month</div>"
                    f"<div class='interv-legend-item'><span class='interv-legend-dot' "
                    f"style='background:{GREEN};'></span>Monitor — keep an eye on it</div>"
                    f"</div>",
                    unsafe_allow_html=True)

                # Group interventions by urgency band so the page doesn't
                # dump every action at once — each band is its own
                # collapsible accordion section, open by default only for
                # the highest urgency band that actually has items in it.
                band_order = ["URGENT", "MEDIUM", "MONITOR"]
                band_labels = {"URGENT": "Urgent", "MEDIUM": "Medium priority", "MONITOR": "Monitor"}
                band_icons = {"URGENT": "🔴", "MEDIUM": "🟡", "MONITOR": "🟢"}
                band_colors = {"URGENT": RED, "MEDIUM": AMBER, "MONITOR": GREEN}
                band_css = {"URGENT": "interv-urgent", "MEDIUM": "interv-medium", "MONITOR": "interv-monitor"}

                grouped_sel = {b: [] for b in band_order}
                for interv in result_sel["interventions"]:
                    for b in band_order:
                        if b in interv["urgency"]:
                            grouped_sel[b].append(interv)
                            break

                first_open_sel = next((b for b in band_order if grouped_sel[b]), None)

                group_blocks = []
                for b in band_order:
                    items = grouped_sel[b]
                    if not items:
                        continue
                    gclr = band_colors[b]
                    cards = []
                    for i, interv in enumerate(items):
                        cards.append(
                            f"<div class='{band_css[b]}' style='animation-delay:{i*0.05}s;'>"
                            f"<div class='interv-head'>"
                            f"<span class='driver-chip'>📌 {interv['driver']}</span>"
                            f"<span class='interv-party'>→ {interv['party']}</span>"
                            f"</div>"
                            f"<p class='interv-action'>{interv['action']}</p>"
                            f"<details class='rationale-toggle'>"
                            f"<summary>Why this helps</summary>"
                            f"<div class='rationale-toggle-body'>📖 {interv['rationale']}</div>"
                            f"</details>"
                            f"</div>"
                        )
                    cards_html = "".join(cards)
                    open_attr = "open" if b == first_open_sel else ""
                    group_blocks.append(
                        f"<details class='interv-group' {open_attr}>"
                        f"<summary class='interv-group-summary' style='color:{gclr};'>"
                        f"<span>{band_icons[b]} {band_labels[b]}</span>"
                        f"<span class='interv-group-badge' style='background:{gclr}18;color:{gclr};'>"
                        f"{len(items)}</span>"
                        f"</summary>"
                        f"<div class='interv-group-body'><div class='interv-grid'>{cards_html}</div></div>"
                        f"</details>"
                    )

                st.markdown(
                    f"<div class='interv-accordion'>{''.join(group_blocks)}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<div class='tip-banner'>💡 Tip: pick another student above and "
                "click <strong>Run Full Analysis</strong> again to review a "
                "different case.</div>",
                unsafe_allow_html=True)


# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="edu-footer">
  <div class="edu-footer-credit">
    EduGuard — Early Dropout Intelligence System &nbsp;·&nbsp;
  </div>
</div>
""", unsafe_allow_html=True)