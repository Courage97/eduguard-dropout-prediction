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
    load_artefacts, roc_chart, confusion_heatmap,
)

st.set_page_config(
    page_title="EduGuard — Model Performance",
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
BORDER = "#E0E3FF"
CARD   = "#F4F5FF"
PURPLE = "#7B2FBE"
MUTED  = "#6B7280"

MODEL_SLUGS = {
    "Logistic Regression": "logistic_regression",
    "Random Forest":       "random_forest",
    "XGBoost":             "xgboost",
}
MODEL_COLORS = {
    "Logistic Regression": AMBER,
    "Random Forest":       BLUE2,
    "XGBoost":             BLUE1,
}


@contextmanager
def padded(extra_style=""):
    st.markdown(f"<div style='{extra_style}'>", unsafe_allow_html=True)
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)


# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
*,html,body,[class*="css"]{font-family:'Inter',sans-serif!important;box-sizing:border-box;}
[data-testid*="Icon"]{font-family:'Material Symbols Rounded'!important;}
.main{background:#F7F8FF!important;font-size:21px;}
.block-container{padding:0 56px!important;max-width:100%!important;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none;}

[data-testid="stMarkdownContainer"] p{font-size:20px;}
[data-testid="stDataFrame"] *{font-size:20px!important;}
[data-testid="stMultiSelect"] *{font-size:20px!important;}
[data-testid="stWidgetLabel"] p{font-size:20px!important;font-weight:600!important;}
[data-testid="stMultiSelectTagsContainer"] span[data-tag]{
  background:#3A4EFB!important;border-radius:8px!important;}

/* ── Navbar ─────────────────────────────────────────────────────────── */
.edu-nav{position:sticky;top:0;z-index:9999;background:rgba(37,41,67,0.92);
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  margin:0 -56px;
  padding:0 56px;display:flex;align-items:center;justify-content:space-between;
  min-height:72px;border-bottom:1px solid rgba(255,255,255,0.07);}
.edu-brand{display:flex;align-items:center;gap:13px;text-decoration:none!important;}
.edu-brand-icon{background:linear-gradient(135deg,#3A4EFB,#33A4FA);width:42px;height:42px;
  border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:26px;
  box-shadow:0 6px 16px rgba(58,78,251,0.35);}
.edu-brand-name{color:#fff;font-size:27px;font-weight:900;letter-spacing:-0.5px;line-height:1.1;}
.edu-brand-sub{color:#E3FF3B;font-size:8px;font-weight:700;letter-spacing:2px;
  text-transform:uppercase;margin-top:2px;}
.edu-navlinks{display:flex;align-items:center;gap:2px;}
.edu-navlinks a{color:rgba(222,224,237,0.72)!important;font-size:17px;font-weight:500;
  padding:9px 18px;border-radius:8px;text-decoration:none!important;transition:color .2s,background .2s;}
.edu-navlinks a:hover{color:#fff!important;background:rgba(255,255,255,0.07);}
.edu-navlinks a.active{color:#E3FF3B!important;background:rgba(227,255,59,0.1);}
.edu-live{display:inline-flex;align-items:center;gap:7px;background:rgba(227,255,59,0.12);
  border:1px solid rgba(227,255,59,0.4);color:#E3FF3B;padding:7px 16px;border-radius:24px;
  font-size:12px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;}
.edu-live-dot{width:6px;height:6px;border-radius:50%;background:#E3FF3B;
  animation:eduPulse 1.8s ease-in-out infinite;}
@keyframes eduPulse{0%,100%{opacity:1;}50%{opacity:0.35;}}

/* ── Page header ────────────────────────────────────────────────────── */
.page-wrap{padding:40px 0 8px;}
.page-eyebrow{font-size:18px;font-weight:700;letter-spacing:2.5px;
  text-transform:uppercase;color:#3A4EFB;margin:0 0 10px;}
.page-h1{font-size:40px;font-weight:900;color:#252943!important;letter-spacing:-1px;
  margin:0 0 10px!important;padding:0!important;line-height:1.15;}
.page-sub{font-size:15px;color:#6B7280; text-align:justify;margin:0 0 18px;line-height:1.75;max-width:600px;}
.context-badges{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px;padding:10px 20px;}
.context-badge{background:#fff;border:2px solid #E0E3FF;color:#374151;
  padding:9px 15px;border-radius:20px;font-size:14.5px;font-weight:900;}

/* ── Shared card header ─────────────────────────────────────────────── */
.domain-card-head{display:flex;align-items:center;gap:10px;margin-bottom:14px;
  padding:12px;border-bottom:1px solid #EEF0FA;}
.domain-card-icon{width:34px;height:34px;border-radius:10px;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;font-size:22px;}
.domain-card-title{font-size:18px;font-weight:800;letter-spacing:0.1px;}

.section-divider{height:1px;background:#E0E3FF;margin:24px;border:none;}
.section-h2{font-size:20px; padding:20px font-weight:900;color:#252943!important;
  letter-spacing:-0.4px;margin:0 0 6px!important;padding:0!important;}
.section-sub{font-size:16.5px;color:#6B7280;margin:0 0 18px;line-height:1.65;}

/* ── KPI strip ─────────────────────────────────────────────────────── */
.kpi-card{background:#fff;border:1px solid #E0E3FF;border-radius:16px;
  padding:20px 22px;box-shadow:0 4px 18px rgba(58,78,251,0.07);
  transition:transform .2s ease, box-shadow .2s ease;height:100%;}
.kpi-card:hover{transform:translateY(-4px);box-shadow:0 14px 28px rgba(58,78,251,0.13); padding:20px 22px;}
.kpi-icon{width:38px;height:38px;border-radius:11px;display:flex;align-items:center;
  justify-content:center;font-size:20px;margin-bottom:14px;}
.kpi-num{font-size:28px;font-weight:900;letter-spacing:-1px;line-height:1;}
.kpi-label{font-size:15.5px;color:#6B7280;font-weight:700;margin-top:7px;
  text-transform:uppercase;letter-spacing:0.4px;}
.kpi-sub{font-size:15.5px;color:#9CA3AF;margin-top:4px;}

/* ── Model overview cards ──────────────────────────────────────────── */
.model-card{background:#fff;border:1px solid #E0E3FF;border-radius:16px;
  padding:0 24px 22px;box-shadow:0 4px 18px rgba(58,78,251,0.07);
  transition:transform .2s ease, box-shadow .2s ease;height:100%;overflow:hidden;}
.model-card:hover{transform:translateY(-4px);box-shadow:0 16px 32px rgba(58,78,251,0.14);}
.model-card-accent{height:6px;margin:0 -24px 22px;}
.model-card-role{font-size:17.5px;font-weight:700;letter-spacing:1.3px;
  text-transform:uppercase;margin-bottom:4px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.model-card-name{font-size:28px;font-weight:900;color:#252943;margin:4px 0 12px;
  letter-spacing:-0.5px;}
.model-card-desc{font-size:19px;color:#6B7280;line-height:1.65;margin:0;}
.metric-pill{display:inline-block;padding:5px 13px;border-radius:20px;
  font-size:17.5px;font-weight:700;margin:3px 6px 0 0;}
.winner-badge{background:linear-gradient(90deg,#3A4EFB,#33A4FA);
  color:#fff;padding:4px 13px;border-radius:20px;font-size:16.5px;
  font-weight:800;letter-spacing:0.3px;display:inline-block;
  box-shadow:0 4px 12px rgba(58,78,251,0.3);}

/* ── Insight / warning callouts ────────────────────────────────────── */
.insight-card{background:#F4F5FF;border:1px solid #E0E3FF;border-left:4px solid #3A4EFB;
  border-radius:14px;padding:20px 24px;margin:16px 0;}
.insight-card-title{font-size:18px;font-weight:800;color:#3A4EFB;letter-spacing:0.5px;
  text-transform:uppercase;margin:0 0 10px;}
.insight-card-body{font-size:19.5px;color:#374151;line-height:1.75;margin:0;}
.insight-card-body strong{color:#252943;}
.cv-note{background:#FFFBF0;border:1px solid #FFE7A8;border-left:4px solid #FFB800;
  border-radius:14px;padding:18px 24px;font-size:19.5px;color:#374151;
  line-height:1.7;margin:18px 0 0;}
.empty-note{background:#F7F8FF;border:1px dashed #C7CCF5;border-radius:14px;
  padding:18px 24px;font-size:19.5px;color:#6B7280;text-align:center;}

.impact-chip{padding:9px 16px;border-radius:10px;font-size:18px;font-weight:700;
  text-align:center;}

/* ── "Why XGBoost" pull-quote cards ─────────────────────────────────── */
.quote-card{background:#252943;border-radius:18px;padding:32px 30px;
  position:relative;overflow:hidden;height:calc(100% - 16px);}
.quote-card::before{content:"\\201C";position:absolute;top:-22px;left:16px;
  font-size:125px;color:rgba(255,255,255,0.07);font-weight:900;line-height:1;
  font-family:Georgia,serif;}
.quote-label{font-size:17px;font-weight:800;letter-spacing:2px;text-transform:uppercase;
  color:#E3FF3B;margin:0 0 14px;position:relative;z-index:1;}
.quote-text{font-size:24px;font-weight:700;color:#fff;line-height:1.5;margin:0;
  position:relative;z-index:1;letter-spacing:-0.2px;}

/* ── Top-level report tabs (bigger, pill-style, act as page navigation) ── */
[data-testid="stTabs"] [role="tablist"]{background:#F4F5FF;padding:6px;border-radius:12px;
  gap:4px!important;margin-bottom:4px;flex-wrap:wrap;}
[data-testid="stTabs"] [data-testid="stTab"]{background:#fff;border-radius:8px!important;
  padding:10px 20px!important;font-weight:700;font-size:19.5px;color:#252943;
  border:1px solid #E0E3FF!important;cursor:pointer;}
[data-testid="stTabs"] [data-testid="stTab"] p{color:inherit!important;font-weight:inherit!important;}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"]{background:#252943!important;
  color:#E3FF3B!important;border-color:#252943!important;}
[data-testid="stTabs"] .react-aria-SelectionIndicator{display:none!important;}

/* ── Confusion matrices row (side by side, no tabs, dynamic column count) ── */
.cm-label{font-size:18px;font-weight:800;color:#252943;text-align:center;
  margin-bottom:6px;}

/* ── Filter bar for interactive model comparison ───────────────────── */
.filter-bar{background:#fff;border:1px solid #E0E3FF;border-radius:14px;
  padding:14px 18px 4px;margin-bottom:18px;display:flex;align-items:flex-start;
  gap:14px;flex-wrap:wrap;}
.filter-bar-label{font-size:18px;font-weight:800;color:#3A4EFB;letter-spacing:0.3px;
  text-transform:uppercase;padding-top:8px;white-space:nowrap;}

/* ── Footer ─────────────────────────────────────────────────────────── */
.edu-footer{background:#252943;padding:32px 56px 24px;margin:56px -56px 0;}
.edu-footer-credit{color:#7A7F9A;font-size:18px;text-align:center;line-height:1.6;}

/* ── Section reveal on load (marker + :has(), same trick as the other
   EduGuard pages) — ten possible slots across the five tabs. ───────── */
@keyframes fadeInUp{from{opacity:0;transform:translateY(16px);}
  to{opacity:1;transform:translateY(0);}}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .reveal-1){
  animation:fadeInUp .5s cubic-bezier(.22,.8,.32,1) both;}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .reveal-2){
  animation:fadeInUp .5s cubic-bezier(.22,.8,.32,1) .06s both;}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .reveal-3){
  animation:fadeInUp .5s cubic-bezier(.22,.8,.32,1) .12s both;}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .reveal-4){
  animation:fadeInUp .5s cubic-bezier(.22,.8,.32,1) .18s both;}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .reveal-5){
  animation:fadeInUp .5s cubic-bezier(.22,.8,.32,1) .24s both;}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .reveal-6){
  animation:fadeInUp .5s cubic-bezier(.22,.8,.32,1) .30s both;}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .reveal-7){
  animation:fadeInUp .5s cubic-bezier(.22,.8,.32,1) .36s both;}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .reveal-8){
  animation:fadeInUp .5s cubic-bezier(.22,.8,.32,1) .42s both;}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .reveal-9){
  animation:fadeInUp .5s cubic-bezier(.22,.8,.32,1) .48s both;}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .reveal-10){
  animation:fadeInUp .5s cubic-bezier(.22,.8,.32,1) .54s both;}

/* Plotly chart container wipes in left-to-right on the Feature Importance
   tab, as the nearest CSS-only equivalent of individual bars growing in —
   Plotly's bars are SVG paths owned by the embedded chart library, so CSS
   can't animate them individually the way it can a plain div. */
@keyframes barsWipeIn{from{clip-path:inset(0 100% 0 0);}to{clip-path:inset(0 0 0 0);}}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .reveal-8)
  [data-testid="stPlotlyChart"]{animation:barsWipeIn .9s cubic-bezier(.22,.8,.32,1) .3s both;}

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
  .page-h1{font-size:32px;}
  .kpi-num{font-size:22px;}
  .back-to-top-btn{bottom:18px;right:18px;width:42px;height:42px;font-size:22px;}
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
    <a href="/batch">Batch Dashboard</a>
    <a href="/performance" class="active">Model Performance</a>
    <a href="/about">About</a>
  </div>
  <div class="edu-live"><span class="edu-live-dot"></span>Live</div>
</nav>
<a href="#page-top" class="back-to-top-btn" title="Back to top">↑</a>
""", unsafe_allow_html=True)

# ── LOAD ARTEFACTS ─────────────────────────────────────────────────────────────
A          = load_artefacts()
comparison = A["comparison"]
shap_imp   = A["shap_imp"]
xgb_fi     = A["xgb_fi"]

xgb_row = comparison[comparison["Model"] == "XGBoost"].iloc[0]
lr_row  = comparison[comparison["Model"] == "Logistic Regression"].iloc[0]
rf_row  = comparison[comparison["Model"] == "Random Forest"].iloc[0]
model_rows = {"Logistic Regression": lr_row, "Random Forest": rf_row, "XGBoost": xgb_row}

# Load once, up front, so every tab below can reuse the same predictions
# instead of re-reading the CSVs per section.
preds = {m: pd.read_csv(f"model/{slug}_predictions.csv") for m, slug in MODEL_SLUGS.items()}

from sklearn.metrics import confusion_matrix
cm_data = {}
for m, df in preds.items():
    tn, fp, fn, tp = confusion_matrix(df["y_true"], df["y_pred"]).ravel()
    cm_data[m] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

best_auc_model = max(model_rows, key=lambda m: model_rows[m]["Roc_auc"])
best_recall_model = max(model_rows, key=lambda m: model_rows[m]["Recall"])
test_n = len(preds["XGBoost"])

# ══════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="page-wrap">
  <p class="page-eyebrow">Technical Evaluation</p>
  <h1 class="page-h1">Model Performance Dashboard</h1>
  <p class="page-sub">
    Full evaluation of all three trained models on the held-out test set
    of 2,000 students. Switch between tabs to explore ROC curves, confusion
    matrices, feature importance, and cross-validation everything needed
    to defend the model selection at the viva.
  </p>
  <div class="context-badges">
    <div class="context-badge">🧪 3 models compared</div>
    <div class="context-badge">🧑‍🎓 2,000-student test set</div>
    <div class="context-badge">🔁 5-fold cross-validated</div>
  </div>
</div>
""", unsafe_allow_html=True)

tab_overview, tab_roc, tab_pr, tab_feat, tab_cv = st.tabs([
    "📊 Overview", "📈 ROC and Confusion Matrix",
    "🎯 Precision-Recall and Calibration", "🔬 Feature Importance",
    "🔁 Validation and Rationale",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════
with tab_overview:
    with padded():
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

        # ── KPI strip ────────────────────────────────────────────────
        with st.container():
            st.markdown("<span class='reveal-1'></span>", unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(
                    f"<div class='kpi-card'>"
                    f"<div class='kpi-icon' style='background:{BLUE1}18;color:{BLUE1};'>🏆</div>"
                    f"<div class='kpi-num' style='color:{BLUE1};'>{model_rows[best_auc_model]['Roc_auc']:.3f}</div>"
                    f"<div class='kpi-label'>Best AUC-ROC</div>"
                    f"<div class='kpi-sub'>{best_auc_model}</div></div>",
                    unsafe_allow_html=True)
            with k2:
                st.markdown(
                    f"<div class='kpi-card'>"
                    f"<div class='kpi-icon' style='background:{RED}18;color:{RED};'>🎯</div>"
                    f"<div class='kpi-num' style='color:{RED};'>{model_rows[best_recall_model]['Recall']:.3f}</div>"
                    f"<div class='kpi-label'>Best Recall</div>"
                    f"<div class='kpi-sub'>{best_recall_model}</div></div>",
                    unsafe_allow_html=True)
            with k3:
                st.markdown(
                    f"<div class='kpi-card'>"
                    f"<div class='kpi-icon' style='background:{PURPLE}18;color:{PURPLE};'>🧪</div>"
                    f"<div class='kpi-num' style='color:{PURPLE};'>3</div>"
                    f"<div class='kpi-label'>Models Compared</div>"
                    f"<div class='kpi-sub'>LR · RF · XGBoost</div></div>",
                    unsafe_allow_html=True)
            with k4:
                st.markdown(
                    f"<div class='kpi-card'>"
                    f"<div class='kpi-icon' style='background:{GREEN}18;color:{GREEN};'>🧑‍🎓</div>"
                    f"<div class='kpi-num' style='color:{GREEN};'>{test_n:,}</div>"
                    f"<div class='kpi-label'>Test Set Size</div>"
                    f"<div class='kpi-sub'>Held out, never trained on</div></div>",
                    unsafe_allow_html=True)

        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)

        # ── Model overview cards ────────────────────────────────────
        with st.container():
            st.markdown("<span class='reveal-2'></span>", unsafe_allow_html=True)
            st.markdown(
                "<h2 class='section-h2'>Three Models Trained and Compared</h2>"
                "<p class='section-sub'>Logistic Regression provides the statistical "
                "baseline. Random Forest and XGBoost are the ensemble methods. "
                "XGBoost was selected as the production model.</p>",
                unsafe_allow_html=True)

            mc1, mc2, mc3 = st.columns(3)
            for col, name, icon, clr, badge, role, desc in [
                (mc1, "Logistic Regression", "📐", AMBER, None, "Baseline",
                 "A linear classifier used as the statistical benchmark. "
                 "All ensemble models must outperform it to justify their added complexity. "
                 "Interpretable coefficients serve as a sanity check on feature relationships."),
                (mc2, "Random Forest", "🌲", BLUE2, None, "Primary Ensemble",
                 "Constructs 300 decision trees on bootstrapped data subsets and aggregates "
                 "by majority vote. Resistant to overfitting. Produces Gini-based "
                 "feature importance scores directly interpretable by educators."),
                (mc3, "XGBoost", "🚀", BLUE1, "⭐ Selected", "Best Performer",
                 "Sequential boosting — each tree corrects residual errors of the previous. "
                 "L1/L2 regularisation prevents overfitting. Supports SHAP TreeExplainer "
                 "for fast, exact feature attribution. Highest AUC-ROC on the test set."),
            ]:
                row = model_rows[name]
                badge_html = f"<span class='winner-badge'>{badge}</span>" if badge else ""
                col.markdown(
                    f"<div class='model-card'>"
                    f"<div class='model-card-accent' style='background:{clr};'></div>"
                    f"<p class='model-card-role' style='color:{clr};'>"
                    f"<span class='domain-card-icon' style='background:{clr}18;width:26px;"
                    f"height:26px;font-size:18px;'>{icon}</span>{role} {badge_html}</p>"
                    f"<h3 class='model-card-name'>{name}</h3>"
                    f"<p class='model-card-desc'>{desc}</p>"
                    f"<div style='margin-top:16px;'>"
                    f"<span class='metric-pill' style='background:{clr}18;color:{clr};'>"
                    f"Accuracy {row['Accuracy']:.3f}</span>"
                    f"<span class='metric-pill' style='background:{RED}18;color:{RED};'>"
                    f"Recall {row['Recall']:.3f}</span>"
                    f"<span class='metric-pill' style='background:{GREEN}18;color:{GREEN};'>"
                    f"AUC {row['Roc_auc']:.3f}</span>"
                    f"</div></div>",
                    unsafe_allow_html=True)

        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)

        # ── Full metrics table ──────────────────────────────────────
        with st.container(border=True):
            st.markdown(
                f"<span class='reveal-3'></span>"
                f"<div class='domain-card-head'>"
                f"<div class='domain-card-icon' style='background:{BLUE1}18;'>📋</div>"
                f"<div class='domain-card-title' style='color:{BLUE1};'>"
                f"Full Metrics Comparison — Test Set</div></div>"
                f"<p class='section-sub'>All metrics evaluated on the held-out test set "
                f"of 2,000 students that the models never saw during training. Green "
                f"cells mark the best score per metric.</p>",
                unsafe_allow_html=True)

            tbl = comparison[
                ["Model","Accuracy","Precision","Recall","F1","Roc_auc"]
            ].copy().rename(columns={"F1":"F1-Score","Roc_auc":"AUC-ROC"})

            st.dataframe(
                tbl.style.format(
                    {c: "{:.4f}" for c in tbl.columns if c != "Model"}
                ).highlight_max(
                    subset=[c for c in tbl.columns if c != "Model"],
                    color="#EEFAF4",
                ).set_properties(**{"font-size":"19.5px"}),
                use_container_width=True,
                hide_index=True,
                height=155,
            )

            st.markdown(
                "<div class='insight-card'>"
                "<p class='insight-card-title'>📌 Why Recall is the primary metric</p>"
                "<p class='insight-card-body'>"
                "In dropout prediction, a <strong>false negative</strong> — failing to "
                "flag a student who will drop out — has a far greater real-world cost than "
                "a <strong>false positive</strong> — flagging a student who stays. "
                "A missed at-risk student receives no intervention and may leave school. "
                "A falsely flagged student receives unnecessary (but harmless) counsellor attention. "
                "The model threshold is therefore calibrated to maximise Recall without "
                "unacceptable loss of Precision."
                "</p></div>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — ROC & CONFUSION MATRIX (interactive model filter)
# ══════════════════════════════════════════════════════════════════════
with tab_roc:
    with padded():
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(
                f"<span class='reveal-4'></span>"
                f"<div class='domain-card-head'>"
                f"<div class='domain-card-icon' style='background:{PURPLE}18;'>📈</div>"
                f"<div class='domain-card-title' style='color:{PURPLE};'>ROC Curves</div></div>"
                f"<p class='section-sub'>"
                f"Area Under Curve (AUC) measures overall discrimination ability across "
                f"all classification thresholds. A random classifier scores 0.50. "
                f"Use the filter to isolate one or two models on the chart below.</p>",
                unsafe_allow_html=True)

            roc_selected = st.multiselect(
                "Compare models", list(MODEL_SLUGS.keys()),
                default=list(MODEL_SLUGS.keys()), key="roc_filter",
            )

            if not roc_selected:
                st.markdown(
                    "<div class='empty-note'>Select at least one model above to "
                    "draw its ROC curve.</div>", unsafe_allow_html=True)
            else:
                fig_roc = roc_chart()
                # roc_chart() doesn't expose which trace belongs to which
                # model directly, so match on trace name — any trace whose
                # name doesn't mention a model (e.g. the diagonal random-
                # classifier reference line) is always left visible.
                for tr in fig_roc.data:
                    name_l = (tr.name or "").lower()
                    is_model_trace = any(
                        slug.replace("_", " ") in name_l or m.lower() in name_l
                        for m, slug in MODEL_SLUGS.items()
                    )
                    if is_model_trace:
                        matched = any(m.lower() in name_l for m in roc_selected)
                        tr.visible = True if matched else "legendonly"
                for i, name in enumerate(roc_selected):
                    fig_roc.add_annotation(
                        x=0.98, y=0.22 - i * 0.09, xref="paper", yref="paper",
                        text=f"<b>{name}</b>  AUC {model_rows[name]['Roc_auc']:.3f}",
                        showarrow=False, xanchor="right", align="right",
                        font=dict(family="Inter", size=13, color="#FFFFFF"),
                        bgcolor=MODEL_COLORS[name], bordercolor=MODEL_COLORS[name],
                        borderpad=6, borderwidth=0, opacity=0.92,
                    )
                fig_roc.update_layout(
                    plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                    font=dict(family="Inter", size=13),
                    height=500, margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_roc, use_container_width=True,
                                 config={"displayModeBar": False})

        st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(
                f"<span class='reveal-5'></span>"
                f"<div class='domain-card-head'>"
                f"<div class='domain-card-icon' style='background:{BLUE2}18;'>🧮</div>"
                f"<div class='domain-card-title' style='color:{BLUE2};'>Confusion Matrices</div></div>"
                f"<p class='section-sub'>"
                f"Rows show actual class, columns show predicted class. The filter above "
                f"applies here too — the bottom-left cell of each matrix is false "
                f"negatives (missed dropouts), the most important number to minimise.</p>",
                unsafe_allow_html=True)

            if not roc_selected:
                st.markdown(
                    "<div class='empty-note'>Select at least one model above to "
                    "view its confusion matrix.</div>", unsafe_allow_html=True)
            else:
                cm_cols = st.columns(len(roc_selected))
                for col, name in zip(cm_cols, roc_selected):
                    with col:
                        st.markdown(f"<div class='cm-label'>{name}</div>", unsafe_allow_html=True)
                        fig_cm = confusion_heatmap(MODEL_SLUGS[name], name)
                        fig_cm.update_layout(
                            plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                            font=dict(family="Inter", size=11),
                            height=300, margin=dict(l=10, r=10, t=30, b=10),
                        )
                        st.plotly_chart(fig_cm, use_container_width=True,
                                         config={"displayModeBar": False})

                fn_lines = []
                for name in roc_selected:
                    d = cm_data[name]
                    total_pos = d["fn"] + d["tp"]
                    fn_lines.append(
                        f"&nbsp;&nbsp;· {name} missed "
                        f"<strong style='color:{RED};'>{d['fn']} students</strong> "
                        f"({d['fn']/total_pos*100:.1f}%) &nbsp;|&nbsp; "
                        f"caught <strong style='color:{GREEN};'>{d['tp']}</strong> "
                        f"&nbsp;|&nbsp; {d['fp']} false alarms"
                    )
                st.markdown(
                    f"<div class='insight-card'>"
                    f"<p class='insight-card-title'>📌 False Negative Comparison</p>"
                    f"<p class='insight-card-body'>"
                    + "<br>".join(fn_lines) +
                    f"</p></div>",
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — PRECISION-RECALL & CALIBRATION (interactive model filter)
# ══════════════════════════════════════════════════════════════════════
with tab_pr:
    with padded():
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(
                f"<span class='reveal-6'></span>"
                f"<div class='domain-card-head'>"
                f"<div class='domain-card-icon' style='background:{GREEN}18;'>🎯</div>"
                f"<div class='domain-card-title' style='color:{GREEN};'>"
                f"Precision-Recall Curves</div></div>"
                f"<p class='section-sub'>Particularly informative for imbalanced datasets. "
                f"Average Precision (AP) summarises the area under the curve — higher is "
                f"better. The dashed baseline represents a random classifier that always "
                f"predicts the positive class (dropout rate = 30%).</p>",
                unsafe_allow_html=True)

            pr_selected = st.multiselect(
                "Compare models", list(MODEL_SLUGS.keys()),
                default=list(MODEL_SLUGS.keys()), key="pr_filter",
            )

            if not pr_selected:
                st.markdown(
                    "<div class='empty-note'>Select at least one model above to "
                    "draw its precision-recall curve.</div>", unsafe_allow_html=True)
            else:
                from sklearn.metrics import precision_recall_curve, average_precision_score

                fig_pr = go.Figure()
                for name in pr_selected:
                    p = preds[name]
                    prec, rec, _ = precision_recall_curve(p["y_true"], p["y_pred_prob"])
                    ap = average_precision_score(p["y_true"], p["y_pred_prob"])
                    fig_pr.add_trace(go.Scatter(
                        x=rec, y=prec, mode="lines",
                        name=f"{name}  (AP = {ap:.4f})",
                        line=dict(color=MODEL_COLORS[name], width=2.5),
                    ))

                baseline = float(preds["XGBoost"]["y_true"].mean())
                fig_pr.add_hline(
                    y=baseline, line_color="#CCCCCC",
                    line_dash="dash", line_width=1.5,
                    annotation_text=f"Random baseline (AP = {baseline:.2f})",
                    annotation_font_color="#999999",
                )
                fig_pr.update_layout(
                    xaxis_title="Recall  (True Positive Rate)",
                    yaxis_title="Precision  (Positive Predictive Value)",
                    height=420,
                    plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                    font=dict(family="Inter", size=13),
                    legend=dict(x=0.01, y=0.06,
                                bgcolor="rgba(255,255,255,0.85)",
                                bordercolor=BORDER, borderwidth=1),
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_pr, use_container_width=True,
                                 config={"displayModeBar": False})

        st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)

        st.markdown(
            "<h2 class='section-h2'>Predicted Probability Distribution — XGBoost</h2>"
            "<p class='section-sub'>A well-calibrated model produces clearly separated "
            "probability distributions for the two classes. Overlap in the middle region "
            "represents inherent prediction uncertainty. Shown for the production model "
            "(XGBoost) regardless of the filter above.</p>",
            unsafe_allow_html=True)

        preds_xgb = preds["XGBoost"]
        col_dist1, col_dist2 = st.columns(2)

        with col_dist1:
            with st.container(border=True):
                st.markdown(
                    f"<span class='reveal-7'></span>"
                    f"<div class='domain-card-head'>"
                    f"<div class='domain-card-icon' style='background:{RED}18;'>📊</div>"
                    f"<div class='domain-card-title' style='color:{RED};'>"
                    f"By Actual Class</div></div>",
                    unsafe_allow_html=True)
                fig_dist = go.Figure()
                for cls, clr, label in [
                    (0, GREEN, "Completed (Actual = 0)"),
                    (1, RED,   "Dropped Out (Actual = 1)"),
                ]:
                    mask = preds_xgb["y_true"] == cls
                    fig_dist.add_trace(go.Histogram(
                        x=preds_xgb[mask]["y_pred_prob"],
                        name=label, marker_color=clr, opacity=0.65, nbinsx=40,
                    ))
                fig_dist.add_vline(
                    x=0.5, line_color=DARK, line_dash="dash", line_width=2,
                    annotation_text="Decision threshold (0.5)",
                    annotation_font_color=DARK,
                )
                fig_dist.update_layout(
                    barmode="overlay",
                    xaxis_title="Predicted Dropout Probability",
                    yaxis_title="Number of Students",
                    height=360,
                    plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                    font=dict(family="Inter", size=13),
                    legend=dict(x=0.35, y=0.95),
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_dist, use_container_width=True,
                                 config={"displayModeBar": False})

        with col_dist2:
            with st.container(border=True):
                st.markdown(
                    f"<span class='reveal-7'></span>"
                    f"<div class='domain-card-head'>"
                    f"<div class='domain-card-icon' style='background:{BLUE1}18;'>🌡️</div>"
                    f"<div class='domain-card-title' style='color:{BLUE1};'>"
                    f"Overall Risk Distribution</div></div>",
                    unsafe_allow_html=True)
                fig_over = go.Figure()
                fig_over.add_trace(go.Histogram(
                    x=preds_xgb["y_pred_prob"],
                    marker_color=BLUE1, opacity=0.85, nbinsx=40,
                    name="All students",
                ))
                for thresh, clr, label, pos in [
                    (0.70, RED,   "High risk threshold (70%)", "top right"),
                    (0.40, AMBER, "Medium threshold (40%)",    "top left"),
                ]:
                    fig_over.add_vline(
                        x=thresh, line_color=clr, line_dash="dash", line_width=2,
                        annotation_text=label, annotation_font_color=clr,
                        annotation_position=pos,
                    )
                fig_over.update_layout(
                    xaxis_title="Predicted Dropout Probability",
                    yaxis_title="Number of Students",
                    height=360,
                    plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                    font=dict(family="Inter", size=13),
                    showlegend=False,
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_over, use_container_width=True,
                                 config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════
with tab_feat:
    with padded():
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(
                f"<span class='reveal-8'></span>"
                f"<div class='domain-card-head'>"
                f"<div class='domain-card-icon' style='background:{PURPLE}18;'>🔬</div>"
                f"<div class='domain-card-title' style='color:{PURPLE};'>"
                f"Global Feature Importance — SHAP</div></div>"
                f"<p class='section-sub'>Mean absolute SHAP value per feature — the "
                f"average magnitude of each feature's contribution to the dropout "
                f"prediction across all 2,000 test students. Unlike Gini importance, "
                f"SHAP values are model-agnostic and direction-aware.</p>",
                unsafe_allow_html=True)

            imp_tab1, imp_tab2 = st.tabs(["SHAP Importance", "XGBoost Gini Importance"])

            with imp_tab1:
                top15_shap = shap_imp.head(15).sort_values(ascending=True)
                shap_colors = [
                    RED   if v >= shap_imp.quantile(0.75)
                    else AMBER if v >= shap_imp.quantile(0.50)
                    else GREEN
                    for v in top15_shap.values
                ]
                fig_shap = go.Figure(go.Bar(
                    x=top15_shap.values, y=top15_shap.index, orientation="h",
                    marker_color=shap_colors,
                    marker_line_color=WHITE, marker_line_width=0.5,
                    text=[f"{v:.4f}" for v in top15_shap.values],
                    textposition="inside", insidetextanchor="middle",
                    textfont=dict(color=WHITE, size=12, family="Inter"),
                    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                ))
                fig_shap.update_layout(
                    xaxis_title="Mean Absolute SHAP Value",
                    height=480, margin=dict(l=10, r=20, t=10, b=40),
                    plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                    font=dict(family="Inter", size=13),
                )
                st.plotly_chart(fig_shap, use_container_width=True,
                                 config={"displayModeBar": False})

                col_leg1, col_leg2, col_leg3 = st.columns(3)
                col_leg1.markdown(
                    f"<div class='impact-chip' style='background:{RED}18;color:{RED};'>"
                    f"🔴 High impact — top 25%</div>", unsafe_allow_html=True)
                col_leg2.markdown(
                    f"<div class='impact-chip' style='background:{AMBER}18;color:{AMBER};'>"
                    f"🟡 Medium impact — 25–50%</div>", unsafe_allow_html=True)
                col_leg3.markdown(
                    f"<div class='impact-chip' style='background:{GREEN}18;color:{GREEN};'>"
                    f"🟢 Lower impact — bottom 50%</div>", unsafe_allow_html=True)

            with imp_tab2:
                top15_gini = xgb_fi.sort_values(ascending=False).head(15).sort_values(ascending=True)
                gini_colors = [
                    BLUE1 if v >= xgb_fi.quantile(0.75)
                    else BLUE2 if v >= xgb_fi.quantile(0.50)
                    else "#A8C4FF"
                    for v in top15_gini.values
                ]
                fig_gini = go.Figure(go.Bar(
                    x=top15_gini.values, y=top15_gini.index, orientation="h",
                    marker_color=gini_colors,
                    marker_line_color=WHITE, marker_line_width=0.5,
                    text=[f"{v:.4f}" for v in top15_gini.values],
                    textposition="inside", insidetextanchor="middle",
                    textfont=dict(color=WHITE, size=12, family="Inter"),
                    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                ))
                fig_gini.update_layout(
                    xaxis_title="Importance Score (XGBoost Gain)",
                    height=480, margin=dict(l=10, r=20, t=10, b=40),
                    plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                    font=dict(family="Inter", size=13),
                )
                st.plotly_chart(fig_gini, use_container_width=True,
                                 config={"displayModeBar": False})
                st.markdown(
                    "<div class='insight-card'>"
                    "<p class='insight-card-title'>SHAP vs Gini — What is the difference?</p>"
                    "<p class='insight-card-body'>"
                    "XGBoost Gini importance measures how often a feature is used in a "
                    "tree split, weighted by the improvement it brings. It reflects "
                    "global model behaviour but can be biased toward high-cardinality features. "
                    "SHAP values measure the actual contribution of each feature to each "
                    "individual prediction — they are more reliable, direction-aware, and "
                    "the basis for per-student explanations in the predictor page."
                    "</p></div>",
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 5 — VALIDATION & RATIONALE (interactive model filter on CV chart)
# ══════════════════════════════════════════════════════════════════════
with tab_cv:
    with padded():
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(
                f"<span class='reveal-9'></span>"
                f"<div class='domain-card-head'>"
                f"<div class='domain-card-icon' style='background:{AMBER}18;'>🔁</div>"
                f"<div class='domain-card-title' style='color:{AMBER};'>"
                f"5-Fold Cross-Validation Results</div></div>"
                f"<p class='section-sub'>Models were evaluated on five different "
                f"train/test splits of the training data. Bars show the mean score per "
                f"metric; error bars show standard deviation across the five folds — "
                f"shorter error bars mean more consistent, generalisable performance.</p>",
                unsafe_allow_html=True)

            cv_selected = st.multiselect(
                "Compare models", list(MODEL_SLUGS.keys()),
                default=list(MODEL_SLUGS.keys()), key="cv_filter",
            )

            if not cv_selected:
                st.markdown(
                    "<div class='empty-note'>Select at least one model above to "
                    "chart its cross-validation results.</div>", unsafe_allow_html=True)
            else:
                cv_plot_data = []
                for _, row in comparison[comparison["Model"].isin(cv_selected)].iterrows():
                    for metric, label in [
                        ("CV_recall_mean", "Recall"),
                        ("CV_f1_mean",     "F1-Score"),
                        ("CV_roc_auc_mean","AUC-ROC"),
                    ]:
                        cv_plot_data.append({
                            "Model":  row["Model"],
                            "Metric": label,
                            "Mean":   row[metric],
                            "Std":    row[metric.replace("mean","std")],
                        })
                cv_plot_df = pd.DataFrame(cv_plot_data)

                fig_cv = px.bar(
                    cv_plot_df, x="Metric", y="Mean", color="Model",
                    barmode="group", error_y="Std",
                    color_discrete_map=MODEL_COLORS,
                    labels={"Mean":"CV Mean Score","Metric":"Metric"},
                )
                fig_cv.update_layout(
                    height=400,
                    plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                    font=dict(family="Inter", size=13),
                    legend=dict(title="", x=0.7, y=0.98),
                    yaxis=dict(range=[0.80, 1.02]),
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_cv, use_container_width=True,
                                 config={"displayModeBar": False})

            st.markdown(
                "<div class='cv-note'>"
                "⚠️ <strong>Important:</strong> All CV results are computed on the "
                "SMOTE-balanced training set (11,200 records). The held-out test set "
                "(2,000 records, unbalanced, 30% dropout) was never seen during "
                "training or cross-validation — ensuring the reported test metrics "
                "reflect genuine out-of-sample generalisation."
                "</div>",
                unsafe_allow_html=True)

        st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)

        with st.container():
            st.markdown(
                "<span class='reveal-10'></span>"
                "<h2 class='section-h2'>Why XGBoost Was Selected</h2>",
                unsafe_allow_html=True)

            r1, r2 = st.columns(2)
            with r1:
                st.markdown(
                    "<div class='quote-card'>"
                    "<p class='quote-label'>Technical</p>"
                    "<p class='quote-text'>Highest AUC-ROC and fewest false alarms — "
                    "and the only model here with fast, exact SHAP TreeExplainer support.</p>"
                    "</div>",
                    unsafe_allow_html=True)
            with r2:
                st.markdown(
                    "<div class='quote-card'>"
                    "<p class='quote-label'>Practical</p>"
                    "<p class='quote-text'>A risk score means nothing without a reason. "
                    "SHAP turns every prediction into a specific, defensible explanation "
                    "a counsellor can act on.</p>"
                    "</div>",
                    unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="edu-footer">
  <div class="edu-footer-credit">
    EduGuard — Early Dropout Intelligence System &nbsp;·&nbsp;
  </div>
</div>
""", unsafe_allow_html=True)