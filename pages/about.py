import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import load_artefacts, SAMPLE_BATCH

st.set_page_config(
    page_title="EduGuard — About",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Palette ────────────────────────────────────────────────────────────────────
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

SCHOOL_IMG = "https://images.unsplash.com/photo-1580582932707-520aed937b7b?w=1200&q=80"

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700;9..144,900&display=swap');
*,html,body,[class*="css"]{font-family:'Inter',sans-serif!important;box-sizing:border-box;}
[data-testid*="Icon"]{font-family:'Material Symbols Rounded'!important;}
.main{background:#F7F8FF!important;font-size:16px;}
.block-container{padding:0 56px!important;max-width:100%!important;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none;}

[data-testid="stMarkdownContainer"] p{font-size:15px;}
[data-testid="stDataFrame"] *{font-size:14.5px!important;}

/* ── Navbar (shared look across all EduGuard pages) ───────────────────── */
.edu-nav{position:sticky;top:0;z-index:9999;background:rgba(37,41,67,0.92);
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  margin:0 -56px;
  padding:0 56px;display:flex;align-items:center;justify-content:space-between;
  min-height:72px;border-bottom:1px solid rgba(255,255,255,0.07);}
.edu-brand{display:flex;align-items:center;gap:13px;text-decoration:none!important;}
.edu-brand-icon{background:linear-gradient(135deg,#3A4EFB,#33A4FA);width:42px;height:42px;
  border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:21px;
  box-shadow:0 6px 16px rgba(58,78,251,0.35);}
.edu-brand-name{color:#fff;font-size:22px;font-weight:900;letter-spacing:-0.5px;line-height:1.1;}
.edu-brand-sub{color:#E3FF3B;font-size:10px;font-weight:700;letter-spacing:2px;
  text-transform:uppercase;margin-top:2px;}
.edu-navlinks{display:flex;align-items:center;gap:2px;}
.edu-navlinks a{color:rgba(222,224,237,0.72)!important;font-size:15px;font-weight:500;
  padding:9px 18px;border-radius:8px;text-decoration:none!important;transition:color .2s,background .2s;}
.edu-navlinks a:hover{color:#fff!important;background:rgba(255,255,255,0.07);}
.edu-navlinks a.active{color:#E3FF3B!important;background:rgba(227,255,59,0.1);}
.edu-live{display:inline-flex;align-items:center;gap:7px;background:rgba(227,255,59,0.12);
  border:1px solid rgba(227,255,59,0.4);color:#E3FF3B;padding:7px 16px;border-radius:24px;
  font-size:12px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;}
.edu-live-dot{width:6px;height:6px;border-radius:50%;background:#E3FF3B;
  animation:eduPulse 1.8s ease-in-out infinite;}
@keyframes eduPulse{0%,100%{opacity:1;}50%{opacity:0.35;}}

.page-wrap{padding:36px 0 48px;}
.page-eyebrow{font-size:13px;font-weight:700;letter-spacing:2.5px;
  text-transform:uppercase;color:#3A4EFB;margin:0 0 8px;}
.page-h1{font-size:36px;font-weight:900;color:#252943;letter-spacing:-1px;
  margin:0 0 8px;line-height:1.15;}
.page-sub{font-size:15.5px;color:#6B7280;margin:0 0 4px;line-height:1.6;max-width:700px;}

.section-h2{font-size:28px;font-weight:800;color:#252943;
  letter-spacing:-0.5px;margin:0 0 6px;}
.section-sub{font-size:15px;color:#6B7280;margin:0 0 24px;line-height:1.6;}

/* ── Hero / Research Identity slide ────────────────────────────────────
   The opening slide of the deck — given the most visual weight since it's
   the first thing an examiner sees, styled after the Home page hero
   (dot-grid texture, radial glow, two-tone accent title). */
.identity-card{background:linear-gradient(135deg,#151833 0%,#1a2060 55%,#0d3a6e 100%);
  border-radius:26px;padding:56px 60px;margin-bottom:8px;position:relative;
  overflow:hidden;min-height:420px;display:flex;flex-direction:column;
  justify-content:center;align-items:center;text-align:center;}
.identity-card::before{content:"";position:absolute;inset:0;z-index:0;
  background-image:radial-gradient(rgba(255,255,255,0.05) 1px, transparent 1px);
  background-size:26px 26px;pointer-events:none;}
.identity-card-glow{position:absolute;top:-90px;right:-80px;width:360px;height:360px;
  border-radius:50%;background:radial-gradient(circle,#33A4FA26,transparent 70%);z-index:0;}
.identity-card-glow-2{position:absolute;bottom:-110px;left:-70px;width:320px;height:320px;
  border-radius:50%;background:radial-gradient(circle,#E3FF3B16,transparent 70%);z-index:0;}
.identity-badge{position:relative;z-index:1;display:block;background:#E3FF3B;
  color:#252943;padding:7px 18px;border-radius:24px;font-size:11.5px;font-weight:800;
  letter-spacing:2.2px;text-transform:uppercase;margin:0 auto 28px;width:fit-content;}
[data-testid="stMarkdownContainer"] h2.identity-title,
.identity-title,
.identity-title *{color:#FFFFFF!important;
  font-family:'Fraunces','Georgia',ui-serif,serif!important;}
[data-testid="stMarkdownContainer"] h2.identity-title{
  text-align:center!important;margin-left:auto!important;margin-right:auto!important;}
.identity-title{position:relative;z-index:1;font-size:39px;font-weight:600;
  font-optical-sizing:auto;line-height:1.34;margin:0 auto 34px;
  letter-spacing:-0.2px;max-width:1120px;width:100%;
  text-wrap:balance;text-shadow:0 2px 18px rgba(0,0,0,0.35);}
.identity-grid{position:relative;z-index:1;display:grid;grid-template-columns:repeat(3,1fr);
  gap:18px;margin-top:6px;width:100%;}
.identity-item{background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.14);
  border-radius:14px;padding:18px 20px;display:flex;align-items:flex-start;gap:13px;
  text-align:left;transition:background .2s ease, transform .2s ease;}
.identity-item:hover{background:rgba(255,255,255,0.12);transform:translateY(-3px);}
.identity-item-icon{font-size:21px;flex-shrink:0;margin-top:1px;}
.identity-item-label{font-size:11px;font-weight:700;color:rgba(255,255,255,0.55);
  letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;}
.identity-item-value{font-size:15.5px;font-weight:600;color:#fff;line-height:1.4;}

/* Stat cards */
.stat-card{background:#fff;border:1px solid #E0E3FF;border-radius:16px;
  padding:22px 20px;box-shadow:0 4px 18px rgba(58,78,251,0.07);
  text-align:center;height:118px;display:flex;flex-direction:column;
  align-items:center;justify-content:center;transition:transform .2s ease, box-shadow .2s ease;}
.stat-card:hover{transform:translateY(-4px);box-shadow:0 14px 28px rgba(58,78,251,0.14);}
.stat-num{font-size:38px;font-weight:900;letter-spacing:-1.5px;line-height:1;}
.stat-label{font-size:13px;color:#6B7280;font-weight:500;
  margin-top:6px;text-transform:uppercase;letter-spacing:0.5px;}

/* Problem section */
.problem-img{width:100%;height:280px;object-fit:cover;object-position:center;
  border-radius:16px;display:block;}
.problem-card{background:#fff;border:1px solid #E0E3FF;border-radius:16px;
  padding:22px 24px;box-shadow:0 4px 18px rgba(58,78,251,0.07);height:100%;
  transition:transform .2s ease, box-shadow .2s ease;}
.problem-card:hover{transform:translateY(-4px);box-shadow:0 14px 28px rgba(58,78,251,0.14);}
.problem-card-icon{font-size:34px;margin-bottom:12px;}
.problem-card-title{font-size:17px;font-weight:800;color:#252943;
  margin-bottom:8px;line-height:1.3;}
.problem-card-body{font-size:14px;color:#6B7280;line-height:1.6;margin:0;}

/* Architecture flowchart */
.arch-layer{border-radius:14px;padding:22px 26px;margin-bottom:0;}
.arch-layer-title{font-size:12.5px;font-weight:800;letter-spacing:2px;
  text-transform:uppercase;margin-bottom:14px;}
.arch-items{display:flex;gap:10px;flex-wrap:wrap;}
.arch-item{padding:9px 17px;border-radius:8px;font-size:14px;
  font-weight:600;border:1px solid rgba(0,0,0,0.08);}
.arch-arrow-wrap{display:flex;justify-content:center;margin:2px 0;}
.arch-arrow{width:0;height:0;border-left:11px solid transparent;
  border-right:11px solid transparent;border-top:15px solid #C7CCF5;}

/* Feature reference — grouped domain cards, tooltip on each chip */
.featref-card{background:#fff;border:1px solid #E0E3FF;border-radius:16px;
  padding:22px 24px;box-shadow:0 4px 18px rgba(58,78,251,0.07);height:100%;
  transition:transform .2s ease, box-shadow .2s ease;}
.featref-card:hover{transform:translateY(-4px);box-shadow:0 14px 28px rgba(58,78,251,0.13);}
.featref-domain-label{font-size:13.5px;font-weight:800;letter-spacing:0.4px;
  text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:8px;}
.featref-row{display:flex;justify-content:space-between;align-items:center;gap:12px;
  padding:10px 0;border-bottom:1px solid #EEF0FA;cursor:help;}
.featref-row:last-child{border-bottom:none;}
.featref-name{font-size:14.5px;font-weight:600;color:#252943;}
.featref-value{font-size:12.5px;font-weight:700;padding:4px 11px;border-radius:20px;
  white-space:nowrap;flex-shrink:0;}

/* Risk cards */
.risk-def-card{border-radius:14px;padding:22px 24px;margin-bottom:0;}
.risk-def-title{font-size:19px;font-weight:800;margin-bottom:6px;}
.risk-def-prob{font-size:14px;font-weight:600;margin-bottom:10px;opacity:0.8;}
.risk-def-body{font-size:14px;line-height:1.6;margin:0;}

/* Insight boxes */
.insight-box{background:#F4F5FF;border:1px solid #E0E3FF;border-radius:12px;
  padding:18px 22px;margin:16px 0;}
.insight-title{font-size:13px;font-weight:700;color:#3A4EFB;
  letter-spacing:0.5px;text-transform:uppercase;margin-bottom:8px;}
.insight-body{font-size:14px;color:#374151;line-height:1.65;margin:0;}

/* Synthetic data source cards */
.source-card{background:#fff;border:1px solid #E0E3FF;border-radius:16px;
  padding:24px 22px;box-shadow:0 4px 18px rgba(58,78,251,0.07);height:100%;
  transition:transform .2s ease, box-shadow .2s ease;}
.source-card:hover{transform:translateY(-4px);box-shadow:0 14px 28px rgba(58,78,251,0.14);}
.source-icon-wrap{width:44px;height:44px;border-radius:12px;display:flex;
  align-items:center;justify-content:center;font-size:21px;margin-bottom:14px;}
.source-name{font-size:15.5px;font-weight:800;color:#252943;margin-bottom:2px;line-height:1.3;}
.source-sub{font-size:12.5px;color:#9CA3AF;margin-bottom:16px;}
.source-stat{display:flex;justify-content:space-between;gap:10px;font-size:13px;
  padding:8px 0;border-top:1px solid #EEF0FA;}
.source-stat-label{color:#6B7280;}
.source-stat-value{color:#252943;font-weight:700;text-align:right;}

.criteria-row{display:flex;gap:12px;flex-wrap:wrap;margin-top:20px;}
.criteria-chip{background:#F0FFF9;border:1px solid #B8F5E1;color:#00875F;
  padding:9px 16px;border-radius:20px;font-size:13px;font-weight:700;
  display:flex;align-items:center;gap:6px;}

.justify-toggle{margin-top:18px;}
.justify-toggle summary{cursor:pointer;font-size:13.5px;font-weight:700;
  color:#3A4EFB;list-style:none;display:flex;align-items:center;gap:6px;}
.justify-toggle summary::-webkit-details-marker{display:none;}
.justify-toggle summary::after{content:"›";margin-left:6px;transition:transform .2s ease;}
.justify-toggle[open] summary::after{transform:rotate(90deg);}
.justify-toggle-body{margin-top:14px;}

/* Reference item */
.ref-item{padding:13px 16px;border-radius:10px;margin-bottom:6px;
  font-size:14.5px;color:#374151;line-height:1.6;transition:background .15s ease;}
.ref-item:hover{background:#F7F8FF;}
.ref-tag{display:inline-block;background:#F4F5FF;color:#3A4EFB;
  padding:3px 11px;border-radius:10px;font-size:12px;font-weight:700;
  margin-right:8px;vertical-align:middle;}

/* Tech stack icon grid */
.techgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;}
.techgrid-item{background:#fff;border:1px solid #E0E3FF;border-radius:14px;
  padding:22px 12px;text-align:center;box-shadow:0 4px 14px rgba(58,78,251,0.06);
  transition:transform .2s ease, box-shadow .2s ease;}
.techgrid-item:hover{transform:translateY(-4px);box-shadow:0 12px 26px rgba(58,78,251,0.14);}
.techgrid-icon{width:50px;height:50px;border-radius:13px;display:flex;align-items:center;
  justify-content:center;font-size:25px;margin:0 auto 12px;}
.techgrid-name{font-size:13.5px;font-weight:700;color:#252943;}

/* Tabs */
[data-testid="stTabs"] [role="tablist"]{background:#F4F5FF;padding:6px;
  border-radius:12px;gap:4px!important;margin-bottom:16px;}
[data-testid="stTabs"] [data-testid="stTab"]{background:#fff;border-radius:8px!important;
  padding:9px 20px!important;font-weight:600;font-size:14.5px;color:#252943;
  border:1px solid #E0E3FF!important;}
[data-testid="stTabs"] [data-testid="stTab"] p{color:inherit!important;font-weight:inherit!important;}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"]{background:#252943!important;
  color:#E3FF3B!important;border-color:#252943!important;}
[data-testid="stTabs"] .react-aria-SelectionIndicator{display:none!important;}
.stTabs [data-baseweb="tab-list"]{background:#F4F5FF;padding:6px 8px;
  border-radius:12px;gap:4px;margin-bottom:16px;}
.stTabs [data-baseweb="tab"]{background:#fff;border-radius:8px;
  padding:9px 20px;font-weight:600;font-size:14.5px;color:#252943;
  border:1px solid #E0E3FF;}
.stTabs [aria-selected="true"]{background:#252943!important;
  color:#E3FF3B!important;border-color:#252943!important;}
[data-baseweb="tab-highlight"]{display:none!important;}

.edu-footer{text-align:center;color:#9CA3AF;font-size:13.5px;
  padding:28px 56px 20px;border-top:1px solid #E0E3FF;margin:40px -56px 0;}

/* ── Slide deck controls ───────────────────────────────────────────────
   The nav-pill rows and Prev/Next buttons are plain st.button elements —
   scoped smaller than the app's default button styling via the same
   marker + :has() trick used for the reveal animation, so this page's
   navigation doesn't inherit the large primary-CTA sizing used for
   things like the CSV download button further down the deck. */
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .navpill-marker)
  button[data-testid^="stBaseButton"]{
  font-size:13px!important;padding:9px 8px!important;border-radius:9px!important;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .navpill-marker)
  button[data-testid^="stBaseButton"] p{font-size:13px!important;}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .navpill-marker)
  button[data-testid^="stBaseButton-secondary"]{
  background:#fff!important;border-color:#E0E3FF!important;color:#374151!important;}

.slide-progress{display:flex;align-items:center;justify-content:space-between;
  margin:18px 0 20px;flex-wrap:wrap;gap:10px;}
.slide-progress-text{font-size:13px;font-weight:700;color:#3A4EFB;
  letter-spacing:0.4px;text-transform:uppercase;}
.slide-dots{display:flex;gap:6px;}
.slide-dot{width:7px;height:7px;border-radius:50%;background:#E0E3FF;}
.slide-dot.done{background:#3A4EFB;}

div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .prevnext-marker)
  button[data-testid^="stBaseButton"]{font-size:14.5px!important;padding:11px 22px!important;}

/* ── Slide content entrance animation (marker + :has(), same mechanism
   used throughout EduGuard) — retriggers on every slide change because
   Streamlit rebuilds the container from a fresh script run. ─────────── */
@keyframes slideFadeIn{from{opacity:0;transform:translateY(22px);}
  to{opacity:1;transform:translateY(0);}}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .slide-anim){
  animation:slideFadeIn .45s cubic-bezier(.22,.8,.32,1) both;}

@media(max-width:768px){
  .block-container{padding:0 20px!important;}
  .edu-nav{margin:0 -20px;padding:0 20px;}.edu-navlinks{display:none;}
  .edu-footer{margin:40px -20px 0;padding:28px 20px 20px;}
  .page-wrap{padding:24px 0 32px;}
  .page-h1{font-size:26px;}
  .identity-card{padding:32px 24px;}
  .identity-title{font-size:24px;}
  .identity-grid{grid-template-columns:1fr 1fr;}
  .techgrid{grid-template-columns:repeat(2,1fr);}
}
</style>
""", unsafe_allow_html=True)

# ── NAVBAR ─────────────────────────────────────────────────────────────────────
st.markdown("""
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
    <a href="/predictor">Student Predictor</a>
    <a href="/batch">Batch Dashboard</a>
    <a href="/performance">Model Performance</a>
    <a href="/about" class="active">About</a>
  </div>
  <div class="edu-live"><span class="edu-live-dot"></span>Live</div>
</nav>
""", unsafe_allow_html=True)

# ── LOAD ───────────────────────────────────────────────────────────────────────
A = load_artefacts()

st.markdown("<div class='page-wrap'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE HEADER (always visible — this is the page identity, not a slide)
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<p class="page-eyebrow">Research Project</p>
<h1 class="page-h1">About EduGuard</h1>
<p class="page-sub">
  A click-through walkthrough of the research built for the viva defence.
  Use the tabs or Next/Previous to move through it one idea at a time.
</p>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PRECOMPUTED DATA SHARED ACROSS SLIDES
# ══════════════════════════════════════════════════════════════════════
meth_df = pd.DataFrame([
    ["Research design",       "Design Science Research (DSR) — produces and evaluates a technical artefact"],
    ["Problem formulation",   "Supervised binary classification — Dropout (1) vs Completed (0)"],
    ["Dataset size",          "10,000 synthetic student records"],
    ["Dataset calibration",   "UNESCO UIS Nigeria (2023), World Bank GHS-Panel, UCI Dropout Dataset (Realinho et al., 2022)"],
    ["Features",              "17 predictors across 5 domains — Demographic, Socioeconomic, Academic, Cognitive, Psychomotor"],
    ["Preprocessing",         "Label/ordinal/one-hot encoding · StandardScaler · Stratified 80/10/10 split (train/validation/test) · SMOTE balancing"],
    ["Models trained",        "Logistic Regression (baseline), Random Forest (n=300), XGBoost (n=500, lr=0.05)"],
    ["Validation",            "5-fold stratified cross-validation on training set · dedicated validation set of 1,000 students for XGBoost early stopping · held-out test set of 1,000 students"],
    ["Selected model",        "XGBoost — highest AUC-ROC, lowest false positives, native SHAP TreeExplainer support"],
    ["Explainability",        "SHAP TreeExplainer — per-student feature attribution, waterfall charts"],
    ["Intervention framework","Ojo & Awonuga (2022) evidence-based Nigerian secondary school intervention taxonomy"],
    ["Deployment",            "Streamlit Community Cloud — public URL, no installation required"],
], columns=["Component", "Detail"])

feat_domains = [
    (BLUE1, "👤", "Demographic", [
        ("Gender", "Male / Female",
         "Adeyemo et al. (2023) — female rural students at 2.3× higher risk"),
        ("Age", "11 – 20",
         "Fayomi & Nwosu (2022) — over-age students linked to class repetition"),
        ("Distance to School (km)", "0.5 – 35.0",
         "Fayomi & Nwosu (2022) — risk rises sharply beyond 5km"),
    ]),
    (PURPLE, "💰", "Socioeconomic", [
        ("Household Income Band", "Low / Medium / High",
         "World Bank GHS-Panel — strongest single socioeconomic predictor"),
        ("Fee Payment Status", "Unpaid / Partial / Paid",
         "Ojo & Awonuga (2022) — unpaid fees primary cause in 38% of cases"),
        ("Parental Education Level", "No Formal Edu → Postgraduate",
         "World Bank GHS-Panel — lower education reduces advocacy for schooling"),
        ("Parental Involvement", "Absent / Passive / Active",
         "Ojo & Awonuga (2022) — absent parents linked to 44% dropout rate"),
    ]),
    (BLUE2, "📚", "Academic", [
        ("Attendance Rate (%)", "0 – 100",
         "Hussain et al. (2023) — strongest individual predictor, AUC contribution 0.31"),
        ("Avg Academic Score (%)", "0 – 100",
         "Adekitan & Salau (2022) — composite score more predictive than single subjects"),
        ("Failed Subjects Count", "0 – 8",
         "Ojo & Awonuga (2022) — more than 2 failures is a recognised risk threshold"),
    ]),
    (AMBER, "🧠", "Cognitive", [
        ("Cognitive Assessment Score", "0 – 100",
         "Dasi & Kanakala (2022) — distinct from academic score, r=0.61 correlation"),
        ("Learning Pace", "Slow / Average / Fast",
         "FME (2020) — teacher-assigned; slow pace linked to progressive disengagement"),
        ("Study Habit Score", "0 – 100",
         "Zimmerman (2002) — self-regulated learning predictor of academic persistence"),
    ]),
    (GREEN, "🏃", "Psychomotor", [
        ("Practical Subject Score", "0 – 100",
         "Kizito (2023) — practical success protective against dropout for low-academic students"),
        ("PE Participation", "Absent / Passive / Active",
         "Kizito (2023) — school belonging protective factor; early withdrawal signals disengagement"),
        ("Psychomotor Dev. Score", "0 – 100",
         "FME (2020) — composite teacher-rated physical coordination and practical skill rubric"),
    ]),
]

sources = [
    (BLUE2, "🔵", "UCI Student Dropout Dataset", "Realinho et al. (2022)",
     [("Dropout rate", "~32%"), ("Role", "Most cited ML dropout benchmark")]),
    (AMBER, "🟡", "UNESCO UIS Nigeria", "Education Statistics (2023)",
     [("Dropout range", "28 – 35%"), ("Coverage", "Attendance & gender ratios")]),
    (GREEN, "🟢", "World Bank GHS-Panel", "Nigeria Household Survey (2023)",
     [("Variables", "Income · Parental Edu · Fees"), ("Scope", "Nigerian public households")]),
]

tech = [
    (BLUE1,  "🐍", "Python 3.13"),
    (AMBER,  "⚡", "XGBoost"),
    (PURPLE, "🔍", "SHAP"),
    (BLUE2,  "🤖", "scikit-learn"),
    (GREEN,  "⚖️", "imbalanced-learn"),
    (RED,    "🌐", "Streamlit"),
    (DARK,   "📊", "Plotly"),
    (BLUE1,  "📦", "pandas / numpy"),
]

@st.cache_data
def generate_sample():
    np.random.seed(99)
    n = 50
    genders   = np.random.choice(["Female","Male"], n, p=[0.52,0.48])
    ages      = np.random.choice(range(15,19), n)
    distances = np.round(np.clip(
        np.random.exponential(5, n) + 1, 0.5, 35.0), 1)
    incomes   = np.random.choice(
        ["Low","Medium","High"], n, p=[0.55,0.30,0.15])
    fees      = np.where(incomes=="Low",
                    np.random.choice(["Unpaid","Partial","Paid"],n,p=[0.45,0.35,0.20]),
                np.where(incomes=="Medium",
                    np.random.choice(["Unpaid","Partial","Paid"],n,p=[0.10,0.35,0.55]),
                    np.random.choice(["Unpaid","Partial","Paid"],n,p=[0.02,0.10,0.88])))
    parent_edus = np.random.choice(
        ["No Formal Education","Primary","Secondary",
         "OND/NCE","HND/BSc","Postgraduate"], n,
        p=[0.12,0.22,0.30,0.18,0.13,0.05])
    involvements = np.random.choice(
        ["Absent","Passive","Active"], n, p=[0.30,0.40,0.30])
    attendance  = np.round(np.clip(
        np.random.normal(70,18,n), 0, 100), 1)
    acad_score  = np.round(np.clip(
        0.45*attendance + np.random.normal(15,10,n), 5, 100), 1)
    failed_subj = np.clip(
        np.array([np.random.binomial(8, max(0.02,min(0.95,
            1-(s/100)*1.3))) for s in acad_score]), 0, 8)
    cog_score   = np.round(np.clip(
        acad_score*0.70+np.random.normal(0,12,n)+15, 5, 100), 1)
    learning_pace = np.array([
        np.random.choice(["Slow","Average","Fast"],
            p=[0.05,0.30,0.65] if s>=70
            else [0.20,0.55,0.25] if s>=50
            else [0.55,0.35,0.10])
        for s in acad_score])
    study_habit = np.round(np.clip(
        np.random.normal(55,18,n), 0, 100), 1)
    practical   = np.round(np.clip(
        np.random.normal(58,14,n), 0, 100), 1)
    pe_part     = np.array([
        np.random.choice(["Active","Passive","Absent"],
            p=[0.65,0.25,0.10] if a>=75
            else [0.30,0.45,0.25] if a>=50
            else [0.10,0.30,0.60])
        for a in attendance])
    psycho      = np.round(np.clip(
        np.where(pe_part=="Active",
            np.random.normal(72,12,n),
        np.where(pe_part=="Passive",
            np.random.normal(55,13,n),
            np.random.normal(38,14,n)))
        + np.random.normal(0,6,n), 0, 100), 1)
    return pd.DataFrame({
        "Student_ID":               [f"SS2/EGA/{i+1:03d}" for i in range(n)],
        "Gender":                   genders,
        "Age":                      ages,
        "Year_Group":               ["SS2"]*n,
        "Distance_to_School_km":    distances,
        "Household_Income_Band":    incomes,
        "Fee_Payment_Status":       fees,
        "Parental_Education_Level": parent_edus,
        "Parental_Involvement":     involvements,
        "Attendance_Rate_Pct":      attendance,
        "Avg_Academic_Score_Pct":   acad_score,
        "Failed_Subjects_Count":    failed_subj,
        "Cognitive_Assessment_Score": cog_score,
        "Learning_Pace":            learning_pace,
        "Study_Habit_Score":        study_habit,
        "Practical_Subject_Score":  practical,
        "PE_Participation":         pe_part,
        "Psychomotor_Dev_Score":    psycho,
    })


def featref_card_html(clr, icon, label, rows):
    rows_html = "".join(
        f"<div class='featref-row' title='{just}'>"
        f"<span class='featref-name'>{name}</span>"
        f"<span class='featref-value' style='background:{clr}18;color:{clr};'>{val}</span>"
        f"</div>"
        for name, val, just in rows
    )
    return (
        f"<div class='featref-card'>"
        f"<div class='featref-domain-label' style='color:{clr};'>{icon} {label}</div>"
        f"{rows_html}</div>"
    )


# ══════════════════════════════════════════════════════════════════════
# SLIDE RENDER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════
def slide_identity():
    st.markdown("""
    <div class="identity-card">
      <div class="identity-card-glow"></div>
      <div class="identity-card-glow-2"></div>
      <div class="identity-badge">Research — Computer Science</div>
      <h2 class="identity-title">
        Design and Implementation of a Machine Learning Model for
        Predicting Drop-Out Risk in Public Secondary Schools
      </h2>
      <div class="identity-grid">
        <div class="identity-item">
          <div class="identity-item-icon">🎓</div>
          <div>
            <div class="identity-item-label">Researcher</div>
            <div class="identity-item-value">Nzeako Jennifer Nkiruka</div>
          </div>
        </div>
        <div class="identity-item">
          <div class="identity-item-icon">🔢</div>
          <div>
            <div class="identity-item-label">Matric Number</div>
            <div class="identity-item-value">DU/C/247</div>
          </div>
        </div>
        <div class="identity-item">
          <div class="identity-item-icon">🧑‍🏫</div>
          <div>
            <div class="identity-item-label">Supervisor</div>
            <div class="identity-item-value">Mr Akinnifesi</div>
          </div>
        </div>
        <div class="identity-item">
          <div class="identity-item-icon">🏫</div>
          <div>
            <div class="identity-item-label">Department</div>
            <div class="identity-item-value">Physical and Mathematical Science</div>
          </div>
        </div>
        <div class="identity-item">
          <div class="identity-item-icon">📜</div>
          <div>
            <div class="identity-item-label">Degree</div>
            <div class="identity-item-value">Computer Science</div>
          </div>
        </div>
        <div class="identity-item">
          <div class="identity-item-icon">🛡️</div>
          <div>
            <div class="identity-item-label">System Name</div>
            <div class="identity-item-value">EduGuard — Early Dropout Intelligence System</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def slide_stats():
    st.markdown(
        "<h2 class='section-h2'>The Research at a Glance</h2>"
        "<p class='section-sub'>Six numbers that summarise the scope and "
        "results of the project.</p>",
        unsafe_allow_html=True)
    s1,s2,s3,s4,s5,s6 = st.columns(6)
    for col, num, lbl, clr in [
        (s1, "10,000", "Records generated",  BLUE1),
        (s2, "17",     "Predictive features", BLUE2),
        (s3, "5",      "Feature domains",     PURPLE),
        (s4, "3",      "Models compared",     DARK),
        (s5, "90.7%",  "XGBoost Recall",      RED),
        (s6, "98.6%",  "AUC-ROC",            GREEN),
    ]:
        col.markdown(
            f"<div class='stat-card'>"
            f"<div class='stat-num' style='color:{clr};'>{num}</div>"
            f"<div class='stat-label'>{lbl}</div>"
            f"</div>",
            unsafe_allow_html=True)


def slide_problem():
    st.markdown(
        "<h2 class='section-h2'>The Problem This System Solves</h2>"
        "<p class='section-sub'>Secondary school dropout in Nigeria is a "
        "large-scale, documented, and partially solvable problem — if the "
        "right tools exist.</p>",
        unsafe_allow_html=True)

    img_col, cards_col = st.columns([1, 1.4])
    with img_col:
        st.markdown(
            f'<img src="{SCHOOL_IMG}" class="problem-img" '
            f'alt="Nigerian secondary school students" '
            f'onerror="this.style.display=\'none\'" />',
            unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:12px;color:#9CA3AF;margin-top:6px;'>"
            "Photo: Unsplash — secondary school classroom</p>",
            unsafe_allow_html=True)

    with cards_col:
        problems = [
            (RED,    "❌", "No early warning mechanism",
             "Schools have no systematic process to flag at-risk students "
             "from existing records before dropout happens."),
            (AMBER,  "📂", "Data exists but is unused",
             "Attendance, CA scores, and term results are collected "
             "routinely but never analysed for patterns."),
            (PURPLE, "🔮", "Prediction without action",
             "Published dropout tools produce risk scores but leave "
             "educators without a concrete next step."),
            (BLUE1,  "🌍", "Contextual invalidity",
             "Most existing models train on Western data that ignores "
             "Nigerian socioeconomic dropout drivers."),
        ]
        pc1, pc2 = st.columns(2)
        for i, (clr, icon, title, body) in enumerate(problems):
            col = pc1 if i % 2 == 0 else pc2
            col.markdown(
                f"<div class='problem-card' style='margin-bottom:12px;'>"
                f"<div class='problem-card-icon'>{icon}</div>"
                f"<div class='problem-card-title' style='color:{clr};'>{title}</div>"
                f"<div class='problem-card-body'>{body}</div>"
                f"</div>",
                unsafe_allow_html=True)


def slide_architecture():
    st.markdown(
        "<h2 class='section-h2'>System Architecture</h2>"
        "<p class='section-sub'>Three-tier architecture — data and model "
        "layer, application layer, and presentation layer — deployed as a "
        "public Streamlit web application.</p>",
        unsafe_allow_html=True)

    arch_col1, arch_col2 = st.columns([1.6, 1])
    with arch_col1:
        layers = [
            (BLUE1, "#EEF0FF", "01 · Data & Model Layer",
             [("🗄️ Synthetic Dataset",    "#DDEEFF"),
              ("⚙️ Preprocessing",        "#DDEEFF"),
              ("📊 SMOTE Balancing",       "#DDEEFF"),
              ("🤖 Logistic Regression",   "#DDEEFF"),
              ("🌳 Random Forest",         "#DDEEFF"),
              ("⚡ XGBoost (selected)",    "#E3FF3B"),
              ("🔍 SHAP TreeExplainer",    "#DDEEFF")]),
            (GREEN, "#EEFAF4", "02 · Application Layer",
             [("🎯 Risk Predictor",        "#C8F5E6"),
              ("📊 Batch Dashboard",       "#C8F5E6"),
              ("📈 Model Performance",     "#C8F5E6"),
              ("📋 Intervention Engine",   "#C8F5E6")]),
            (AMBER, "#FFFBF0", "03 · Presentation Layer",
             [("🌐 Streamlit Web App",     "#FFE8A0"),
              ("📱 Mobile Responsive",     "#FFE8A0"),
              ("🔗 Public URL",            "#FFE8A0")]),
        ]
        for idx, (clr, bg, title, items) in enumerate(layers):
            items_html = "".join(
                f"<div class='arch-item' style='background:{ibg};color:{DARK};'>{item}</div>"
                for item, ibg in items
            )
            st.markdown(
                f"<div class='arch-layer' style='background:{bg};border:1px solid {clr}33;'>"
                f"<div class='arch-layer-title' style='color:{clr};'>{title}</div>"
                f"<div class='arch-items'>{items_html}</div>"
                f"</div>",
                unsafe_allow_html=True)
            if idx < len(layers) - 1:
                st.markdown(
                    "<div class='arch-arrow-wrap'><div class='arch-arrow'></div></div>",
                    unsafe_allow_html=True)

    with arch_col2:
        st.markdown(
            "<div class='insight-box' style='margin-top:0;'>"
            "<p class='insight-title'>Why this architecture</p>"
            "<p class='insight-body'>"
            "Separating the data/model layer from the application layer means "
            "the trained models are saved as artefacts (joblib .pkl files) and "
            "loaded at runtime — not retrained on every request. "
            "This keeps prediction response time under one second even on "
            "the free Streamlit Community Cloud tier.<br><br>"
            "The SHAP explainer is also pre-fitted and cached — so per-student "
            "explanations are computed instantly without the overhead of "
            "refitting the explainer on each prediction."
            "</p></div>",
            unsafe_allow_html=True)
        st.markdown(
            "<div class='insight-box'>"
            "<p class='insight-title'>Deployment</p>"
            "<p class='insight-body'>"
            "Deployed to <strong>Streamlit Community Cloud</strong> — "
            "free, public, accessible from any device via a URL. "
            "The examiner can open the live system during the viva defence "
            "presentation without installing anything."
            "</p></div>",
            unsafe_allow_html=True)


def slide_methodology():
    st.markdown(
        "<h2 class='section-h2'>Methodology Summary</h2>"
        "<p class='section-sub'>The design-science approach behind the "
        "system, end to end.</p>",
        unsafe_allow_html=True)
    st.dataframe(
        meth_df.style.set_properties(**{
            "font-size": "14.5px",
            "text-align": "left",
        }).set_table_styles([
            {"selector":"th", "props":[
                ("background-color",DARK),
                ("color","white"),
                ("font-weight","700"),
                ("font-size","13px"),
                ("letter-spacing","0.5px"),
            ]}
        ]),
        use_container_width=True,
        hide_index=True,
        height=460,
    )


def slide_features():
    st.markdown(
        "<h2 class='section-h2'>All 17 Features — Domain Reference</h2>"
        "<p class='section-sub'>Grouped by domain and colour-coded to match "
        "the rest of EduGuard. Hover a feature for its literature "
        "justification.</p>",
        unsafe_allow_html=True)
    fr1, fr2, fr3 = st.columns(3)
    for col, dom in zip([fr1, fr2, fr3], feat_domains[:3]):
        col.markdown(featref_card_html(*dom), unsafe_allow_html=True)
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    fr4, fr5 = st.columns(2)
    for col, dom in zip([fr4, fr5], feat_domains[3:]):
        col.markdown(featref_card_html(*dom), unsafe_allow_html=True)


def slide_risk():
    st.markdown(
        "<h2 class='section-h2'>Risk Level Definitions</h2>"
        "<p class='section-sub'>How a raw probability becomes an "
        "actionable risk band.</p>",
        unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    for col, clr, bg, emoji, level, prob, meaning, action in [
        (r1, RED,   "#FFF0F0",
         "🔴", "HIGH RISK", "Probability ≥ 70%",
         "The model predicts a strong likelihood of dropout based on multiple "
         "concurrent risk factors. The student is in immediate need of coordinated "
         "support from teachers, counsellor, and school administration.",
         "Contact family immediately · Arrange fee support · Escalate to VP Academics · "
         "Weekly check-ins required"),
        (r2, AMBER, "#FFFBF0",
         "🟡", "MEDIUM RISK", "Probability 40–69%",
         "Elevated risk present — typically one or two significant risk factors "
         "identified. The student is not in immediate danger of dropout but "
         "requires preventive intervention before risk compounds further.",
         "Counsellor referral · Attendance monitoring · Parent engagement · "
         "Peer mentorship programme"),
        (r3, GREEN, "#F0FFF9",
         "🟢", "LOW RISK", "Probability < 40%",
         "No significant risk factors identified. The student is on track. "
         "Standard monitoring is sufficient — no active intervention required "
         "unless new risk signals emerge in subsequent assessments.",
         "Continue standard monitoring · Reassess each term · "
         "Flag if attendance drops below 75%"),
    ]:
        col.markdown(
            f"<div class='risk-def-card' style='background:{bg};"
            f"border:2px solid {clr}44;'>"
            f"<div class='risk-def-title' style='color:{clr};'>{emoji} {level}</div>"
            f"<div class='risk-def-prob' style='color:{clr};'>{prob}</div>"
            f"<p class='risk-def-body' style='color:#374151;margin-bottom:12px;'>"
            f"{meaning}</p>"
            f"<div style='background:{clr}18;border-radius:8px;padding:10px 12px;"
            f"font-size:13px;color:{clr};font-weight:600;line-height:1.6;'>"
            f"Recommended response:<br>{action}"
            f"</div></div>",
            unsafe_allow_html=True)


def slide_synthetic():
    st.markdown(
        "<h2 class='section-h2'>Synthetic Data — Why and How</h2>"
        "<p class='section-sub'>Every distributional parameter in the "
        "10,000-record dataset is calibrated to one of three published "
        "external sources.</p>",
        unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3)
    for col, (clr, icon, name, sub, stats) in zip([sc1, sc2, sc3], sources):
        stats_html = "".join(
            f"<div class='source-stat'>"
            f"<span class='source-stat-label'>{lbl}</span>"
            f"<span class='source-stat-value'>{val}</span></div>"
            for lbl, val in stats
        )
        col.markdown(
            f"<div class='source-card'>"
            f"<div class='source-icon-wrap' style='background:{clr}18;'>{icon}</div>"
            f"<div class='source-name'>{name}</div>"
            f"<div class='source-sub'>{sub}</div>"
            f"{stats_html}</div>",
            unsafe_allow_html=True)

    st.markdown(
        "<div class='criteria-row'>"
        "<div class='criteria-chip'>✅ Calibrated to real statistics</div>"
        "<div class='criteria-chip'>✅ Realistic feature correlations</div>"
        "<div class='criteria-chip'>✅ Correct dropout rate (30%)</div>"
        "<div class='criteria-chip'>✅ Transparent & reproducible</div>"
        "</div>",
        unsafe_allow_html=True)

    st.markdown(
        "<details class='justify-toggle'>"
        "<summary>Full justification & precedent in the literature</summary>"
        "<div class='justify-toggle-body'>"
        "<div class='insight-box' style='margin-top:0;'>"
        "<p class='insight-title'>Why synthetic data was necessary</p>"
        "<p class='insight-body'>"
        "Systematic engagement with public secondary schools in Ibadan, "
        "Oyo State confirmed that most schools do not maintain digitised "
        "student records at the granularity required for this research. "
        "This is consistent with <strong>Okonkwo and Ade-Ibijola (2021)</strong>, "
        "who identified limited digitisation as the primary barrier to "
        "secondary school educational data mining in Nigeria — a barrier "
        "affecting most Sub-Saharan African school systems.<br><br>"
        "Additionally, collecting data from minors in Nigerian secondary "
        "schools requires multi-stage ethics clearance, parental consent, "
        "and school administrative cooperation — a process that typically "
        "exceeds a standard project timeline."
        "</p></div>"
        "<div class='insight-box'>"
        "<p class='insight-title'>Precedent in published research</p>"
        "<p class='insight-body'>"
        "Synthetic dataset generation for educational prediction research "
        "is well-established in the literature:<br><br>"
        "· <strong>Hussain et al. (2023)</strong> used a partially synthetic "
        "dataset in a dropout prediction study, citing data access constraints.<br>"
        "· <strong>Batool et al. (2023)</strong> generated synthetic student "
        "interaction records calibrated to institutional statistics.<br>"
        "· <strong>Romero & Ventura (2020)</strong> explicitly recommend "
        "synthetic generation for educational data mining in contexts where "
        "privacy regulations or institutional barriers prevent real data collection."
        "</p></div>"
        "</div></details>",
        unsafe_allow_html=True)


def slide_intervention():
    st.markdown(
        "<h2 class='section-h2'>Intervention Recommendation Engine</h2>"
        "<p class='section-sub'>The novel contribution of this research. "
        "Most dropout prediction systems stop at the risk score. EduGuard "
        "bridges prediction and prevention.</p>",
        unsafe_allow_html=True)
    ie1, ie2, ie3, ie4 = st.columns(4)
    for col, step, icon, clr, title, body in [
        (ie1, "01", "🎯", BLUE1, "SHAP identifies drivers",
         "For each student prediction, SHAP TreeExplainer computes the contribution "
         "of every feature. The top 3 positive contributors are identified as "
         "the primary risk drivers."),
        (ie2, "02", "🗺️", PURPLE, "Taxonomy lookup",
         "Each driver is matched to the Ojo & Awonuga (2022) evidence-based "
         "intervention taxonomy — 16 features mapped to specific educator actions "
         "across 5 domains."),
        (ie3, "03", "⚖️", AMBER, "Urgency assignment",
         "Each intervention is labelled Urgent, Medium, or Monitor based on "
         "the severity of the risk driver and the directness of its exclusion mechanism."),
        (ie4, "04", "📋", GREEN, "Ranked output",
         "A ranked list of interventions is delivered — Urgent actions first — "
         "each with the responsible party named and a literature rationale cited."),
    ]:
        col.markdown(
            f"<div style='background:#fff;border:1px solid #E0E3FF;"
            f"border-radius:16px;padding:24px 20px;height:100%;"
            f"box-shadow:0 4px 18px rgba(58,78,251,0.07);'>"
            f"<div style='font-size:11px;font-weight:800;color:{clr};"
            f"letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;'>"
            f"Step {step}</div>"
            f"<div style='font-size:28px;margin-bottom:10px;'>{icon}</div>"
            f"<div style='font-size:16px;font-weight:700;color:{DARK};"
            f"margin-bottom:8px;line-height:1.3;'>{title}</div>"
            f"<div style='font-size:14px;color:#6B7280;line-height:1.6;'>{body}</div>"
            f"</div>",
            unsafe_allow_html=True)


def slide_references():
    st.markdown(
        "<h2 class='section-h2'>Academic References</h2>"
        "<p class='section-sub'>Organised by theme for quick lookup during "
        "questions.</p>",
        unsafe_allow_html=True)

    ref_tab1, ref_tab2, ref_tab3, ref_tab4 = st.tabs([
        "Nigerian Context", "ML & Ensemble Methods",
        "Explainability & Interventions", "Dataset Calibration",
    ])

    with ref_tab1:
        for author, text in [
            ("Adeyemo et al., 2023",
             "Gender-disaggregated analysis of secondary school dropout trends "
             "in Nigeria: Evidence from six states. "
             "African Journal of Educational Studies, 21(2), 44–61."),
            ("Aluko et al., 2024",
             "Predicting secondary school student performance using data mining "
             "techniques: Evidence from Ogun State, Nigeria. "
             "Journal of Educational Technology in Developing Countries, 6(1), 12–28."),
            ("Fayomi & Nwosu, 2022",
             "Determinants of secondary school dropout in Nigeria: A multi-state "
             "analysis. International Journal of Educational Development, 88, 102530."),
            ("Okonkwo & Ade-Ibijola, 2021",
             "Chatbots applications in education: A systematic review. "
             "Computers and Education: Artificial Intelligence, 2, 100033."),
            ("Ojo & Awonuga, 2022",
             "An evidence-based framework for school dropout intervention in Nigerian "
             "secondary schools. Nigerian Journal of Educational Research and Policy, "
             "17(1), 88–107."),
            ("Federal Ministry of Education, 2020",
             "National Curriculum Framework for Basic and Senior Secondary Education. "
             "Abuja: Federal Ministry of Education."),
        ]:
            st.markdown(
                f"<div class='ref-item'><span class='ref-tag'>{author}</span>{text}</div>",
                unsafe_allow_html=True)

    with ref_tab2:
        for author, text in [
            ("Batool et al., 2023",
             "Educational data mining to predict students' academic performance: "
             "A survey study. Education and Information Technologies, 28(1), 905–971."),
            ("Hussain et al., 2023",
             "Educational data mining and analysis of students' academic performance "
             "using WEKA. Indonesian Journal of Electrical Engineering and Computer "
             "Science, 9(2), 447–459."),
            ("Mduma et al., 2021",
             "A survey of machine learning approaches and techniques for student "
             "dropout prediction. Data Science in Finance and Economics, 1(2), 109–127."),
            ("Adekitan & Salau, 2022",
             "The impact of engineering students' performance in the first three years "
             "on their graduation result using educational data mining. "
             "PeerJ Computer Science, 5, e189."),
            ("Romero & Ventura, 2020",
             "Educational data mining and learning analytics: An updated survey. "
             "WIREs Data Mining and Knowledge Discovery, 10(3), e1355."),
            ("Kizito, 2023",
             "Early warning systems and dropout prevention in Ugandan secondary "
             "schools: A two-year quasi-experimental evaluation. "
             "International Journal of Educational Research Open, 4, 100231."),
        ]:
            st.markdown(
                f"<div class='ref-item'><span class='ref-tag'>{author}</span>{text}</div>",
                unsafe_allow_html=True)

    with ref_tab3:
        for author, text in [
            ("Dasi & Kanakala, 2022",
             "Cognitive ability assessment for educational dropout prediction: "
             "A machine learning approach. "
             "International Journal of Learning Analytics, 14(3), 211–229."),
            ("Zimmerman, 2002",
             "Becoming a self-regulated learner: An overview. "
             "Theory Into Practice, 41(2), 64–70."),
            ("Bloom et al., 1956",
             "Taxonomy of Educational Objectives: The Classification of Educational "
             "Goals. Handbook I: Cognitive Domain. New York: David McKay Company."),
            ("Gardner, 1983",
             "Frames of Mind: The Theory of Multiple Intelligences. "
             "New York: Basic Books."),
            ("Rumberger & Losen, 2021",
             "The hidden cost of California's harsh school discipline and the "
             "financial case for investing in school climate. "
             "Center for Civil Rights Remedies at the Civil Rights Project, UCLA."),
        ]:
            st.markdown(
                f"<div class='ref-item'><span class='ref-tag'>{author}</span>{text}</div>",
                unsafe_allow_html=True)

    with ref_tab4:
        for author, text in [
            ("Realinho et al., 2022",
             "Predicting student dropout and academic success. "
             "Data, 7(11), 146. UCI Machine Learning Repository. "
             "https://doi.org/10.3390/data7110146"),
            ("UNESCO UIS, 2023",
             "Out-of-school children and youth: Global data and statistics. "
             "Montreal: UNESCO Institute for Statistics."),
            ("World Bank, 2023",
             "Nigeria General Household Survey Panel (GHS-Panel), Wave 4. "
             "Washington DC: World Bank Microdata Library. "
             "https://microdata.worldbank.org/"),
            ("Adeyemo et al., 2023",
             "Gender-disaggregated dropout analysis in Nigeria. "
             "African Journal of Educational Studies, 21(2), 44–61."),
        ]:
            st.markdown(
                f"<div class='ref-item'><span class='ref-tag'>{author}</span>{text}</div>",
                unsafe_allow_html=True)


def slide_resources():
    st.markdown(
        "<h2 class='section-h2'>Resources & Tech Stack</h2>"
        "<p class='section-sub'>The batch template for hands-on demos, and "
        "what the system is built with.</p>",
        unsafe_allow_html=True)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.markdown(
            "<p style='font-size:15px;font-weight:700;color:#252943;"
            "margin-bottom:12px;'>Download Resources</p>",
            unsafe_allow_html=True)
        sample_csv = generate_sample()
        st.download_button(
            "⬇️  SS2 Batch Template — 50 Students (CSV)",
            data=sample_csv.to_csv(index=False),
            file_name="EduGuard_SS2_EGA_Sample.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(
            "50 SS2 students tagged SS2/EGA/001–050 · "
            "realistic risk distribution · "
            "ready to upload to the Batch Dashboard"
        )

    with dl2:
        st.markdown(
            "<p style='font-size:15px;font-weight:700;color:#252943;"
            "margin-bottom:12px;'>Built With</p>",
            unsafe_allow_html=True)
        tech_html = "".join(
            f"<div class='techgrid-item'>"
            f"<div class='techgrid-icon' style='background:{clr}18;'>{icon}</div>"
            f"<div class='techgrid-name'>{name}</div>"
            f"</div>"
            for clr, icon, name in tech
        )
        st.markdown(f"<div class='techgrid'>{tech_html}</div>", unsafe_allow_html=True)


SLIDES = [
    ("🎓 Identity",     slide_identity),
    ("📊 Overview",      slide_stats),
    ("❗ The Problem",   slide_problem),
    ("🏗️ Architecture",  slide_architecture),
    ("📋 Methodology",   slide_methodology),
    ("🗂️ Features",      slide_features),
    ("🚦 Risk Levels",   slide_risk),
    ("🧪 Synthetic Data",slide_synthetic),
    ("⚙️ Interventions", slide_intervention),
    ("📚 References",    slide_references),
    ("🛠️ Resources",     slide_resources),
]

# ══════════════════════════════════════════════════════════════════════
# SLIDE NAVIGATION
# ══════════════════════════════════════════════════════════════════════
if "about_slide" not in st.session_state:
    st.session_state.about_slide = 0

current = st.session_state.about_slide

with st.container():
    st.markdown("<span class='navpill-marker'></span>", unsafe_allow_html=True)
    row1 = st.columns(6)
    row2 = st.columns(6)
    for i, (label, _) in enumerate(SLIDES):
        target_row = row1 if i < 6 else row2
        col = target_row[i if i < 6 else i - 6]
        with col:
            if st.button(label, key=f"navpill_{i}", use_container_width=True,
                         type="primary" if i == current else "secondary"):
                st.session_state.about_slide = i
                st.rerun()

st.markdown(
    f"<div class='slide-progress'>"
    f"<span class='slide-progress-text'>Slide {current+1} of {len(SLIDES)} "
    f"— {SLIDES[current][0]}</span>"
    f"<div class='slide-dots'>" +
    "".join(f"<span class='slide-dot{' done' if i <= current else ''}'></span>"
            for i in range(len(SLIDES))) +
    f"</div></div>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════
# CURRENT SLIDE CONTENT (animated entrance)
# ══════════════════════════════════════════════════════════════════════
with st.container():
    st.markdown("<span class='slide-anim'></span>", unsafe_allow_html=True)
    SLIDES[current][1]()

# ══════════════════════════════════════════════════════════════════════
# PREV / NEXT CONTROLS
# ══════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
with st.container():
    st.markdown("<span class='prevnext-marker'></span>", unsafe_allow_html=True)
    pcol1, pcol2, pcol3 = st.columns([1, 3, 1])
    with pcol1:
        if st.button("‹ Previous", use_container_width=True,
                      disabled=(current == 0)):
            st.session_state.about_slide = max(0, current - 1)
            st.rerun()
    with pcol3:
        if st.button("Next ›", use_container_width=True, type="primary",
                      disabled=(current == len(SLIDES) - 1)):
            st.session_state.about_slide = min(len(SLIDES) - 1, current + 1)
            st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="edu-footer">
  EduGuard — Early Dropout Intelligence System &nbsp;·&nbsp;
  Nzeako Jennifer Nkiruka (DU/C/247) &nbsp;·&nbsp;
  Computer Science &nbsp;·&nbsp;
  Built with Streamlit · XGBoost · SHAP
</div>
""", unsafe_allow_html=True)