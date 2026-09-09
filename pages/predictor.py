import time
from contextlib import contextmanager

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import (
    load_artefacts, encode_input, get_unscaled,
    predict_student, shap_waterfall_chart,
)
from utils.interventions import get_interventions

st.set_page_config(
    page_title="EduGuard — Student Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Palette (matches the EduGuard landing page) ─────────────────────────────
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

DOMAIN_COLORS = {
    "demographic": BLUE1, "socioeconomic": PURPLE, "academic": BLUE2,
    "cognitive": AMBER, "psychomotor": GREEN,
}

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
        number={"suffix": "%", "font": {"size": 48, "color": clr, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": BORDER,
                     "tickfont": {"size": 12, "color": "#9CA3AF"}},
            "bar": {"color": clr, "thickness": 0.32},
            "bgcolor": CARD,
            "borderwidth": 0,
        },
    ))
    fig.update_layout(
        height=190,
        margin=dict(l=26, r=26, t=18, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
*,html,body,[class*="css"]{font-family:'Inter',sans-serif!important;box-sizing:border-box;}
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
   segmented control options, dataframe cells) renders at browser default
   (~14px) unless told otherwise — bump the shared baseline across the page. */
[data-testid="stWidgetLabel"] p{font-size:20px!important;font-weight:600!important;}
[data-testid="stMarkdownContainer"] p{font-size:20px;}
[data-testid="stDataFrame"] *{font-size:19px!important;}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"] p{font-size:19px!important;}
[data-testid="stSelectbox"] *{font-size:19px!important;}
[data-testid="stTextInput"] input{font-size:19px!important;}

/* ── Navbar (shared look with the EduGuard home page) ─────────────────── */
.edu-nav{position:sticky;top:0;z-index:9999;background:rgba(37,41,67,0.92);
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  margin:0 -56px;
  padding:0 56px;display:flex;align-items:center;justify-content:space-between;
  min-height:72px;border-bottom:1px solid rgba(255,255,255,0.07);}
.edu-brand{display:flex;align-items:center;gap:13px;text-decoration:none!important;}
.edu-brand-icon{background:linear-gradient(135deg,#3A4EFB,#33A4FA);width:42px;height:42px;
  border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:21px;
  box-shadow:0 6px 16px rgba(58,78,251,0.35);}
.edu-brand-name{color:#fff;font-size:25px;font-weight:900;letter-spacing:-0.5px;line-height:1.1;}
.edu-brand-sub{color:#E3FF3B;font-size:8px;font-weight:700;letter-spacing:2px;
  text-transform:uppercase;margin-top:2px;}
.edu-navlinks{display:flex;align-items:center;gap:2px;}
.edu-navlinks a{color:rgba(222,224,237,0.72)!important;font-size:17px;font-weight:500;
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
.page-sub{font-size:20px;color:#6B7280;margin:0 0 18px;line-height:1.65;max-width:640px;}
.context-badges{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px;}
.context-badge{background:#fff;border:1px solid #E0E3FF;color:#374151;
  padding:7px 15px;border-radius:20px;font-size:19px;font-weight:600;}

/* ── Form shell ─────────────────────────────────────────────────────── */
[data-testid="stForm"]{background:transparent;border:none;padding:0;}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px!important;}
.domain-card-head{display:flex;align-items:center;gap:10px;margin-bottom:14px;
  padding-bottom:12px;border-bottom:1px solid #EEF0FA;}
.domain-card-icon{width:34px;height:34px;border-radius:10px;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;font-size:22px;}
.domain-card-title{font-size:21px;font-weight:800;letter-spacing:0.1px;}

[data-testid="stButtonGroup"] button[data-variant="segmented_control"]{
  border-radius:9px!important;border-color:#E0E3FF!important;
  background:#fff!important;font-weight:600!important;color:#374151!important;
  transition:background .15s ease, border-color .15s ease;}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"] p{color:inherit!important;}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"]{
  background:#3A4EFB!important;border-color:#3A4EFB!important;color:#fff!important;
  box-shadow:0 4px 12px rgba(58,78,251,0.28);}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"]:hover{
  border-color:#3A4EFB!important;}

/* Slider fill track: BaseWeb paints it as a dynamically-generated
   red-to-transparent gradient (a new emotion class per value, so it can't
   be targeted by class name) — hue-rotate reliably repaints it brand-blue
   regardless of the current value. Thumb + value bubble use a plain
   background-color, so those get a direct override instead. */
[data-testid="stSlider"] [role="group"]>div>div:nth-child(1){
  filter:hue-rotate(234deg) saturate(1.05);}
[data-testid="stSlider"] [role="group"]>div>div:nth-child(2){
  background-color:#3A4EFB!important;border-color:#3A4EFB!important;}
[data-testid="stSlider"] [role="group"]>div>div:nth-child(2) input{
  background-color:#3A4EFB!important;}
/* the floating value bubble must stay transparent — it sits on top of the
   thumb, so giving it a solid fill (even matching blue) hides its own text */
[data-testid="stSliderThumbValue"]{background-color:transparent!important;}
[data-testid="stSliderThumbValue"] p{color:#3A4EFB!important;font-weight:700!important;font-size:19px!important;}

button[data-testid^="stBaseButton-primary"]{
  background:linear-gradient(90deg,#3A4EFB,#33A4FA)!important;
  color:#fff!important;border:none!important;border-radius:12px!important;
  font-weight:700!important;font-size:21px!important;padding:14px 28px!important;
  box-shadow:0 10px 26px rgba(58,78,251,0.25)!important;transition:transform .15s ease;}
button[data-testid^="stBaseButton-primary"] p{color:#fff!important;font-weight:700!important;font-size:21px!important;}
button[data-testid^="stBaseButton-primary"]:hover{transform:translateY(-2px);}
button[data-testid^="stBaseButton-secondary"]{border-radius:10px!important;
  font-weight:600!important;border-color:#E0E3FF!important;}
button[data-testid^="stBaseButton-secondary"] p{font-size:20px!important;}

.section-divider{height:1px;background:#E0E3FF;margin:32px 0 28px;border:none;}
.section-h3{font-size:27px;font-weight:800;color:#252943!important;
  letter-spacing:-0.3px;margin:0 0 6px!important;padding:0!important;}
.section-caption{font-size:19px;color:#6B7280;margin:0 0 18px;}

/* ── Result hero ────────────────────────────────────────────────────── */
.result-id{font-size:18px;color:#6B7280;font-weight:700;text-transform:uppercase;
  letter-spacing:0.8px;margin:0 0 8px;}
.result-level-badge{display:inline-block;padding:6px 16px;border-radius:20px;
  font-size:19px;font-weight:800;letter-spacing:0.3px;margin-bottom:14px;}
.result-summary{font-size:20px;color:#374151;line-height:1.7;max-width:520px;margin:0;}
.mini-card{background:#fff;border:1px solid #E0E3FF;padding:16px 18px;
  border-radius:14px;text-align:center;margin-bottom:12px;
  box-shadow:0 4px 14px rgba(58,78,251,0.06);}
.mini-card-num{font-size:37px;font-weight:900;letter-spacing:-1px;line-height:1;}
.mini-card-label{font-size:17px;color:#6B7280;font-weight:500;margin-top:5px;}

.tip-banner{background:#F4F5FF;border:1px dashed #C7CCF5;border-radius:12px;
  padding:14px 18px;font-size:19px;color:#3A4EFB;font-weight:600;margin:24px 0 8px;}

/* ── Tabs ───────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid #E0E3FF;}
[data-testid="stTabs"] [data-baseweb="tab"]{
  font-weight:700!important;font-size:20px!important;color:#6B7280!important;
  padding:10px 4px!important;}
[data-testid="stTabs"] [aria-selected="true"]{color:#3A4EFB!important;}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{background-color:#3A4EFB!important;
  height:3px!important;border-radius:3px 3px 0 0;}

/* ── Entrance animation shared by driver + intervention cards ─────────── */
@keyframes eduCardIn{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:translateY(0);}}

/* ── Driver cards (grid-aligned) ───────────────────────────────────── */
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
  display:flex;align-items:center;gap:12px;font-weight:800;font-size:20px;
  user-select:none;}
.interv-group-summary::-webkit-details-marker{display:none;}
.interv-group-summary::after{content:"›";margin-left:auto;font-size:25px;
  color:#9CA3AF;transition:transform .2s ease;}
.interv-group[open]>.interv-group-summary::after{transform:rotate(90deg);}
.interv-group-badge{font-size:12.5px;font-weight:800;padding:3px 11px;border-radius:20px;}
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
  font-size:21px;font-weight:600;border:1px solid #E0E3FF;}
.interv-party{font-size:21px;color:#9CA3AF;}
.interv-action{font-size:28px;color:#252943;font-weight:500;
  margin:0 0 10px;line-height:1.5;flex-grow:1;}
.rationale-toggle{margin-top:auto;padding-top:10px;border-top:1px solid rgba(0,0,0,0.08);}
.rationale-toggle summary{cursor:pointer;font-size:18px;font-weight:700;
  color:#3A4EFB;list-style:none;display:flex;align-items:center;gap:6px;user-select:none;}
.rationale-toggle summary::-webkit-details-marker{display:none;}
.rationale-toggle summary::after{content:"›";margin-left:auto;transition:transform .2s ease;}
.rationale-toggle[open] summary::after{transform:rotate(90deg);}
.rationale-toggle-body{font-size:18px;color:#6B7280;line-height:1.6;padding-top:8px;}

.interv-legend{display:flex;gap:18px;flex-wrap:wrap;margin:14px 0 18px;}
.interv-legend-item{display:flex;align-items:center;gap:7px;font-size:23px;
  color:#6B7280;font-weight:500;}
.interv-legend-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0;}

/* ── Footer ─────────────────────────────────────────────────────────── */
.edu-footer{background:#252943;padding:32px 56px 24px;margin:56px -56px 0;}
.edu-footer-credit{color:#7A7F9A;font-size:17px;text-align:center;line-height:1.6;}

@media(max-width:768px){
  .block-container{padding:0 20px!important;}
  .edu-nav{margin:0 -20px;padding:0 20px;}.edu-navlinks{display:none;}
  .edu-footer{margin:56px -20px 0;padding:32px 20px 24px;}
  .page-wrap{padding:28px 0 8px;}
  .page-h1{font-size:33px;}
  .mini-card-num{font-size:30px;}
  .interv-grid{grid-template-columns:1fr;}
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
    <a href="/app/pages/predictor.py" class="active">Student Predictor</a>
    <a href="/batch">Batch Dashboard</a>
    <a href="/performance">Model Performance</a>
    <a href="/about">About</a>
  </div>
  <div class="edu-live"><span class="edu-live-dot"></span>Live</div>
</nav>
""", unsafe_allow_html=True)

# ── LOAD ARTEFACTS ─────────────────────────────────────────────────────────────
A             = load_artefacts()
model         = A["model"]
scaler        = A["scaler"]
feature_cols  = A["feature_cols"]
cont_cols     = A["cont_cols"]
explainer     = A["explainer"]
display_names = A["display_names"]
expected_val  = A["expected_val"]

if "session_log" not in st.session_state:
    st.session_state.session_log = []

# ── PAGE HEADER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-wrap">
  <div class="page-header">
    <p class="page-eyebrow">Single Student Assessment</p>
    <h1 class="page-h1">Student Risk Predictor</h1>
    <p class="page-sub">
      Enter the student's details across all five domains. EduGuard will
      predict their dropout probability, explain which factors are driving
      the risk, and deliver ranked interventions for school staff.
    </p>
    <div class="context-badges">
      <div class="context-badge">🗂️ 5 domains</div>
      <div class="context-badge">🧬 17 inputs</div>
      <div class="context-badge">⏱️ ~2 minutes</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── INPUT FORM ─────────────────────────────────────────────────────────────────
with padded():
    with st.form("predictor_form"):

        row1 = st.columns(3)

        with row1[0]:
            with st.container(border=True):
                st.markdown(
                    f"<div class='domain-card-head'>"
                    f"<div class='domain-card-icon' style='background:{BLUE1}18;'>👤</div>"
                    f"<div class='domain-card-title' style='color:{BLUE1};'>Demographic</div>"
                    f"</div>", unsafe_allow_html=True)
                student_id = st.text_input("🆔 Student ID", value="STU-001",
                                            help="Anonymised student reference")
                gender = st.segmented_control("Gender", ["Male", "Female"], default="Male")
                age = st.slider("🎂 Age", 11, 20, 14)
                year_group = st.segmented_control(
                    "Year Group", ["JSS1", "JSS2", "JSS3", "SS1", "SS2", "SS3"],
                    default="JSS1")
                distance = st.slider("📍 Distance to School (km)",
                                      0.5, 35.0, 3.0, step=0.5)

        with row1[1]:
            with st.container(border=True):
                st.markdown(
                    f"<div class='domain-card-head'>"
                    f"<div class='domain-card-icon' style='background:{PURPLE}18;'>💰</div>"
                    f"<div class='domain-card-title' style='color:{PURPLE};'>Socioeconomic</div>"
                    f"</div>", unsafe_allow_html=True)
                income = st.segmented_control("Household Income Band",
                                               ["Low", "Medium", "High"], default="Low")
                fee_status = st.segmented_control("Fee Payment Status",
                                                   ["Unpaid", "Partial", "Paid"], default="Unpaid")
                parent_edu = st.selectbox("Parental Education Level",
                                           ["No Formal Education", "Primary", "Secondary",
                                            "OND/NCE", "HND/BSc", "Postgraduate"])
                parent_inv = st.segmented_control("Parental Involvement",
                                                   ["Absent", "Passive", "Active"], default="Passive")

        with row1[2]:
            with st.container(border=True):
                st.markdown(
                    f"<div class='domain-card-head'>"
                    f"<div class='domain-card-icon' style='background:{BLUE2}18;'>📚</div>"
                    f"<div class='domain-card-title' style='color:{BLUE2};'>Academic</div>"
                    f"</div>", unsafe_allow_html=True)
                attendance = st.slider("✅ Attendance Rate (%)", 0, 100, 75)
                acad_score = st.slider("📈 Avg Academic Score (%)", 0, 100, 50)
                failed_subj = st.slider("❌ Failed Subjects", 0, 8, 1)

        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        row2 = st.columns(2)

        with row2[0]:
            with st.container(border=True):
                st.markdown(
                    f"<div class='domain-card-head'>"
                    f"<div class='domain-card-icon' style='background:{AMBER}18;'>🧠</div>"
                    f"<div class='domain-card-title' style='color:{AMBER};'>Cognitive</div>"
                    f"</div>", unsafe_allow_html=True)
                cog_score = st.slider("🧩 Cognitive Assessment Score", 0, 100, 55)
                learning_pace = st.segmented_control("Learning Pace",
                                                      ["Slow", "Average", "Fast"], default="Average")
                study_habit = st.slider("📖 Study Habit Score", 0, 100, 55)

        with row2[1]:
            with st.container(border=True):
                st.markdown(
                    f"<div class='domain-card-head'>"
                    f"<div class='domain-card-icon' style='background:{GREEN}18;'>🏃</div>"
                    f"<div class='domain-card-title' style='color:{GREEN};'>Psychomotor</div>"
                    f"</div>", unsafe_allow_html=True)
                practical = st.slider("🔧 Practical Subject Score", 0, 100, 58)
                pe_part = st.segmented_control("PE Participation",
                                                ["Absent", "Passive", "Active"], default="Passive")
                psychomotor = st.slider("🤸 Psychomotor Development Score", 0, 100, 55)

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🔍  Analyse Dropout Risk",
            use_container_width=True,
            type="primary",
        )


# ── PREDICTION OUTPUT ──────────────────────────────────────────────────────────
if submitted:
    # st.segmented_control toggles a pill OFF (returning None) if a user
    # clicks the already-selected option — fall back to the field's default
    # so a stray click never sends None into encode_input()/get_unscaled().
    raw = {
        "Gender": gender or "Male", "Age": age,
        "Year_Group": year_group or "JSS1",
        "Distance_to_School_km": distance,
        "Household_Income_Band": income or "Low",
        "Fee_Payment_Status": fee_status or "Unpaid",
        "Parental_Education_Level": parent_edu,
        "Parental_Involvement": parent_inv or "Passive",
        "Attendance_Rate_Pct": attendance,
        "Avg_Academic_Score_Pct": acad_score,
        "Failed_Subjects_Count": failed_subj,
        "Cognitive_Assessment_Score": cog_score,
        "Learning_Pace": learning_pace or "Average",
        "Study_Habit_Score": study_habit,
        "Practical_Subject_Score": practical,
        "PE_Participation": pe_part or "Passive",
        "Psychomotor_Dev_Score": psychomotor,
    }

    with st.status("Analysing student profile...", expanded=True) as status:
        st.write("📥 Encoding student profile...")
        time.sleep(0.35)
        encoded = encode_input(raw, feature_cols, cont_cols, scaler)

        st.write("🤖 Running XGBoost ensemble model...")
        time.sleep(0.35)
        unscaled = get_unscaled(raw, feature_cols)
        prob, shap_v = predict_student(model, explainer, encoded)

        st.write("🔍 Computing SHAP attributions...")
        time.sleep(0.3)

        st.write("📋 Ranking interventions...")
        time.sleep(0.3)
        result = get_interventions(
            shap_vals=shap_v,
            feature_vals=unscaled,
            feature_names=list(feature_cols),
            risk_score=prob,
            display_names=display_names,
            top_n_drivers=3,
            max_interventions_per_driver=3,
        )
        status.update(
            label=f"Analysis complete — {result['risk_level']} risk identified",
            state="complete", expanded=False,
        )

    level = result["risk_level"]
    urgent_count = sum(1 for i in result["interventions"] if "URGENT" in i["urgency"])
    clr = RED if level == "HIGH" else AMBER if level == "MEDIUM" else GREEN
    level_emoji = "🔴" if level == "HIGH" else "🟡" if level == "MEDIUM" else "🟢"

    with padded():
        st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)

        gcol, mcol, ccol = st.columns([1, 2, 1])
        with gcol:
            st.plotly_chart(risk_gauge_chart(prob, clr), use_container_width=True,
                             config={"displayModeBar": False})
        with mcol:
            st.markdown(
                f"<div class='result-id'>Assessment — {student_id}</div>"
                f"<div class='result-level-badge' "
                f"style='background:{clr}18;color:{clr};border:1px solid {clr}44;'>"
                f"{level_emoji} {level} RISK</div>"
                f"<p class='result-summary'>{result['summary']}</p>",
                unsafe_allow_html=True)
        with ccol:
            st.markdown(
                f"<div class='mini-card'>"
                f"<div class='mini-card-num' style='color:{RED};'>{urgent_count}</div>"
                f"<div class='mini-card-label'>Urgent actions</div></div>"
                f"<div class='mini-card'>"
                f"<div class='mini-card-num' style='color:{BLUE1};'>"
                f"{len(result['interventions'])}</div>"
                f"<div class='mini-card-label'>Total interventions</div></div>",
                unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(
            ["📊  Explanation", "🔍  Top Drivers", "📋  Interventions"])

        with tab1:
            st.markdown(
                f"<p style='font-size:14px;color:#6B7280;margin:14px 0 6px;'>"
                f"<span style='color:{RED};font-weight:600;'>Red bars</span> "
                f"increase dropout risk &nbsp;·&nbsp; "
                f"<span style='color:{GREEN};font-weight:600;'>Green bars</span> "
                f"decrease dropout risk</p>",
                unsafe_allow_html=True)
            fig = shap_waterfall_chart(
                shap_v, unscaled, feature_cols, display_names,
                expected_val, prob,
                title=f"SHAP Feature Contributions — {student_id}",
            )
            fig.update_layout(
                plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                font=dict(family="Inter", size=13),
            )
            st.plotly_chart(fig, use_container_width=True,
                             config={"displayModeBar": False})

        with tab2:
            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
            top3 = result["top_drivers"][:3]
            max_abs = max((abs(d["shap_val"]) for d in top3), default=0) or 1
            d1c, d2c, d3c = st.columns(3)
            for i, (col, driver) in enumerate(zip([d1c, d2c, d3c], top3)):
                sign = "+" if driver["shap_val"] > 0 else ""
                dclr = RED if driver["shap_val"] > 0 else GREEN
                direction = "↑ increases risk" if driver["shap_val"] > 0 else "↓ reduces risk"
                # A top driver can be a categorical feature (e.g. Fee_Payment_Status
                # = "Unpaid"), not just a continuous score — guard the numeric
                # format so a string value never crashes the card.
                fv = driver["feat_val"]
                fv_display = f"{fv:.1f}" if isinstance(fv, (int, float)) else str(fv)
                pct = round(abs(driver["shap_val"]) / max_abs * 100, 1)
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
                    f"<div class='driver-bar-fill' style='width:{pct}%;background:{dclr};'></div>"
                    f"</div>"
                    f"<div class='driver-val'>Student value: {fv_display}</div>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True)

        with tab3:
            st.markdown(
                "<p style='font-size:19px;color:#6B7280;margin:14px 0 4px;'>"
                "Grouped by urgency and linked to dominant risk drivers · "
                "Based on Ojo &amp; Awonuga (2022) Nigerian school intervention taxonomy. "
                "Click a group to expand it, and a card's "
                "<strong>Why this helps</strong> to see the rationale.</p>",
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

            # Group interventions by urgency band so the page doesn't dump
            # every action at once — each band is its own collapsible
            # accordion section, open by default only for the highest
            # urgency band that actually has items in it.
            band_order = ["URGENT", "MEDIUM", "MONITOR"]
            band_labels = {"URGENT": "Urgent", "MEDIUM": "Medium priority", "MONITOR": "Monitor"}
            band_icons = {"URGENT": "🔴", "MEDIUM": "🟡", "MONITOR": "🟢"}
            band_colors = {"URGENT": RED, "MEDIUM": AMBER, "MONITOR": GREEN}
            band_css = {"URGENT": "interv-urgent", "MEDIUM": "interv-medium", "MONITOR": "interv-monitor"}

            grouped = {b: [] for b in band_order}
            for interv in result["interventions"]:
                for b in band_order:
                    if b in interv["urgency"]:
                        grouped[b].append(interv)
                        break

            first_open = next((b for b in band_order if grouped[b]), None)

            group_blocks = []
            for b in band_order:
                items = grouped[b]
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
                open_attr = "open" if b == first_open else ""
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
            "<div class='tip-banner'>💡 Tip: adjust the values above and click "
            "<strong>Analyse Dropout Risk</strong> again to see how the "
            "prediction changes for the same student.</div>",
            unsafe_allow_html=True)

    st.session_state.session_log.append({
        "Student_ID": student_id,
        "Risk_Score_%": round(prob * 100, 1),
        "Risk_Level": level,
        "Top_Driver": (result["top_drivers"][0]["display"]
                       if result["top_drivers"] else "N/A"),
        "Urgent_Actions": urgent_count,
    })


# ── SESSION HISTORY (always visible once at least one assessment exists) ──────
if st.session_state.session_log:
    log_df = pd.DataFrame(st.session_state.session_log)
    n_assessed = len(log_df)
    n_high = int((log_df["Risk_Level"] == "HIGH").sum())

    with padded():
        st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
        st.markdown(
            f"<h3 class='section-h3'>🗂️ Session History</h3>"
            f"<p class='section-caption'>{n_assessed} student(s) assessed this "
            f"session · {n_high} flagged high risk</p>",
            unsafe_allow_html=True)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
        if st.button("🗑️  Clear Session History"):
            st.session_state.session_log = []
            st.rerun()


# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="edu-footer">
  <div class="edu-footer-credit">
    EduGuard — Early Dropout Intelligence System &nbsp;·&nbsp;
    Nzeako Jennifer Nkiruka (DU/C/247) &nbsp;·&nbsp;
    MSc Computer Science
  </div>
</div>
""", unsafe_allow_html=True)