"""
=======================================================================
utils/helpers.py
Shared functions used across all pages
=======================================================================
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import joblib
import json
import os

# ── Colours ────────────────────────────────────────────────────────────────────
PURPLE = "#4B3FA0"
TEAL   = "#0F6E56"
RED    = "#991B1B"
AMBER  = "#B45309"
WHITE  = "#FFFFFF"
GL     = "#F5F5F3"

# ── Load all artefacts once (cached globally) ──────────────────────────────────
@st.cache_resource
def load_artefacts():
    a = {}
    a["model"]         = joblib.load("model/best_model.pkl")
    a["scaler"]        = joblib.load("data/scaler.pkl")
    a["feature_cols"]  = joblib.load("data/feature_columns.pkl")
    a["encoding_maps"] = joblib.load("data/encoding_maps.pkl")
    a["cont_cols"]     = joblib.load("data/continuous_cols.pkl")
    a["explainer"]     = joblib.load("model/shap_explainer.pkl")
    a["expected_val"]  = joblib.load("model/shap_expected_value.pkl")
    a["comparison"]    = pd.read_csv("model/model_comparison.csv")
    a["xgb_fi"]        = pd.read_csv(
                             "model/xgb_feature_importance.csv",
                             index_col=0).squeeze()
    a["shap_imp"]      = pd.read_csv(
                             "model/shap_importance.csv",
                             index_col=0).squeeze()
    with open("model/feature_display_names.json") as f:
        a["display_names"] = json.load(f)
    return a


# ── Encoding ───────────────────────────────────────────────────────────────────
ORDINAL_MAPS = {
    "Household_Income_Band":    {"Low":0,"Medium":1,"High":2},
    "Fee_Payment_Status":       {"Unpaid":0,"Partial":1,"Paid":2},
    "Parental_Education_Level": {
        "No Formal Education":0,"Primary":1,"Secondary":2,
        "OND/NCE":3,"HND/BSc":4,"Postgraduate":5,
    },
    "Parental_Involvement":     {"Absent":0,"Passive":1,"Active":2},
    "Learning_Pace":            {"Slow":0,"Average":1,"Fast":2},
    "PE_Participation":         {"Absent":0,"Passive":1,"Active":2},
}
BINARY_MAP  = {"Male":0,"Female":1}
CONT_FEATS  = [
    "Age","Distance_to_School_km","Attendance_Rate_Pct",
    "Avg_Academic_Score_Pct","Failed_Subjects_Count",
    "Cognitive_Assessment_Score","Study_Habit_Score",
    "Practical_Subject_Score","Psychomotor_Dev_Score",
]
YEAR_GROUPS = ["JSS2","JSS3","SS1","SS2","SS3"]


def encode_input(raw: dict, feature_cols, cont_cols, scaler) -> pd.DataFrame:
    """Encode a raw student dict into a model-ready scaled DataFrame."""
    row = {}
    for feat, mapping in ORDINAL_MAPS.items():
        row[feat] = mapping.get(
            raw.get(feat, list(mapping.keys())[0]), 0
        )
    row["Gender"] = BINARY_MAP.get(raw.get("Gender","Male"), 0)
    for feat in CONT_FEATS:
        row[feat] = float(raw.get(feat, 0))
    selected_yg = raw.get("Year_Group","JSS1")
    for yg in YEAR_GROUPS:
        row[f"Year_Group_{yg}"] = 1 if selected_yg == yg else 0

    df = pd.DataFrame([row])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]
    df[cont_cols] = scaler.transform(df[cont_cols])
    return df


def get_unscaled(raw: dict, feature_cols) -> np.ndarray:
    """Return numeric feature values in feature_cols order (unscaled)."""
    lookup = {}
    for feat, mapping in ORDINAL_MAPS.items():
        lookup[feat] = float(mapping.get(raw.get(feat,""), 0))
    lookup["Gender"] = float(BINARY_MAP.get(raw.get("Gender","Male"), 0))
    for feat in CONT_FEATS:
        lookup[feat] = float(raw.get(feat, 0))
    selected_yg = raw.get("Year_Group","JSS1")
    for yg in YEAR_GROUPS:
        lookup[f"Year_Group_{yg}"] = 1.0 if selected_yg == yg else 0.0
    return np.array([lookup.get(c, 0.0) for c in feature_cols], dtype=float)


def predict_student(model, explainer, df: pd.DataFrame):
    """Return (probability, shap_values) for one student."""
    prob      = model.predict_proba(df)[0][1]
    shap_vals = explainer.shap_values(df)[0]
    return prob, shap_vals


# ── Risk classification ────────────────────────────────────────────────────────
def classify_risk(prob: float) -> str:
    if prob >= 0.70:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    return "LOW"


def risk_badge_html(level: str, prob: float) -> str:
    css = {
        "HIGH":   "risk-high",
        "MEDIUM": "risk-medium",
        "LOW":    "risk-low",
    }.get(level, "risk-low")
    em = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}.get(level,"🟢")
    return f'<span class="{css}">{em} {level} RISK — {prob:.1%}</span>'


# ── Charts ─────────────────────────────────────────────────────────────────────
def shap_waterfall_chart(
    shap_vals, unscaled_vals, feature_cols,
    display_names, base_val, prob, title=""
):
    top_idx   = np.argsort(np.abs(shap_vals))[::-1][:12]
    top_shap  = shap_vals[top_idx]
    top_feats = [display_names.get(feature_cols[i], feature_cols[i])
                 for i in top_idx]
    top_vals  = unscaled_vals[top_idx]
    labels    = [f"{f}  =  {v:.1f}" for f, v in zip(top_feats, top_vals)]
    colors    = [RED if s > 0 else TEAL for s in top_shap]

    fig = go.Figure(go.Bar(
        x=top_shap, y=labels, orientation="h",
        marker_color=colors,
        marker_line_color=WHITE, marker_line_width=0.5,
        text=[f"{'+' if s>0 else ''}{s:.3f}" for s in top_shap],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(color=WHITE, size=11, family="Arial"),
    ))
    fig.add_vline(x=0, line_color="#888888",
                  line_dash="dash", line_width=1.5)
    fig.add_vline(
        x=prob - base_val,
        line_color=RED if prob > 0.5 else TEAL,
        line_width=2.5,
        annotation_text=f"Net effect: {prob-base_val:+.3f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=PURPLE)),
        xaxis_title="SHAP Value  (contribution to dropout probability)",
        yaxis_title="",
        height=500,
        margin=dict(l=10, r=20, t=50, b=40),
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        font=dict(family="Arial", size=11),
        showlegend=False,
    )
    return fig


def roc_chart():
    from sklearn.metrics import roc_curve, auc
    colors = {
        "Logistic Regression": AMBER,
        "Random Forest":       TEAL,
        "XGBoost":             PURPLE,
    }
    fig = go.Figure()
    for name, slug in [
        ("Logistic Regression","logistic_regression"),
        ("Random Forest",      "random_forest"),
        ("XGBoost",            "xgboost"),
    ]:
        p           = pd.read_csv(f"model/{slug}_predictions.csv")
        fpr,tpr,_   = roc_curve(p["y_true"], p["y_pred_prob"])
        roc_auc     = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name}  (AUC = {roc_auc:.4f})",
            line=dict(color=colors[name], width=2.5),
        ))
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode="lines",
        name="Random Classifier",
        line=dict(color="#CCCCCC", dash="dash", width=1.5),
    ))
    fig.update_layout(
        title="ROC Curves — All Three Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=440,
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        font=dict(family="Arial", size=11),
        legend=dict(x=0.50, y=0.08, bgcolor="rgba(255,255,255,0.85)"),
    )
    return fig


def confusion_heatmap(slug: str, name: str):
    from sklearn.metrics import confusion_matrix
    p     = pd.read_csv(f"model/{slug}_predictions.csv")
    cm    = confusion_matrix(p["y_true"], p["y_pred"])
    cm_pc = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    lbls  = ["Completed (0)","Dropped Out (1)"]
    txt   = [
        [f"{cm[i][j]:,}<br>({cm_pc[i][j]*100:.1f}%)"
         for j in range(2)]
        for i in range(2)
    ]
    fig = go.Figure(go.Heatmap(
        z=cm_pc, x=lbls, y=lbls,
        colorscale="Blues", showscale=False,
        text=txt, texttemplate="%{text}",
        textfont=dict(size=14, family="Arial"),
    ))
    fig.update_layout(
        title=name,
        xaxis_title="Predicted", yaxis_title="Actual",
        height=340,
        margin=dict(l=10,r=10,t=50,b=40),
        font=dict(family="Arial",size=11),
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
    )
    return fig


# ── Shared CSS injected on every page ─────────────────────────────────────────
SHARED_CSS = """
<style>
  /* Global */
  .main { background-color: #F8F7FF; }
  html, body, [class*="css"] { font-family: Arial, sans-serif; }

  /* Hide default Streamlit chrome */
  #MainMenu  { visibility: hidden; }
  footer     { visibility: hidden; }
  header     { visibility: hidden; }
  [data-testid="stSidebar"]        { display: none; }
  [data-testid="collapsedControl"] { display: none; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
  }

  /* Section header strip */
  .section-header {
    background: linear-gradient(90deg, #4B3FA0, #0F6E56);
    color: white;
    padding: 13px 22px;
    border-radius: 10px;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 20px;
  }

  /* Risk badges */
  .risk-high {
    background:#FEE2E2; color:#991B1B;
    padding:6px 20px; border-radius:20px;
    font-weight:700; font-size:17px; display:inline-block;
  }
  .risk-medium {
    background:#FEF3C7; color:#B45309;
    padding:6px 20px; border-radius:20px;
    font-weight:700; font-size:17px; display:inline-block;
  }
  .risk-low {
    background:#E1F5EE; color:#0F6E56;
    padding:6px 20px; border-radius:20px;
    font-weight:700; font-size:17px; display:inline-block;
  }

  /* Driver chip */
  .driver-chip {
    background:#EAE8F8; color:#4B3FA0;
    padding:2px 10px; border-radius:12px;
    font-size:12px; font-weight:600;
    display:inline-block; margin-bottom:4px;
  }

  /* Intervention cards */
  .interv-urgent {
    border-left:5px solid #991B1B; background:#FEE2E2;
    padding:14px 18px; border-radius:8px; margin-bottom:12px;
  }
  .interv-medium {
    border-left:5px solid #B45309; background:#FEF3C7;
    padding:14px 18px; border-radius:8px; margin-bottom:12px;
  }
  .interv-monitor {
    border-left:5px solid #0F6E56; background:#E1F5EE;
    padding:14px 18px; border-radius:8px; margin-bottom:12px;
  }

  /* Stat card */
  .stat-card {
    background:#FFFFFF; border:1px solid #E0E0E0;
    border-radius:10px; padding:18px 20px;
    box-shadow:0 2px 6px rgba(0,0,0,0.06);
    text-align:center; margin-bottom:12px;
  }
  .stat-number {
    font-size:28px; font-weight:700; color:#4B3FA0;
  }
  .stat-label {
    font-size:13px; color:#555555; margin-top:4px;
  }

  /* Footer */
  .footer {
    text-align:center; color:#888888;
    font-size:12px; padding:20px 0 10px;
    border-top:1px solid #EEEEEE; margin-top:40px;
  }
</style>
"""


def inject_css():
    """Call at the top of every page to apply shared styles."""
    st.markdown(SHARED_CSS, unsafe_allow_html=True)


def navbar():
    """Render the top navigation bar on every page."""
    st.markdown("""
    <div style="
      background: linear-gradient(90deg,#1A1A2E 0%,#4B3FA0 60%,#0F6E56 100%);
      padding:16px 32px; border-radius:12px; margin-bottom:6px;
      display:flex; align-items:center; justify-content:space-between;
    ">
      <div style="display:flex; align-items:center; gap:14px;">
        <span style="font-size:32px;">🎓</span>
        <div>
          <div style="color:#FFF;font-size:22px;font-weight:700;
                      letter-spacing:0.5px;line-height:1.2;">
            MnemoniQ
          </div>
          <div style="color:#AABBCC;font-size:12px;margin-top:2px;">
            Dropout Risk Prediction &amp; Intervention Recommendation System
          </div>
        </div>
      </div>
      <div style="text-align:right;">
        <div style="color:#FFF;font-size:13px;font-weight:600;">
          Nzeako Jennifer Nkiruka &nbsp;|&nbsp; DU/C/247
        </div>
        <div style="color:#AABBCC;font-size:11px;margin-top:2px;">
          Model: XGBoost &nbsp;|&nbsp; Dataset: 10,000 records &nbsp;|&nbsp;
          MSc Computer Science
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def footer():
    st.markdown("""
    <div class="footer">
      MnemoniQ Dropout Risk Prediction System &nbsp;|&nbsp;
      Nzeako Jennifer Nkiruka (DU/C/247) &nbsp;|&nbsp;
      MSc Computer Science &nbsp;|&nbsp;
      Built with Streamlit · XGBoost · SHAP
    </div>
    """, unsafe_allow_html=True)


# ── Sample batch data ──────────────────────────────────────────────────────────
SAMPLE_BATCH = pd.DataFrame({
    "Student_ID":              ["STU-001","STU-002","STU-003",
                                "STU-004","STU-005","STU-006"],
    "Gender":                  ["Female","Male","Female",
                                "Male","Female","Male"],
    "Age":                     [15,14,16,13,17,15],
    "Year_Group":              ["SS1","JSS3","SS2","JSS1","SS3","SS2"],
    "Distance_to_School_km":   [12.0,2.5,18.0,1.0,9.0,5.5],
    "Household_Income_Band":   ["Low","Medium","Low","High","Low","Medium"],
    "Fee_Payment_Status":      ["Unpaid","Paid","Partial","Paid","Unpaid","Partial"],
    "Parental_Education_Level":["Primary","Secondary","No Formal Education",
                                "HND/BSc","Primary","OND/NCE"],
    "Parental_Involvement":    ["Absent","Active","Absent","Active","Passive","Passive"],
    "Attendance_Rate_Pct":     [45.0,88.0,38.0,92.0,55.0,70.0],
    "Avg_Academic_Score_Pct":  [38.0,72.0,32.0,85.0,45.0,58.0],
    "Failed_Subjects_Count":   [4,0,5,0,3,2],
    "Cognitive_Assessment_Score":[32.0,70.0,28.0,82.0,40.0,55.0],
    "Learning_Pace":           ["Slow","Fast","Slow","Fast","Average","Average"],
    "Study_Habit_Score":       [30.0,78.0,25.0,85.0,42.0,55.0],
    "Practical_Subject_Score": [42.0,68.0,35.0,75.0,50.0,60.0],
    "PE_Participation":        ["Absent","Active","Absent","Active","Passive","Active"],
    "Psychomotor_Dev_Score":   [35.0,72.0,28.0,80.0,48.0,62.0],
})