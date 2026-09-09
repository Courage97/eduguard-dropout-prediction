"""
=======================================================================
Step 5 — SHAP Explainability
Dropout Risk Prediction System
Nzeako Jennifer Nkiruka | DU/C/247
=======================================================================
What this script does:
  1.  Loads the trained XGBoost model and test data
  2.  Computes SHAP values using TreeExplainer (fast, exact)
  3.  Generates and saves:
        - Global summary plot (beeswarm)
        - Global bar chart (mean absolute SHAP)
        - SHAP dependence plots for top 5 features
        - Waterfall chart for a high-risk student example
        - Waterfall chart for a low-risk student example
        - Force plot for high-risk student (HTML)
        - SHAP values saved as CSV for use in Streamlit app
  4.  Prints top feature contributions

All charts saved to: assets/charts/shap/
SHAP values saved to: model/shap_values.npy
                      model/shap_expected_value.pkl

Run:  python shap_analysis.py
=======================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import shap
import os
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = "data"
MODEL_DIR = "model"
CHART_DIR = "assets/charts/shap"
os.makedirs(CHART_DIR, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
PURPLE = "#4B3FA0"
TEAL   = "#0F6E56"
RED    = "#991B1B"
AMBER  = "#B45309"
WHITE  = "#FFFFFF"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   PURPLE,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  WHITE,
    "axes.facecolor":    WHITE,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": WHITE,
})

print("=" * 65)
print("  Step 5 — SHAP Explainability")
print("  Dropout Risk Prediction System — Jennifer Nkiruka")
print("=" * 65)

chart_count = 0

def save_chart(fig, filename, title):
    global chart_count
    chart_count += 1
    path = f"{CHART_DIR}/{filename}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [{chart_count:02d}] Saved: {filename}  — {title}")


# ══════════════════════════════════════════════════════════════════════
# 5.1  LOAD MODEL AND DATA
# ══════════════════════════════════════════════════════════════════════
print("\n[5.1] Loading XGBoost model and test data...")

xgb_model    = joblib.load(f"{MODEL_DIR}/xgboost.pkl")
feature_cols = joblib.load(f"{DATA_DIR}/feature_columns.pkl")
encoding_maps = joblib.load(f"{DATA_DIR}/encoding_maps.pkl")

X_test          = pd.read_csv(f"{DATA_DIR}/X_test.csv")
X_test_unscaled = pd.read_csv(f"{DATA_DIR}/X_test_unscaled.csv")
y_test          = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()
predictions     = pd.read_csv(f"{MODEL_DIR}/xgboost_predictions.csv")

print(f"      Test set:       {X_test.shape[0]:,} rows x {X_test.shape[1]} features")
print(f"      Dropout rate:   {y_test.mean()*100:.1f}%")

# Clean feature display names for charts
def clean_name(name):
    replacements = {
        "Household_Income_Band":     "Income Band",
        "Fee_Payment_Status":        "Fee Payment Status",
        "Parental_Education_Level":  "Parental Education",
        "Parental_Involvement":      "Parental Involvement",
        "Attendance_Rate_Pct":       "Attendance Rate (%)",
        "Avg_Academic_Score_Pct":    "Academic Score (%)",
        "Failed_Subjects_Count":     "Failed Subjects",
        "Cognitive_Assessment_Score":"Cognitive Score",
        "Learning_Pace":             "Learning Pace",
        "Study_Habit_Score":         "Study Habit Score",
        "Practical_Subject_Score":   "Practical Score",
        "PE_Participation":          "PE Participation",
        "Psychomotor_Dev_Score":     "Psychomotor Score",
        "Distance_to_School_km":     "Distance (km)",
        "Gender":                    "Gender",
        "Age":                       "Age",
    }
    for key in replacements:
        if key in name:
            return replacements[key]
    # Handle one-hot encoded year group columns
    if "Year_Group_" in name:
        return name.replace("Year_Group_", "Year Group: ")
    return name.replace("_", " ")

clean_names = [clean_name(c) for c in feature_cols]


# ══════════════════════════════════════════════════════════════════════
# 5.2  COMPUTE SHAP VALUES
# ══════════════════════════════════════════════════════════════════════
print("\n[5.2] Computing SHAP values using TreeExplainer...")
print("      This may take 60-120 seconds for 2,000 test samples...")

explainer    = shap.TreeExplainer(xgb_model)
shap_values  = explainer.shap_values(X_test)

# For binary classification XGBoost returns a 2D array
# Each row = one student, each column = one feature's SHAP contribution
print(f"      SHAP values shape: {shap_values.shape}")
print(f"      Expected value (base rate): {explainer.expected_value:.4f}")

# Save SHAP values for use in Streamlit app
np.save(f"{MODEL_DIR}/shap_values.npy", shap_values)
joblib.dump(explainer.expected_value, f"{MODEL_DIR}/shap_expected_value.pkl")
joblib.dump(explainer, f"{MODEL_DIR}/shap_explainer.pkl")
print(f"      Saved → model/shap_values.npy")
print(f"      Saved → model/shap_expected_value.pkl")
print(f"      Saved → model/shap_explainer.pkl")

# Mean absolute SHAP per feature
mean_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.Series(mean_shap, index=feature_cols).sort_values(
    ascending=False
)
shap_importance.index = [clean_name(c) for c in shap_importance.index]
shap_importance.to_csv(f"{MODEL_DIR}/shap_importance.csv", header=["mean_abs_shap"])

print(f"\n      Top 10 features by mean |SHAP| value:")
print(f"      {'Feature':<35} {'Mean |SHAP|':>12}")
print(f"      {'-'*48}")
for feat, val in shap_importance.head(10).items():
    bar = "█" * int(val * 300)
    print(f"      {feat:<35} {val:>10.4f}  {bar}")


# ══════════════════════════════════════════════════════════════════════
# 5.3  CHART S1 — GLOBAL SHAP SUMMARY (Beeswarm)
# ══════════════════════════════════════════════════════════════════════
print("\n[5.3] Generating SHAP beeswarm summary plot...")

# Use a DataFrame with clean names for display
X_test_display = X_test.copy()
X_test_display.columns = clean_names

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X_test_display,
    plot_type="dot",
    max_display=15,
    show=False,
    color_bar=True,
)
plt.title(
    "Figure S1: SHAP Summary Plot — Global Feature Impact\n"
    "(Each dot = one student | Colour = feature value | "
    "x-axis = impact on dropout prediction)",
    fontsize=11, color=PURPLE, pad=12
)
plt.tight_layout()
save_chart(plt.gcf(), "S01_shap_beeswarm.png",
           "SHAP beeswarm summary plot")


# ══════════════════════════════════════════════════════════════════════
# 5.4  CHART S2 — GLOBAL SHAP BAR CHART (Mean |SHAP|)
# ══════════════════════════════════════════════════════════════════════
print("\n[5.4] Generating global SHAP bar chart...")

top15 = shap_importance.head(15).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))

colors = [
    RED   if v >= shap_importance.quantile(0.75)
    else AMBER if v >= shap_importance.quantile(0.50)
    else TEAL
    for v in top15.values
]

bars = ax.barh(
    top15.index, top15.values,
    color=colors, edgecolor=WHITE, linewidth=0.6, height=0.7
)

for bar, val in zip(bars, top15.values):
    ax.text(
        val + 0.001,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}",
        va="center", fontsize=9
    )

legend_patches = [
    mpatches.Patch(color=RED,   label="High impact  (top 25%)"),
    mpatches.Patch(color=AMBER, label="Medium impact (25–50%)"),
    mpatches.Patch(color=TEAL,  label="Lower impact  (bottom 50%)"),
]
ax.legend(handles=legend_patches, frameon=False, loc="lower right")
ax.set_xlabel("Mean |SHAP Value|  (average impact on model output)")
ax.set_title(
    "Figure S2: Mean Absolute SHAP Values — Top 15 Features\n"
    "(Higher value = feature has larger average impact on dropout prediction)"
)
fig.tight_layout()
save_chart(fig, "S02_shap_global_bar.png",
           "Global SHAP bar chart")


# ══════════════════════════════════════════════════════════════════════
# 5.5  CHART S3 — SHAP DEPENDENCE PLOTS (top 5 features)
# ══════════════════════════════════════════════════════════════════════
print("\n[5.5] Generating SHAP dependence plots for top 5 features...")

top5_features = shap_importance.head(5).index.tolist()

# Map clean names back to original column names for indexing
reverse_map = {clean_name(c): c for c in feature_cols}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes_flat = axes.flatten()

for i, feat_clean in enumerate(top5_features):
    feat_orig = reverse_map.get(feat_clean, feat_clean)

    if feat_orig not in feature_cols:
        # Try partial match
        matches = [c for c in feature_cols if feat_clean.replace(" ", "_") in c]
        if matches:
            feat_orig = matches[0]
        else:
            continue

    feat_idx = list(feature_cols).index(feat_orig)
    ax = axes_flat[i]

    shap_feat_vals  = shap_values[:, feat_idx]
    actual_feat_vals = X_test_display.iloc[:, feat_idx].values

    scatter = ax.scatter(
        actual_feat_vals,
        shap_feat_vals,
        c=shap_feat_vals,
        cmap="RdYlGn_r",
        alpha=0.4,
        s=8,
        vmin=-0.5, vmax=0.5
    )

    ax.axhline(0, color="#CCCCCC", linestyle="--", linewidth=1)
    ax.set_xlabel(feat_clean, fontsize=10)
    ax.set_ylabel("SHAP Value", fontsize=10)
    ax.set_title(f"{feat_clean}", color=PURPLE, fontsize=11)

    plt.colorbar(scatter, ax=ax, shrink=0.8, label="SHAP")

# Hide last empty panel
axes_flat[-1].set_visible(False)

fig.suptitle(
    "Figure S3: SHAP Dependence Plots — Top 5 Features\n"
    "(x-axis = feature value | y-axis = SHAP contribution to dropout risk | "
    "Points above 0 = increases dropout risk)",
    fontsize=11, color=PURPLE, y=1.01
)
fig.tight_layout()
save_chart(fig, "S03_shap_dependence_plots.png",
           "SHAP dependence plots — top 5 features")


# ══════════════════════════════════════════════════════════════════════
# 5.6  FIND EXAMPLE STUDENTS FOR WATERFALL CHARTS
# ══════════════════════════════════════════════════════════════════════
print("\n[5.6] Selecting example students for waterfall charts...")

prob_col = predictions["y_pred_prob"].values

# High risk: actual dropout = 1, predicted probability > 0.85
high_risk_idx = np.where(
    (y_test.values == 1) & (prob_col > 0.85)
)[0]

# Low risk: actual completed = 0, predicted probability < 0.10
low_risk_idx = np.where(
    (y_test.values == 0) & (prob_col < 0.10)
)[0]

if len(high_risk_idx) == 0:
    high_risk_idx = np.where(prob_col > 0.75)[0]
if len(low_risk_idx) == 0:
    low_risk_idx  = np.where(prob_col < 0.15)[0]

hr_idx = high_risk_idx[0]
lr_idx = low_risk_idx[0]

print(f"      High-risk student index: {hr_idx}  "
      f"(prob={prob_col[hr_idx]:.3f}, actual={y_test.values[hr_idx]})")
print(f"      Low-risk  student index: {lr_idx}  "
      f"(prob={prob_col[lr_idx]:.3f}, actual={y_test.values[lr_idx]})")


# ══════════════════════════════════════════════════════════════════════
# 5.7  CHART S4 — WATERFALL CHART: HIGH-RISK STUDENT
# ══════════════════════════════════════════════════════════════════════
print("\n[5.7] Generating waterfall chart — high-risk student...")

hr_shap   = shap_values[hr_idx]
hr_prob   = prob_col[hr_idx]
base_val  = explainer.expected_value

# Sort features by absolute SHAP value
sorted_idx    = np.argsort(np.abs(hr_shap))[::-1]
top_n         = 12
top_idx       = sorted_idx[:top_n]

top_features  = [clean_names[i] for i in top_idx]
top_shap_vals = [hr_shap[i]     for i in top_idx]
top_feat_vals = [X_test_display.iloc[hr_idx, i] for i in top_idx]

# Sort for waterfall (positive first descending, then negative)
combined      = sorted(
    zip(top_shap_vals, top_features, top_feat_vals),
    key=lambda x: abs(x[0]), reverse=True
)
top_shap_vals, top_features, top_feat_vals = zip(*combined)

fig, ax = plt.subplots(figsize=(11, 7))

cumulative = base_val
bar_lefts  = []
bar_widths = []
bar_colors = []

for sv in top_shap_vals:
    bar_lefts.append(cumulative)
    bar_widths.append(sv)
    bar_colors.append(RED if sv > 0 else TEAL)
    cumulative += sv

# Draw waterfall bars
y_positions = range(len(top_features))
bars = ax.barh(
    list(y_positions),
    bar_widths,
    left=bar_lefts,
    color=bar_colors,
    edgecolor=WHITE,
    linewidth=0.8,
    height=0.6,
)

# Base value line
ax.axvline(base_val, color="#888888", linestyle="--",
           linewidth=1.5, label=f"Base value: {base_val:.3f}")
# Final prediction line
ax.axvline(hr_prob, color=RED, linestyle="-",
           linewidth=2.5, label=f"Predicted probability: {hr_prob:.3f}")

# Labels on bars
for i, (left, width, sv, fv) in enumerate(
    zip(bar_lefts, bar_widths, top_shap_vals, top_feat_vals)
):
    sign   = "+" if sv > 0 else ""
    color  = RED if sv > 0 else TEAL
    ax.text(
        left + width / 2, i,
        f"{sign}{sv:.3f}",
        ha="center", va="center",
        fontsize=8.5, fontweight="bold", color=WHITE
    )

# Feature labels with their values
feature_labels = [
    f"{feat}  =  {val:.2f}" if isinstance(val, float)
    else f"{feat}  =  {val}"
    for feat, val in zip(top_features, top_feat_vals)
]
ax.set_yticks(list(y_positions))
ax.set_yticklabels(feature_labels, fontsize=9)
ax.set_xlabel("SHAP Value  (contribution to dropout probability)")
ax.set_title(
    f"Figure S4: SHAP Waterfall — HIGH RISK Student\n"
    f"Predicted Dropout Probability: {hr_prob:.1%}  |  "
    f"Actual: {'Dropout' if y_test.values[hr_idx]==1 else 'Completed'}",
    color=RED
)

legend_patches = [
    mpatches.Patch(color=RED,  label="Increases dropout risk"),
    mpatches.Patch(color=TEAL, label="Decreases dropout risk"),
    plt.Line2D([0],[0], color="#888888", linestyle="--",
               label=f"Base value: {base_val:.3f}"),
    plt.Line2D([0],[0], color=RED, linestyle="-", linewidth=2,
               label=f"Final prediction: {hr_prob:.3f}"),
]
ax.legend(handles=legend_patches, frameon=True, framealpha=0.9,
          loc="lower right", fontsize=9)

fig.tight_layout()
save_chart(fig, "S04_waterfall_high_risk.png",
           "SHAP waterfall — high risk student")


# ══════════════════════════════════════════════════════════════════════
# 5.8  CHART S5 — WATERFALL CHART: LOW-RISK STUDENT
# ══════════════════════════════════════════════════════════════════════
print("\n[5.8] Generating waterfall chart — low-risk student...")

lr_shap   = shap_values[lr_idx]
lr_prob   = prob_col[lr_idx]

sorted_idx    = np.argsort(np.abs(lr_shap))[::-1]
top_idx       = sorted_idx[:top_n]
top_features  = [clean_names[i] for i in top_idx]
top_shap_vals = [lr_shap[i]     for i in top_idx]
top_feat_vals = [X_test_display.iloc[lr_idx, i] for i in top_idx]

combined = sorted(
    zip(top_shap_vals, top_features, top_feat_vals),
    key=lambda x: abs(x[0]), reverse=True
)
top_shap_vals, top_features, top_feat_vals = zip(*combined)

fig, ax = plt.subplots(figsize=(11, 7))

cumulative = base_val
bar_lefts  = []
bar_widths = []
bar_colors = []

for sv in top_shap_vals:
    bar_lefts.append(cumulative)
    bar_widths.append(sv)
    bar_colors.append(RED if sv > 0 else TEAL)
    cumulative += sv

bars = ax.barh(
    list(y_positions),
    bar_widths,
    left=bar_lefts,
    color=bar_colors,
    edgecolor=WHITE,
    linewidth=0.8,
    height=0.6,
)

ax.axvline(base_val, color="#888888", linestyle="--",
           linewidth=1.5, label=f"Base value: {base_val:.3f}")
ax.axvline(lr_prob, color=TEAL, linestyle="-",
           linewidth=2.5, label=f"Predicted probability: {lr_prob:.3f}")

for i, (left, width, sv, fv) in enumerate(
    zip(bar_lefts, bar_widths, top_shap_vals, top_feat_vals)
):
    sign   = "+" if sv > 0 else ""
    color  = RED if sv > 0 else TEAL
    ax.text(
        left + width / 2, i,
        f"{sign}{sv:.3f}",
        ha="center", va="center",
        fontsize=8.5, fontweight="bold", color=WHITE
    )

feature_labels = [
    f"{feat}  =  {val:.2f}" if isinstance(val, float)
    else f"{feat}  =  {val}"
    for feat, val in zip(top_features, top_feat_vals)
]
ax.set_yticks(list(y_positions))
ax.set_yticklabels(feature_labels, fontsize=9)
ax.set_xlabel("SHAP Value  (contribution to dropout probability)")
ax.set_title(
    f"Figure S5: SHAP Waterfall — LOW RISK Student\n"
    f"Predicted Dropout Probability: {lr_prob:.1%}  |  "
    f"Actual: {'Dropout' if y_test.values[lr_idx]==1 else 'Completed'}",
    color=TEAL
)

legend_patches = [
    mpatches.Patch(color=RED,  label="Increases dropout risk"),
    mpatches.Patch(color=TEAL, label="Decreases dropout risk"),
    plt.Line2D([0],[0], color="#888888", linestyle="--",
               label=f"Base value: {base_val:.3f}"),
    plt.Line2D([0],[0], color=TEAL, linestyle="-", linewidth=2,
               label=f"Final prediction: {lr_prob:.3f}"),
]
ax.legend(handles=legend_patches, frameon=True, framealpha=0.9,
          loc="lower right", fontsize=9)

fig.tight_layout()
save_chart(fig, "S05_waterfall_low_risk.png",
           "SHAP waterfall — low risk student")


# ══════════════════════════════════════════════════════════════════════
# 5.9  CHART S6 — DOMAIN-LEVEL SHAP CONTRIBUTION
# ══════════════════════════════════════════════════════════════════════
print("\n[5.9] Generating domain-level SHAP contribution chart...")

domain_map = {
    "Income Band":          "Socioeconomic",
    "Fee Payment Status":   "Socioeconomic",
    "Parental Education":   "Socioeconomic",
    "Parental Involvement": "Socioeconomic",
    "Attendance Rate (%)":  "Academic / Behavioural",
    "Academic Score (%)":   "Academic / Behavioural",
    "Failed Subjects":      "Academic / Behavioural",
    "Cognitive Score":      "Cognitive",
    "Learning Pace":        "Cognitive",
    "Study Habit Score":    "Cognitive",
    "Practical Score":      "Psychomotor",
    "PE Participation":     "Psychomotor",
    "Psychomotor Score":    "Psychomotor",
    "Distance (km)":        "Demographic / Geographic",
    "Gender":               "Demographic / Geographic",
    "Age":                  "Demographic / Geographic",
}

domain_shap = {}
for feat_clean, shap_val in zip(
    shap_importance.index, shap_importance.values
):
    domain = domain_map.get(feat_clean, "Other")
    domain_shap[domain] = domain_shap.get(domain, 0) + shap_val

domain_series = pd.Series(domain_shap).sort_values(ascending=True)

domain_colors = {
    "Socioeconomic":            RED,
    "Academic / Behavioural":   PURPLE,
    "Cognitive":                AMBER,
    "Psychomotor":              TEAL,
    "Demographic / Geographic": "#1E3A5F",
    "Other":                    "#888888",
}

fig, ax = plt.subplots(figsize=(9, 5))

colors = [domain_colors.get(d, "#888888") for d in domain_series.index]
bars   = ax.barh(
    domain_series.index, domain_series.values,
    color=colors, edgecolor=WHITE, linewidth=0.8, height=0.6
)

for bar, val in zip(bars, domain_series.values):
    ax.text(
        val + 0.001,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}",
        va="center", fontsize=10, fontweight="bold"
    )

ax.set_xlabel("Total Mean |SHAP Value|  (summed across features in domain)")
ax.set_title(
    "Figure S6: Cumulative SHAP Contribution by Feature Domain\n"
    "(Which domain drives dropout prediction most?)"
)
fig.tight_layout()
save_chart(fig, "S06_shap_by_domain.png",
           "SHAP contribution by domain")


# ══════════════════════════════════════════════════════════════════════
# 5.10  SAVE SHAP SUMMARY FOR STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════
print("\n[5.10] Saving SHAP summary data for Streamlit app...")

shap_df = pd.DataFrame(
    shap_values,
    columns=feature_cols
)
shap_df.to_csv(f"{MODEL_DIR}/shap_values_df.csv", index=False)
print(f"       Saved → model/shap_values_df.csv")

# Save feature display name mapping
import json
display_map = {orig: clean_name(orig) for orig in feature_cols}
with open(f"{MODEL_DIR}/feature_display_names.json", "w") as f:
    json.dump(display_map, f, indent=2)
print(f"       Saved → model/feature_display_names.json")


# ══════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print(f"  SHAP ANALYSIS COMPLETE")
print(f"{'=' * 65}")
print(f"  Charts generated: {chart_count}")
print(f"  Saved to:  {CHART_DIR}/")
print()
print("  Chart index:")
print("  S01  Global SHAP beeswarm summary")
print("  S02  Global SHAP bar chart (mean |SHAP|)")
print("  S03  SHAP dependence plots — top 5 features")
print("  S04  Waterfall chart — high-risk student example")
print("  S05  Waterfall chart — low-risk student example")
print("  S06  SHAP contribution by feature domain")
print()
print("  Artefacts saved to model/:")
print("    shap_values.npy            — full SHAP value matrix")
print("    shap_values_df.csv         — SHAP values as DataFrame")
print("    shap_expected_value.pkl    — model base rate")
print("    shap_explainer.pkl         — fitted TreeExplainer")
print("    shap_importance.csv        — ranked feature importances")
print("    feature_display_names.json — clean name mapping for app")
print()
print(f"\n  Top 5 features driving dropout prediction:")
for i, (feat, val) in enumerate(shap_importance.head(5).items(), 1):
    print(f"    {i}. {feat:<35} mean|SHAP| = {val:.4f}")
print(f"\n{'=' * 65}")
print(f"\n  Step 5 complete. Build the intervention engine next.\n")