"""
=======================================================================
Step 4 — Model Evaluation Charts
Dropout Risk Prediction System
Nzeako Jennifer Nkiruka | DU/C/247
=======================================================================
What this script generates:
  1.  ROC curves — all 3 models on one plot
  2.  Precision-Recall curves — all 3 models
  3.  Confusion matrix heatmaps — one per model
  4.  Random Forest feature importance bar chart
  5.  XGBoost feature importance bar chart
  6.  Side-by-side feature importance comparison
  7.  Model metrics comparison bar chart
  8.  Cross-validation scores comparison
  9.  Predicted probability distribution (XGBoost)
  10. Combined evaluation summary dashboard

All charts saved to: assets/charts/evaluation/
Resolution: 300 DPI — thesis ready

Run:  python evaluate.py
=======================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix,
    average_precision_score
)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR = "model"
DATA_DIR  = "data"
CHART_DIR = "assets/charts/evaluation"
os.makedirs(CHART_DIR, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
PURPLE     = "#4B3FA0"
TEAL       = "#0F6E56"
RED        = "#991B1B"
AMBER      = "#B45309"
BLUE       = "#1E3A5F"
LIGHT_GRAY = "#F5F5F3"
WHITE      = "#FFFFFF"

MODEL_COLORS = {
    "Logistic Regression": AMBER,
    "Random Forest":       TEAL,
    "XGBoost":             PURPLE,
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   PURPLE,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.facecolor":    WHITE,
    "figure.facecolor":  WHITE,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": WHITE,
})

print("=" * 65)
print("  Step 4 — Model Evaluation Charts")
print("  Dropout Risk Prediction System — Jennifer Nkiruka")
print("=" * 65)


# ══════════════════════════════════════════════════════════════════════
# LOAD DATA AND PREDICTIONS
# ══════════════════════════════════════════════════════════════════════
print("\n[4.1] Loading models and predictions...")

y_test      = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()
feature_cols = joblib.load(f"{DATA_DIR}/feature_columns.pkl")

model_names = {
    "Logistic Regression": "logistic_regression",
    "Random Forest":       "random_forest",
    "XGBoost":             "xgboost",
}

predictions = {}
for display_name, slug in model_names.items():
    preds = pd.read_csv(f"{MODEL_DIR}/{slug}_predictions.csv")
    predictions[display_name] = {
        "y_true":      preds["y_true"].values,
        "y_pred":      preds["y_pred"].values,
        "y_pred_prob": preds["y_pred_prob"].values,
    }
    print(f"      Loaded: {display_name}")

summary_df   = pd.read_csv(f"{MODEL_DIR}/model_comparison.csv")
rf_fi        = pd.read_csv(f"{MODEL_DIR}/rf_feature_importance.csv",
                            index_col=0).squeeze()
xgb_fi       = pd.read_csv(f"{MODEL_DIR}/xgb_feature_importance.csv",
                            index_col=0).squeeze()

chart_count = 0

def save_chart(fig, filename, title):
    global chart_count
    chart_count += 1
    path = f"{CHART_DIR}/{filename}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [{chart_count:02d}] Saved: {filename}  — {title}")


# ══════════════════════════════════════════════════════════════════════
# CHART 1 — ROC CURVES (all 3 models)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 7))

for model_name, preds in predictions.items():
    fpr, tpr, _ = roc_curve(preds["y_true"], preds["y_pred_prob"])
    roc_auc     = auc(fpr, tpr)
    ax.plot(
        fpr, tpr,
        color=MODEL_COLORS[model_name],
        linewidth=2.5,
        label=f"{model_name}  (AUC = {roc_auc:.4f})"
    )

# Diagonal reference line (random classifier)
ax.plot(
    [0, 1], [0, 1],
    color="#CCCCCC", linestyle="--", linewidth=1.5,
    label="Random Classifier (AUC = 0.50)"
)

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate  (1 - Specificity)")
ax.set_ylabel("True Positive Rate  (Recall / Sensitivity)")
ax.set_title(
    "Figure E1: ROC Curves — Model Comparison\n"
    "Higher AUC = Better Overall Discrimination"
)
ax.legend(loc="lower right", frameon=True, framealpha=0.9)

# Shade area under XGBoost curve lightly
xgb_preds = predictions["XGBoost"]
fpr_xgb, tpr_xgb, _ = roc_curve(
    xgb_preds["y_true"], xgb_preds["y_pred_prob"]
)
ax.fill_between(fpr_xgb, tpr_xgb, alpha=0.08, color=PURPLE)

fig.tight_layout()
save_chart(fig, "E01_roc_curves.png", "ROC curves — all 3 models")


# ══════════════════════════════════════════════════════════════════════
# CHART 2 — PRECISION-RECALL CURVES
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 7))

for model_name, preds in predictions.items():
    precision, recall, _ = precision_recall_curve(
        preds["y_true"], preds["y_pred_prob"]
    )
    ap = average_precision_score(preds["y_true"], preds["y_pred_prob"])
    ax.plot(
        recall, precision,
        color=MODEL_COLORS[model_name],
        linewidth=2.5,
        label=f"{model_name}  (AP = {ap:.4f})"
    )

# Baseline — random classifier
baseline = y_test.mean()
ax.axhline(
    baseline, color="#CCCCCC",
    linestyle="--", linewidth=1.5,
    label=f"Random Baseline  (AP = {baseline:.2f})"
)

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("Recall  (True Positive Rate)")
ax.set_ylabel("Precision  (Positive Predictive Value)")
ax.set_title(
    "Figure E2: Precision-Recall Curves — Model Comparison\n"
    "AP = Average Precision (area under curve)"
)
ax.legend(loc="upper right", frameon=True, framealpha=0.9)
fig.tight_layout()
save_chart(fig, "E02_precision_recall_curves.png",
           "Precision-Recall curves")


# ══════════════════════════════════════════════════════════════════════
# CHART 3 — CONFUSION MATRIX HEATMAPS (one per model)
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

cm_labels = ["Completed\n(0)", "Dropped\nOut (1)"]

for ax, (model_name, preds) in zip(axes, predictions.items()):
    cm = confusion_matrix(preds["y_true"], preds["y_pred"])

    # Normalised version (percentages) overlaid on counts
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_norm,
        annot=False,
        fmt=".2%",
        cmap="Blues",
        ax=ax,
        cbar=True,
        linewidths=1,
        linecolor=WHITE,
        vmin=0, vmax=1,
    )

    # Annotate with both count and percentage
    for i in range(2):
        for j in range(2):
            count   = cm[i, j]
            percent = cm_norm[i, j] * 100
            color   = WHITE if cm_norm[i, j] > 0.5 else "#333333"
            ax.text(
                j + 0.5, i + 0.45,
                f"{count:,}",
                ha="center", va="center",
                fontsize=15, fontweight="bold", color=color
            )
            ax.text(
                j + 0.5, i + 0.62,
                f"({percent:.1f}%)",
                ha="center", va="center",
                fontsize=10, color=color
            )

    ax.set_xticklabels(cm_labels, fontsize=11)
    ax.set_yticklabels(cm_labels, fontsize=11, rotation=0)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(f"{model_name}", color=MODEL_COLORS[model_name])

    tn, fp, fn, tp = cm.ravel()
    ax.text(
        0.5, -0.18,
        f"TP={tp}  FP={fp}  FN={fn}  TN={tn}",
        ha="center", transform=ax.transAxes,
        fontsize=9, color="#555555"
    )

fig.suptitle(
    "Figure E3: Confusion Matrices — All Three Models\n"
    "(Row = Actual class   |   Column = Predicted class)",
    fontsize=13, fontweight="bold", color=PURPLE, y=1.04
)
fig.tight_layout()
save_chart(fig, "E03_confusion_matrices.png",
           "Confusion matrix heatmaps — all 3 models")


# ══════════════════════════════════════════════════════════════════════
# CHART 4 — RANDOM FOREST FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))

rf_top15 = rf_fi.sort_values(ascending=True).tail(15)
colors   = [
    RED if v >= rf_fi.quantile(0.75)
    else AMBER if v >= rf_fi.quantile(0.50)
    else TEAL
    for v in rf_top15.values
]

bars = ax.barh(
    rf_top15.index, rf_top15.values,
    color=colors, edgecolor=WHITE, linewidth=0.6, height=0.7
)

for bar, val in zip(bars, rf_top15.values):
    ax.text(
        val + 0.001,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}",
        va="center", fontsize=9
    )

ax.set_xlabel("Feature Importance (Gini Impurity Reduction)")
ax.set_title(
    "Figure E4: Random Forest — Top 15 Feature Importances\n"
    "(Higher value = feature reduces impurity more = more predictive)"
)

import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(color=RED,   label="High importance (top 25%)"),
    mpatches.Patch(color=AMBER, label="Medium importance (50–75%)"),
    mpatches.Patch(color=TEAL,  label="Lower importance (bottom 50%)"),
]
ax.legend(handles=legend_patches, frameon=False, loc="lower right")
fig.tight_layout()
save_chart(fig, "E04_rf_feature_importance.png",
           "Random Forest feature importance")


# ══════════════════════════════════════════════════════════════════════
# CHART 5 — XGBOOST FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))

xgb_top15 = xgb_fi.sort_values(ascending=True).tail(15)
colors = [
    RED if v >= xgb_fi.quantile(0.75)
    else AMBER if v >= xgb_fi.quantile(0.50)
    else TEAL
    for v in xgb_top15.values
]

bars = ax.barh(
    xgb_top15.index, xgb_top15.values,
    color=colors, edgecolor=WHITE, linewidth=0.6, height=0.7
)

for bar, val in zip(bars, xgb_top15.values):
    ax.text(
        val + 0.001,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}",
        va="center", fontsize=9
    )

ax.set_xlabel("Feature Importance (XGBoost Gain)")
ax.set_title(
    "Figure E5: XGBoost — Top 15 Feature Importances\n"
    "(Gain = average improvement in loss when feature is used in a split)"
)
ax.legend(handles=legend_patches, frameon=False, loc="lower right")
fig.tight_layout()
save_chart(fig, "E05_xgb_feature_importance.png",
           "XGBoost feature importance")


# ══════════════════════════════════════════════════════════════════════
# CHART 6 — SIDE-BY-SIDE FEATURE IMPORTANCE (RF vs XGBoost)
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Align both to same top-10 features
top10_rf  = rf_fi.nlargest(10).sort_values(ascending=True)
top10_xgb = xgb_fi.nlargest(10).sort_values(ascending=True)

axes[0].barh(
    top10_rf.index, top10_rf.values,
    color=TEAL, edgecolor=WHITE, linewidth=0.6, height=0.7
)
axes[0].set_xlabel("Importance Score (Gini)")
axes[0].set_title("Random Forest\nTop 10 Features", color=TEAL)
for i, (idx, val) in enumerate(top10_rf.items()):
    axes[0].text(
        val + 0.0005, i,
        f"{val:.4f}", va="center", fontsize=9
    )

axes[1].barh(
    top10_xgb.index, top10_xgb.values,
    color=PURPLE, edgecolor=WHITE, linewidth=0.6, height=0.7
)
axes[1].set_xlabel("Importance Score (Gain)")
axes[1].set_title("XGBoost\nTop 10 Features", color=PURPLE)
for i, (idx, val) in enumerate(top10_xgb.items()):
    axes[1].text(
        val + 0.0005, i,
        f"{val:.4f}", va="center", fontsize=9
    )

fig.suptitle(
    "Figure E6: Feature Importance Comparison — Random Forest vs XGBoost\n"
    "(Features contributing most to dropout prediction in each model)",
    fontsize=13, fontweight="bold", color=PURPLE, y=1.02
)
fig.tight_layout()
save_chart(fig, "E06_feature_importance_comparison.png",
           "RF vs XGBoost feature importance comparison")


# ══════════════════════════════════════════════════════════════════════
# CHART 7 — MODEL METRICS COMPARISON BAR CHART
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))

metrics   = ["Accuracy", "Precision", "Recall", "F1", "Roc_auc"]
col_map   = {
    "Accuracy":  "Accuracy",
    "Precision": "Precision",
    "Recall":    "Recall",
    "F1":        "F1",
    "Roc_auc":   "Roc_auc",
}

n_models  = len(summary_df)
n_metrics = len(metrics)
x         = np.arange(n_metrics)
width     = 0.25

for idx, (_, row) in enumerate(summary_df.iterrows()):
    model_name = row["Model"]
    values = [row[col_map[m]] for m in metrics]
    bars = ax.bar(
        x + idx * width,
        values,
        width,
        label=model_name,
        color=list(MODEL_COLORS.values())[idx],
        edgecolor=WHITE,
        linewidth=0.6,
    )
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", fontsize=7.5, fontweight="bold"
        )

ax.set_xticks(x + width)
ax.set_xticklabels(
    ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
    fontsize=11
)
ax.set_ylim(0.75, 1.02)
ax.set_ylabel("Score")
ax.set_title(
    "Figure E7: Model Performance Comparison — All Metrics\n"
    "(Test set scores — higher is better for all metrics)"
)
ax.legend(frameon=False)
ax.axhline(0.90, color=LIGHT_GRAY, linestyle="--",
           linewidth=1, zorder=0, label="0.90 reference")
fig.tight_layout()
save_chart(fig, "E07_metrics_comparison.png",
           "Model metrics comparison bar chart")


# ══════════════════════════════════════════════════════════════════════
# CHART 8 — CROSS-VALIDATION SCORES WITH ERROR BARS
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cv_metrics = ["recall", "f1", "roc_auc", "precision", "accuracy"]
cv_labels  = ["Recall", "F1-Score", "AUC-ROC", "Precision", "Accuracy"]

for ax_idx, (col_prefix, ax, title) in enumerate(zip(
    ["CV_", "CV_"],
    axes,
    ["CV Means (5-fold)", "CV Standard Deviations (5-fold)"]
)):
    x = np.arange(len(cv_metrics))
    width = 0.25

    for idx, (_, row) in enumerate(summary_df.iterrows()):
        model_name = row["Model"]
        if ax_idx == 0:
            values = [row[f"CV_{m}_mean"] for m in cv_metrics]
            errors = [row[f"CV_{m}_std"]  for m in cv_metrics]
            bars = ax.bar(
                x + idx * width, values, width,
                label=model_name,
                color=list(MODEL_COLORS.values())[idx],
                edgecolor=WHITE, linewidth=0.6,
                yerr=errors,
                capsize=3,
                error_kw={"elinewidth": 1.2, "ecolor": "#555555"}
            )
        else:
            values = [row[f"CV_{m}_std"] for m in cv_metrics]
            bars = ax.bar(
                x + idx * width, values, width,
                label=model_name,
                color=list(MODEL_COLORS.values())[idx],
                edgecolor=WHITE, linewidth=0.6,
                alpha=0.8
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(cv_labels, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    if ax_idx == 0:
        ax.set_ylim(0.75, 1.05)

fig.suptitle(
    "Figure E8: Cross-Validation Results — 5-Fold Stratified\n"
    "(Error bars show ± 1 standard deviation across folds)",
    fontsize=13, fontweight="bold", color=PURPLE, y=1.02
)
fig.tight_layout()
save_chart(fig, "E08_cross_validation_scores.png",
           "Cross-validation scores with error bars")


# ══════════════════════════════════════════════════════════════════════
# CHART 9 — PREDICTED PROBABILITY DISTRIBUTION (XGBoost)
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

xgb_preds = predictions["XGBoost"]
probs      = xgb_preds["y_pred_prob"]
y_true     = xgb_preds["y_true"]

# Left: overall distribution
axes[0].hist(
    probs, bins=50,
    color=PURPLE, edgecolor=WHITE, linewidth=0.4, alpha=0.85
)
axes[0].axvline(
    0.5, color=RED, linestyle="--", linewidth=2,
    label="Decision threshold (0.5)"
)
axes[0].set_xlabel("Predicted Dropout Probability")
axes[0].set_ylabel("Number of Students")
axes[0].set_title("Overall Probability Distribution")
axes[0].legend(frameon=False)

# Right: by actual class
for cls, color, label in zip(
    [0, 1], [TEAL, RED],
    ["Completed (Actual = 0)", "Dropped Out (Actual = 1)"]
):
    mask = y_true == cls
    axes[1].hist(
        probs[mask], bins=40, alpha=0.65,
        color=color, edgecolor=WHITE, linewidth=0.4,
        label=label
    )
axes[1].axvline(
    0.5, color=AMBER, linestyle="--", linewidth=2,
    label="Decision threshold (0.5)"
)
axes[1].set_xlabel("Predicted Dropout Probability")
axes[1].set_ylabel("Number of Students")
axes[1].set_title("Probability Distribution by Actual Class")
axes[1].legend(frameon=False)

fig.suptitle(
    "Figure E9: XGBoost — Predicted Probability Distribution\n"
    "(Good separation between classes indicates strong discrimination)",
    fontsize=13, fontweight="bold", color=PURPLE, y=1.02
)
fig.tight_layout()
save_chart(fig, "E09_probability_distribution.png",
           "XGBoost predicted probability distribution")


# ══════════════════════════════════════════════════════════════════════
# CHART 10 — SUMMARY EVALUATION DASHBOARD
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel A: ROC curves ───────────────────────────────────────────────
ax_roc = fig.add_subplot(gs[0, 0])
for model_name, preds in predictions.items():
    fpr, tpr, _ = roc_curve(preds["y_true"], preds["y_pred_prob"])
    roc_auc     = auc(fpr, tpr)
    ax_roc.plot(
        fpr, tpr, color=MODEL_COLORS[model_name],
        linewidth=2, label=f"{model_name}\n(AUC={roc_auc:.3f})"
    )
ax_roc.plot([0,1],[0,1], color="#CCCCCC", linestyle="--", linewidth=1)
ax_roc.set_xlabel("False Positive Rate", fontsize=9)
ax_roc.set_ylabel("True Positive Rate", fontsize=9)
ax_roc.set_title("A. ROC Curves")
ax_roc.legend(fontsize=7, frameon=True, framealpha=0.9)

# ── Panel B: XGBoost confusion matrix ────────────────────────────────
ax_cm = fig.add_subplot(gs[0, 1])
xgb_preds = predictions["XGBoost"]
cm     = confusion_matrix(xgb_preds["y_true"], xgb_preds["y_pred"])
cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(
    cm_pct, annot=False, cmap="Blues", ax=ax_cm,
    linewidths=1, linecolor=WHITE, vmin=0, vmax=1, cbar=False
)
for i in range(2):
    for j in range(2):
        c = WHITE if cm_pct[i,j] > 0.5 else "#333333"
        ax_cm.text(j+0.5, i+0.4, f"{cm[i,j]:,}",
                   ha="center", fontsize=12, fontweight="bold", color=c)
        ax_cm.text(j+0.5, i+0.62, f"({cm_pct[i,j]*100:.1f}%)",
                   ha="center", fontsize=9, color=c)
ax_cm.set_xticklabels(["Completed", "Dropout"], fontsize=9)
ax_cm.set_yticklabels(["Completed", "Dropout"], fontsize=9, rotation=0)
ax_cm.set_title("B. XGBoost Confusion Matrix", color=PURPLE)

# ── Panel C: Metrics bar chart ────────────────────────────────────────
ax_metrics = fig.add_subplot(gs[0, 2])
metrics_plot = ["Accuracy", "Precision", "Recall", "F1", "Roc_auc"]
col_map2     = {
    "Accuracy": "Accuracy","Precision":"Precision",
    "Recall":"Recall","F1":"F1","Roc_auc":"Roc_auc"
}

x     = np.arange(len(metrics_plot))
width = 0.25
for idx, (_, row) in enumerate(summary_df.iterrows()):
    values = [row[col_map2[m]] for m in metrics_plot]
    ax_metrics.bar(
        x + idx*width, values, width,
        label=row["Model"],
        color=list(MODEL_COLORS.values())[idx],
        edgecolor=WHITE, linewidth=0.5
    )
ax_metrics.set_xticks(x + width)
ax_metrics.set_xticklabels(
    ["Acc","Prec","Rec","F1","AUC"], fontsize=9
)
ax_metrics.set_ylim(0.75, 1.02)
ax_metrics.set_title("C. Metrics Comparison")
ax_metrics.legend(fontsize=7, frameon=False)

# ── Panel D: XGBoost feature importance ──────────────────────────────
ax_fi = fig.add_subplot(gs[1, 0:2])
xgb_top10 = xgb_fi.nlargest(10).sort_values(ascending=True)
fi_colors  = [
    RED if v >= xgb_fi.quantile(0.75)
    else AMBER if v >= xgb_fi.quantile(0.50)
    else TEAL
    for v in xgb_top10.values
]
bars = ax_fi.barh(
    xgb_top10.index, xgb_top10.values,
    color=fi_colors, edgecolor=WHITE, linewidth=0.5, height=0.65
)
for bar, val in zip(bars, xgb_top10.values):
    ax_fi.text(
        val+0.0005, bar.get_y()+bar.get_height()/2,
        f"{val:.4f}", va="center", fontsize=8
    )
ax_fi.set_xlabel("Importance Score (XGBoost Gain)", fontsize=9)
ax_fi.set_title("D. XGBoost — Top 10 Feature Importances", color=PURPLE)

# ── Panel E: Probability distribution ────────────────────────────────
ax_prob = fig.add_subplot(gs[1, 2])
for cls, color, label in zip(
    [0, 1], [TEAL, RED], ["Completed", "Dropped Out"]
):
    mask = y_true == cls
    ax_prob.hist(
        probs[mask], bins=30, alpha=0.65,
        color=color, edgecolor=WHITE, linewidth=0.3,
        label=label, density=True
    )
ax_prob.axvline(
    0.5, color=AMBER, linestyle="--",
    linewidth=1.5, label="Threshold"
)
ax_prob.set_xlabel("Predicted Probability", fontsize=9)
ax_prob.set_ylabel("Density", fontsize=9)
ax_prob.set_title("E. XGBoost Probability Distribution", color=PURPLE)
ax_prob.legend(fontsize=8, frameon=False)

fig.suptitle(
    "Figure E10: Model Evaluation Summary Dashboard\n"
    "Dropout Risk Prediction System — Nzeako Jennifer Nkiruka | DU/C/247",
    fontsize=14, fontweight="bold", color=PURPLE, y=1.01
)

save_chart(fig, "E10_evaluation_dashboard.png",
           "Full evaluation summary dashboard")


# ══════════════════════════════════════════════════════════════════════
# PRINT FINAL METRICS TABLE
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print("  FINAL METRICS SUMMARY — TEST SET")
print(f"{'=' * 65}")

col_display = {
    "Accuracy":  "Accuracy ",
    "Precision": "Precision",
    "Recall":    "Recall   ",
    "F1":        "F1-Score ",
    "Roc_auc":   "AUC-ROC  ",
}

print(f"\n  {'Metric':<14}", end="")
for _, row in summary_df.iterrows():
    print(f"  {row['Model']:<22}", end="")
print()
print("  " + "-" * 68)

for col, label in col_display.items():
    print(f"  {label:<14}", end="")
    vals = [row[col] for _, row in summary_df.iterrows()]
    best = max(vals)
    for val in vals:
        mark = " ◄" if abs(val - best) < 0.0001 else ""
        print(f"  {val:.4f}{mark:<16}", end="")
    print()

print(f"\n{'=' * 65}")
print(f"  EVALUATION COMPLETE")
print(f"{'=' * 65}")
print(f"  Charts generated: {chart_count}")
print(f"  Saved to:  {CHART_DIR}/")
print()
print("  Chart index:")
print("  E01  ROC curves — all 3 models")
print("  E02  Precision-Recall curves")
print("  E03  Confusion matrix heatmaps")
print("  E04  Random Forest feature importance")
print("  E05  XGBoost feature importance")
print("  E06  Feature importance comparison (RF vs XGBoost)")
print("  E07  Model metrics comparison bar chart")
print("  E08  Cross-validation scores with error bars")
print("  E09  XGBoost probability distribution")
print("  E10  Full evaluation dashboard (thesis figure)")
print(f"\n  Step 4 complete. Run shap_analysis.py next.\n")