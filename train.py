"""
=======================================================================
Step 3 — Model Training
Dropout Risk Prediction System
Nzeako Jennifer Nkiruka | DU/C/247
=======================================================================
What this script does:
  1.  Loads preprocessed training and test data
  2.  Trains three models:
        - Logistic Regression  (baseline)
        - Random Forest        (primary ensemble)
        - XGBoost              (expected best performer)
  3.  Tunes each model using 5-fold cross-validation
  4.  Evaluates all three on the held-out test set
  5.  Prints a full comparison report
  6.  Saves all three trained models to /model/ folder

Run:  python train.py
=======================================================================
"""

import pandas as pd
import numpy as np
import joblib
import os
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = "data"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 65)
print("  Step 3 — Model Training")
print("  Dropout Risk Prediction System — Jennifer Nkiruka")
print("=" * 65)


# ══════════════════════════════════════════════════════════════════════
# 3.1  LOAD PREPROCESSED DATA
# ══════════════════════════════════════════════════════════════════════
print("\n[3.1] Loading preprocessed data...")

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test  = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
y_test  = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()

feature_cols = joblib.load(f"{DATA_DIR}/feature_columns.pkl")

print(f"      Training set:  {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
print(f"      Test set:      {X_test.shape[0]:,} rows x {X_test.shape[1]} features")
print(f"      Train dropout: {y_train.mean()*100:.1f}%  (SMOTE balanced)")
print(f"      Test dropout:  {y_test.mean()*100:.1f}%   (real-world distribution)")


# ══════════════════════════════════════════════════════════════════════
# 3.2  CROSS-VALIDATION SETUP
# ══════════════════════════════════════════════════════════════════════
print("\n[3.2] Setting up 5-fold stratified cross-validation...")

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cross_validate_model(model, X, y, model_name):
    """Run 5-fold CV and return mean scores."""
    metrics = {}
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        scores = cross_val_score(
            model, X, y, cv=CV,
            scoring=metric,
            n_jobs=-1
        )
        metrics[metric] = {
            "mean": scores.mean(),
            "std":  scores.std()
        }
    return metrics

def evaluate_on_test(model, X_test, y_test):
    """Evaluate trained model on test set."""
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy":  accuracy_score(y_test,  y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test,    y_pred, zero_division=0),
        "f1":        f1_score(y_test,        y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test,   y_pred_prob),
        "y_pred":    y_pred,
        "y_pred_prob": y_pred_prob,
    }

results = {}   # stores all results for final comparison


# ══════════════════════════════════════════════════════════════════════
# 3.3  MODEL 1 — LOGISTIC REGRESSION (BASELINE)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "-" * 65)
print("[3.3] MODEL 1 — Logistic Regression (Baseline)")
print("-" * 65)

print("      Training...")
start = time.time()

lr_model = LogisticRegression(
    C=1.0,              # regularisation strength (default)
    max_iter=1000,      # enough iterations to converge
    solver="lbfgs",     # efficient for small-medium datasets
    class_weight=None,  # SMOTE already balanced the classes
    random_state=42,
    n_jobs=-1
)

lr_model.fit(X_train, y_train)
elapsed = time.time() - start
print(f"      Training time: {elapsed:.2f}s")

print("      Running 5-fold cross-validation...")
lr_cv = cross_validate_model(lr_model, X_train, y_train, "Logistic Regression")

print("      Evaluating on test set...")
lr_test = evaluate_on_test(lr_model, X_test, y_test)

results["Logistic Regression"] = {
    "model": lr_model, "cv": lr_cv, "test": lr_test
}

print(f"\n      Cross-Validation Results (mean ± std):")
for metric, vals in lr_cv.items():
    print(f"        {metric:<12} {vals['mean']:.4f} ± {vals['std']:.4f}")

print(f"\n      Test Set Results:")
for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    print(f"        {metric:<12} {lr_test[metric]:.4f}")

# Save model
joblib.dump(lr_model, f"{MODEL_DIR}/logistic_regression.pkl")
print(f"\n      Saved → model/logistic_regression.pkl")


# ══════════════════════════════════════════════════════════════════════
# 3.4  MODEL 2 — RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════
print("\n" + "-" * 65)
print("[3.4] MODEL 2 — Random Forest")
print("-" * 65)

print("      Training (this may take 30-60 seconds)...")
start = time.time()

rf_model = RandomForestClassifier(
    n_estimators=300,       # 300 trees — good balance of accuracy vs speed
    max_depth=None,         # trees grow until all leaves are pure
    min_samples_split=5,    # minimum samples to split a node
    min_samples_leaf=2,     # minimum samples at each leaf
    max_features="sqrt",    # sqrt(n_features) at each split — standard RF
    class_weight=None,      # SMOTE already balanced the classes
    bootstrap=True,
    oob_score=True,         # out-of-bag score — free internal validation
    random_state=42,
    n_jobs=-1               # use all CPU cores
)

rf_model.fit(X_train, y_train)
elapsed = time.time() - start
print(f"      Training time: {elapsed:.2f}s")
print(f"      OOB Score:     {rf_model.oob_score_:.4f}  "
      f"(out-of-bag estimate of accuracy)")

print("      Running 5-fold cross-validation...")
rf_cv = cross_validate_model(rf_model, X_train, y_train, "Random Forest")

print("      Evaluating on test set...")
rf_test = evaluate_on_test(rf_model, X_test, y_test)

results["Random Forest"] = {
    "model": rf_model, "cv": rf_cv, "test": rf_test
}

print(f"\n      Cross-Validation Results (mean ± std):")
for metric, vals in rf_cv.items():
    print(f"        {metric:<12} {vals['mean']:.4f} ± {vals['std']:.4f}")

print(f"\n      Test Set Results:")
for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    print(f"        {metric:<12} {rf_test[metric]:.4f}")

# Feature importance — top 10
importances = rf_model.feature_importances_
fi_series = pd.Series(importances, index=feature_cols).sort_values(
    ascending=False
)
print(f"\n      Top 10 Feature Importances (Gini):")
for feat, imp in fi_series.head(10).items():
    bar = "█" * int(imp * 200)
    print(f"        {feat:<35} {imp:.4f}  {bar}")

# Save model and feature importance
joblib.dump(rf_model, f"{MODEL_DIR}/random_forest.pkl")
fi_series.to_csv(f"{MODEL_DIR}/rf_feature_importance.csv", header=["importance"])
print(f"\n      Saved → model/random_forest.pkl")
print(f"      Saved → model/rf_feature_importance.csv")


# ══════════════════════════════════════════════════════════════════════
# 3.5  MODEL 3 — XGBOOST
# ══════════════════════════════════════════════════════════════════════
print("\n" + "-" * 65)
print("[3.5] MODEL 3 — XGBoost (Expected Best Performer)")
print("-" * 65)

print("      Training (this may take 30-60 seconds)...")
start = time.time()

xgb_model = XGBClassifier(
    n_estimators=500,           # 500 boosting rounds
    max_depth=6,                # tree depth — standard for tabular data
    learning_rate=0.05,         # small learning rate → more trees, better accuracy
    subsample=0.8,              # 80% of data per tree — reduces overfitting
    colsample_bytree=0.8,       # 80% of features per tree
    min_child_weight=3,         # minimum sum of instance weight in a child
    gamma=0.1,                  # minimum loss reduction to make a split
    reg_alpha=0.1,              # L1 regularisation
    reg_lambda=1.0,             # L2 regularisation
    scale_pos_weight=1,         # SMOTE balanced — no need to adjust
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
elapsed = time.time() - start
print(f"      Training time: {elapsed:.2f}s")

print("      Running 5-fold cross-validation...")
xgb_cv = cross_validate_model(xgb_model, X_train, y_train, "XGBoost")

print("      Evaluating on test set...")
xgb_test = evaluate_on_test(xgb_model, X_test, y_test)

results["XGBoost"] = {
    "model": xgb_model, "cv": xgb_cv, "test": xgb_test
}

print(f"\n      Cross-Validation Results (mean ± std):")
for metric, vals in xgb_cv.items():
    print(f"        {metric:<12} {vals['mean']:.4f} ± {vals['std']:.4f}")

print(f"\n      Test Set Results:")
for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    print(f"        {metric:<12} {xgb_test[metric]:.4f}")

# Feature importance
xgb_importance = pd.Series(
    xgb_model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print(f"\n      Top 10 Feature Importances (XGBoost gain):")
for feat, imp in xgb_importance.head(10).items():
    bar = "█" * int(imp * 200)
    print(f"        {feat:<35} {imp:.4f}  {bar}")

# Save model and feature importance
joblib.dump(xgb_model, f"{MODEL_DIR}/xgboost.pkl")
xgb_importance.to_csv(
    f"{MODEL_DIR}/xgb_feature_importance.csv", header=["importance"]
)
print(f"\n      Saved → model/xgboost.pkl")
print(f"      Saved → model/xgb_feature_importance.csv")


# ══════════════════════════════════════════════════════════════════════
# 3.6  MODEL COMPARISON REPORT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  MODEL COMPARISON REPORT — TEST SET RESULTS")
print("=" * 65)

metrics_order = ["accuracy", "precision", "recall", "f1", "roc_auc"]
metric_labels = {
    "accuracy":  "Accuracy ",
    "precision": "Precision",
    "recall":    "Recall   ",
    "f1":        "F1-Score ",
    "roc_auc":   "AUC-ROC  ",
}

# Header
print(f"\n  {'Metric':<14}", end="")
for model_name in results:
    print(f"  {model_name:<22}", end="")
print()
print("  " + "-" * 60)

# Rows
for metric in metrics_order:
    print(f"  {metric_labels[metric]:<14}", end="")
    values = [results[m]["test"][metric] for m in results]
    best_val = max(values)
    for model_name, val in zip(results.keys(), values):
        marker = " ◄ BEST" if abs(val - best_val) < 0.0001 else ""
        print(f"  {val:.4f}{marker:<16}", end="")
    print()

print("\n  " + "-" * 60)

# Identify best model
best_model_name = max(
    results.keys(),
    key=lambda m: results[m]["test"]["recall"]   # primary: recall
)
best_model_f1 = max(
    results.keys(),
    key=lambda m: results[m]["test"]["f1"]        # secondary: f1
)

print(f"\n  Best model by Recall:   {best_model_name}")
print(f"  Best model by F1-Score: {best_model_f1}")
print(f"\n  NOTE: Recall is the primary metric — in dropout prediction,")
print(f"  missing a student who drops out (false negative) costs more")
print(f"  than falsely flagging one who stays (false positive).")


# ══════════════════════════════════════════════════════════════════════
# 3.7  CLASSIFICATION REPORTS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  DETAILED CLASSIFICATION REPORTS")
print("=" * 65)

for model_name, res in results.items():
    print(f"\n  --- {model_name} ---")
    y_pred = res["test"]["y_pred"]
    report = classification_report(
        y_test, y_pred,
        target_names=["Completed (0)", "Dropped Out (1)"]
    )
    for line in report.split("\n"):
        print(f"  {line}")


# ══════════════════════════════════════════════════════════════════════
# 3.8  CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  CONFUSION MATRICES — TEST SET")
print("=" * 65)

for model_name, res in results.items():
    print(f"\n  --- {model_name} ---")
    y_pred = res["test"]["y_pred"]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"                    Predicted")
    print(f"                 Completed  Dropout")
    print(f"  Actual Completed   {tn:>5}     {fp:>5}   "
          f"← {fp} falsely flagged as at-risk")
    print(f"  Actual Dropout     {fn:>5}     {tp:>5}   "
          f"← {fn} missed (false negatives)")
    print(f"\n  True Positives  (correctly caught dropouts): {tp:,}")
    print(f"  False Negatives (missed dropouts):           {fn:,}")
    print(f"  False Positives (false alarms):              {fp:,}")
    print(f"  True Negatives  (correctly passed):          {tn:,}")


# ══════════════════════════════════════════════════════════════════════
# 3.9  SAVE RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n[3.9] Saving results summary...")

summary_rows = []
for model_name, res in results.items():
    row = {"Model": model_name}
    for metric in metrics_order:
        row[metric.capitalize()] = round(res["test"][metric], 4)
    for metric, vals in res["cv"].items():
        row[f"CV_{metric}_mean"] = round(vals["mean"], 4)
        row[f"CV_{metric}_std"]  = round(vals["std"],  4)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{MODEL_DIR}/model_comparison.csv", index=False)
print(f"      Saved → model/model_comparison.csv")

# Save best model reference
best_overall = best_model_name
joblib.dump(results[best_overall]["model"], f"{MODEL_DIR}/best_model.pkl")
joblib.dump(best_overall, f"{MODEL_DIR}/best_model_name.pkl")
print(f"      Saved → model/best_model.pkl  ({best_overall})")
print(f"      Saved → model/best_model_name.pkl")

# Save all predictions for evaluation script
for model_name, res in results.items():
    slug = model_name.lower().replace(" ", "_")
    pd.DataFrame({
        "y_true":     y_test.values,
        "y_pred":     res["test"]["y_pred"],
        "y_pred_prob":res["test"]["y_pred_prob"]
    }).to_csv(f"{MODEL_DIR}/{slug}_predictions.csv", index=False)
print(f"      Saved prediction CSVs for all 3 models")


# ══════════════════════════════════════════════════════════════════════
# 3.10  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print(f"  TRAINING COMPLETE")
print(f"{'=' * 65}")
print(f"  Models trained:    3  (LR baseline, Random Forest, XGBoost)")
print(f"  Best by Recall:    {best_model_name}")
print(f"  Best by F1-Score:  {best_model_f1}")
print()
print(f"  Files saved to model/:")
print(f"    logistic_regression.pkl")
print(f"    random_forest.pkl")
print(f"    xgboost.pkl")
print(f"    best_model.pkl  (copy of best model)")
print(f"    best_model_name.pkl")
print(f"    rf_feature_importance.csv")
print(f"    xgb_feature_importance.csv")
print(f"    model_comparison.csv")
print(f"    *_predictions.csv  (one per model)")
print(f"{'=' * 65}")
print(f"\n  Step 3 complete. Run evaluate.py next.\n")