"""
=======================================================================
Step 2 — Data Preprocessing Pipeline
Dropout Risk Prediction System
Nzeako Jennifer Nkiruka | DU/C/247
=======================================================================
What this script does:
  1. Loads the raw synthetic dataset
  2. Separates features from target variable
  3. Encodes all categorical variables
  4. Scales all continuous features
  5. Splits into train / validation / test sets (80/10/10 stratified split)
  6. Applies SMOTE to balance the training set
  7. Saves all preprocessed artefacts to /data/ folder
  8. Prints a full preprocessing report

Run:  python preprocess.py
=======================================================================
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DATA  = "data/dropout_dataset_10000.csv"
DATA_DIR  = "data"
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 65)
print("  Step 2 — Data Preprocessing Pipeline")
print("  Dropout Risk Prediction System")
print("=" * 65)


# ══════════════════════════════════════════════════════════════════════
# 2.1  LOAD DATA
# ══════════════════════════════════════════════════════════════════════
print("\n[2.1] Loading raw dataset...")

df = pd.read_csv(RAW_DATA)
print(f"      Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")
print(f"      Dropout rate: {df['Dropout_Status'].mean()*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════
# 2.2  DEFINE FEATURE TYPES
# ══════════════════════════════════════════════════════════════════════
print("\n[2.2] Defining feature types...")

# Drop Student_ID — not a predictor
df = df.drop(columns=["Student_ID"])

TARGET = "Dropout_Status"

# Categorical features and their encoding type
ORDINAL_FEATURES = {
    "Household_Income_Band":     ["Low", "Medium", "High"],
    "Fee_Payment_Status":        ["Unpaid", "Partial", "Paid"],
    "Parental_Education_Level":  [
        "No Formal Education", "Primary", "Secondary",
        "OND/NCE", "HND/BSc", "Postgraduate"
    ],
    "Parental_Involvement":      ["Absent", "Passive", "Active"],
    "Learning_Pace":             ["Slow", "Average", "Fast"],
    "PE_Participation":          ["Absent", "Passive", "Active"],
}

BINARY_FEATURES = {
    "Gender": ["Male", "Female"],
}

NOMINAL_FEATURES = [
    "Year_Group",
]

CONTINUOUS_FEATURES = [
    "Age",
    "Distance_to_School_km",
    "Attendance_Rate_Pct",
    "Avg_Academic_Score_Pct",
    "Failed_Subjects_Count",
    "Cognitive_Assessment_Score",
    "Study_Habit_Score",
    "Practical_Subject_Score",
    "Psychomotor_Dev_Score",
]

print(f"      Ordinal features:    {len(ORDINAL_FEATURES)}")
print(f"      Binary features:     {len(BINARY_FEATURES)}")
print(f"      Nominal features:    {len(NOMINAL_FEATURES)}")
print(f"      Continuous features: {len(CONTINUOUS_FEATURES)}")


# ══════════════════════════════════════════════════════════════════════
# 2.3  CHECK FOR MISSING VALUES
# ══════════════════════════════════════════════════════════════════════
print("\n[2.3] Checking for missing values...")

missing = df.isnull().sum()
missing_cols = missing[missing > 0]

if len(missing_cols) == 0:
    print("      No missing values found. Dataset is clean.")
else:
    print(f"      Missing values found in {len(missing_cols)} columns:")
    for col, count in missing_cols.items():
        pct = count / len(df) * 100
        print(f"        {col}: {count} ({pct:.1f}%)")
        if col in CONTINUOUS_FEATURES:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)


# ══════════════════════════════════════════════════════════════════════
# 2.4  ENCODE CATEGORICAL FEATURES
# ══════════════════════════════════════════════════════════════════════
print("\n[2.4] Encoding categorical features...")

encoding_maps = {}

for feature, order in ORDINAL_FEATURES.items():
    mapping = {val: idx for idx, val in enumerate(order)}
    df[feature] = df[feature].map(mapping)
    encoding_maps[feature] = mapping
    print(f"      Ordinal: {feature} → {mapping}")

for feature, classes in BINARY_FEATURES.items():
    mapping = {classes[0]: 0, classes[1]: 1}
    df[feature] = df[feature].map(mapping)
    encoding_maps[feature] = mapping
    print(f"      Binary:  {feature} → {mapping}")

for feature in NOMINAL_FEATURES:
    dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=[feature])
    encoding_maps[feature] = list(dummies.columns)
    print(f"      One-hot: {feature} → {list(dummies.columns)}")

joblib.dump(encoding_maps, "data/encoding_maps.pkl")
print(f"\n      Encoding maps saved → data/encoding_maps.pkl")


# ══════════════════════════════════════════════════════════════════════
# 2.5  SEPARATE FEATURES AND TARGET
# ══════════════════════════════════════════════════════════════════════
print("\n[2.5] Separating features and target...")

feature_cols = [col for col in df.columns if col != TARGET]
joblib.dump(feature_cols, "data/feature_columns.pkl")

X = df[feature_cols]
y = df[TARGET]

print(f"      Feature matrix shape: {X.shape}")
print(f"      Target distribution:")
print(f"        Dropout (1):   {y.sum():,} ({y.mean()*100:.1f}%)")
print(f"        Completed (0): {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════
# 2.6  TRAIN / VALIDATION / TEST SPLIT (80/10/10)
# ══════════════════════════════════════════════════════════════════════
print("\n[2.6] Splitting into train, validation and test sets (80/10/10)...")

# Step 1: hold out 10% of the full dataset as the test set
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=0.10,
    random_state=42,
    stratify=y
)

# Step 2: split the remaining 90% into 80% train / 10% validation
# (10% of total = 0.1111 of the 90% remainder)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.1111,
    random_state=42,
    stratify=y_train_val
)

print(f"      Training set:   {X_train.shape[0]:,} rows "
      f"({y_train.mean()*100:.1f}% dropout)")
print(f"      Validation set: {X_val.shape[0]:,} rows  "
      f"({y_val.mean()*100:.1f}% dropout)")
print(f"      Test set:       {X_test.shape[0]:,} rows  "
      f"({y_test.mean()*100:.1f}% dropout)")


# ══════════════════════════════════════════════════════════════════════
# 2.7  FEATURE SCALING
# ══════════════════════════════════════════════════════════════════════
print("\n[2.7] Scaling continuous features...")

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_val_scaled   = X_val.copy()
X_test_scaled  = X_test.copy()

continuous_cols_present = [
    col for col in CONTINUOUS_FEATURES if col in X_train.columns
]

X_train_scaled[continuous_cols_present] = scaler.fit_transform(
    X_train[continuous_cols_present]
)
X_val_scaled[continuous_cols_present] = scaler.transform(
    X_val[continuous_cols_present]
)
X_test_scaled[continuous_cols_present] = scaler.transform(
    X_test[continuous_cols_present]
)

joblib.dump(scaler, "data/scaler.pkl")
joblib.dump(continuous_cols_present, "data/continuous_cols.pkl")

print(f"      Scaled {len(continuous_cols_present)} continuous features")
print(f"      Scaler saved → data/scaler.pkl")


# ══════════════════════════════════════════════════════════════════════
# 2.8  SMOTE — HANDLE CLASS IMBALANCE
# ══════════════════════════════════════════════════════════════════════
print("\n[2.8] Applying SMOTE to training set...")
print(f"      Before SMOTE:")
print(f"        Dropout (1):   {y_train.sum():,}")
print(f"        Completed (0): {(y_train==0).sum():,}")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_scaled, y_train
)

print(f"      After SMOTE:")
print(f"        Dropout (1):   {y_train_balanced.sum():,}")
print(f"        Completed (0): {(y_train_balanced==0).sum():,}")
print(f"        Total:         {len(y_train_balanced):,}")


# ══════════════════════════════════════════════════════════════════════
# 2.9  SAVE ALL PREPROCESSED ARTEFACTS
# ══════════════════════════════════════════════════════════════════════
print("\n[2.9] Saving preprocessed artefacts...")

X_train_df = pd.DataFrame(X_train_balanced, columns=feature_cols)
X_val_df   = pd.DataFrame(X_val_scaled,     columns=feature_cols)
X_test_df  = pd.DataFrame(X_test_scaled,    columns=feature_cols)

X_train_df.to_csv("data/X_train.csv", index=False)
X_val_df.to_csv(  "data/X_val.csv",   index=False)
X_test_df.to_csv( "data/X_test.csv",  index=False)

pd.Series(y_train_balanced, name="Dropout_Status").to_csv(
    "data/y_train.csv", index=False
)
pd.Series(y_val.values, name="Dropout_Status").to_csv(
    "data/y_val.csv", index=False
)
pd.Series(y_test.values, name="Dropout_Status").to_csv(
    "data/y_test.csv", index=False
)

X_val.to_csv( "data/X_val_unscaled.csv",  index=False)
X_test.to_csv("data/X_test_unscaled.csv", index=False)

print("      data/X_train.csv         — balanced training features")
print("      data/X_val.csv           — scaled validation features")
print("      data/X_test.csv          — scaled test features")
print("      data/y_train.csv         — training labels (SMOTE balanced)")
print("      data/y_val.csv           — validation labels")
print("      data/y_test.csv          — test labels")
print("      data/X_val_unscaled.csv  — unscaled validation features for SHAP")
print("      data/X_test_unscaled.csv — unscaled test features for SHAP")
print("      data/scaler.pkl          — fitted StandardScaler")
print("      data/encoding_maps.pkl   — categorical encoding mappings")
print("      data/feature_columns.pkl — ordered feature column names")


# ══════════════════════════════════════════════════════════════════════
# 2.10  SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print("  PREPROCESSING SUMMARY")
print(f"{'=' * 65}")
print(f"  Raw dataset:          10,000 rows x 19 columns")
print(f"  After dropping ID:    10,000 rows x 18 columns")
print(f"  Features after enc.:  {len(feature_cols)} features")
print(f"  Missing values:       0")
print(f"  Split ratio:          80 / 10 / 10 (train / val / test, stratified)")
print(f"  Train set (raw):      {X_train.shape[0]:,} rows  "
      f"({y_train.mean()*100:.1f}% dropout)")
print(f"  Validation set:       {X_val.shape[0]:,} rows  "
      f"({y_val.mean()*100:.1f}% dropout)")
print(f"  Test set:             {X_test.shape[0]:,} rows  "
      f"({y_test.mean()*100:.1f}% dropout)")
print(f"  Train set (SMOTE):    {len(X_train_balanced):,} rows (50/50 balanced)")
print(f"  Val/Test dropout:     real-world distribution (no SMOTE)")
print(f"{'=' * 65}")
print(f"\n  Step 2 complete. Run train.py next.\n")