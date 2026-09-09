# EduGuard — Early Dropout Intelligence System

> **Predict · Explain · Intervene**

A dropout risk prediction and intervention recommendation system
for Nigerian secondary school students, built using ensemble machine
learning and deployed as a public web application.

---

## Research Context

**Title:** A Dropout Risk Prediction and Intervention Recommendation
System for Secondary School Students Using Ensemble Learning on
Socioeconomic and Academic Datasets

**Researcher:** Nzeako Jennifer Nkiruka | DU/C/247
**Supervisor:** Mr Akinnifesi
**Department:** Physical and Mathematical Science
**Degree:** MSc Computer Science

---

## What EduGuard Does

Secondary school dropout is one of the most consequential educational
failures in Nigeria. EduGuard uses **XGBoost ensemble learning** and
**SHAP explainability** to identify students at risk of dropping out —
weeks before they leave — and delivers specific, ranked interventions
for school administrators and teachers.

Unlike most dropout prediction tools that stop at a risk score,
EduGuard bridges **prediction and prevention** through a rule-based
intervention recommendation engine mapped to the Ojo & Awonuga (2022)
Nigerian school intervention taxonomy.

---

## System Features

| Feature | Description |
|---|---|
| 🎯 Student Predictor | Enter one student's details — get risk score, SHAP explanation, and ranked interventions |
| 📊 Batch Dashboard | Upload a class CSV — assess all students, filter by risk, download reports |
| 📈 Model Performance | ROC curves, confusion matrices, SHAP importance, cross-validation results |
| 📋 About | Research context, methodology, synthetic data justification, references |

---

## Dataset

- **10,000 synthetic student records**
- **17 predictive features** across 5 domains
- **Calibrated to:** UNESCO UIS Nigeria (2023), World Bank GHS-Panel,
  UCI Student Dropout Dataset (Realinho et al., 2022)
- **Dropout rate:** 30.0% — consistent with Nigerian public secondary
  school literature

### Feature Domains

| Domain | Features |
|---|---|
| Demographic | Gender, Age, Distance to school |
| Socioeconomic | Income band, Fee payment status, Parental education, Parental involvement |
| Academic | Attendance rate, Academic score, Failed subjects |
| Cognitive | Cognitive score, Learning pace, Study habit score |
| Psychomotor | Practical score, PE participation, Psychomotor development score |

---

## Model Results

| Model | Accuracy | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression | 0.9325 | 0.9117 | 0.8902 | 0.9892 |
| Random Forest | 0.9195 | 0.8683 | 0.8662 | 0.9911 |
| **XGBoost (selected)** | **0.9315** | **0.8967** | **0.8871** | **0.9944** |

**XGBoost was selected** as the production model for its superior
AUC-ROC, lowest false positive count, and native SHAP TreeExplainer
support for fast, exact per-student explanations.

---

## Tech Stack

- **Python 3.13**
- **XGBoost** — primary prediction model
- **SHAP** — model explainability
- **scikit-learn** — preprocessing and evaluation
- **imbalanced-learn** — SMOTE class balancing
- **Streamlit** — web application framework
- **Plotly** — interactive charts
- **pandas / numpy** — data manipulation

---

## Project Structure

eduguard-dropout-prediction/
├── app.py ← Home page (landing)
├── requirements.txt
├── README.md
│
├── pages/
│ ├── predictor.py ← Single student risk predictor
│ ├── batch.py ← Batch risk dashboard
│ ├── performance.py ← Model performance dashboard
│ └── about.py ← Research context and references
│
├── utils/
│ ├── init.py
│ ├── helpers.py ← Shared functions and artefact loader
│ └── interventions.py ← Intervention recommendation engine
│
├── model/
│ ├── best_model.pkl ← XGBoost (production model)
│ ├── xgboost.pkl
│ ├── random_forest.pkl
│ ├── logistic_regression.pkl
│ ├── shap_explainer.pkl
│ ├── shap_importance.csv
│ ├── model_comparison.csv
│ └── *_predictions.csv
│
└── data/
├── dropout_dataset_10000.csv
├── feature_columns.pkl
├── encoding_maps.pkl
├── scaler.pkl
├── continuous_cols.pkl
├── X_test.csv
└── y_test.csv


---

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Courage97/eduguard-dropout-prediction.git
cd eduguard-dropout-prediction

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Live Demo

🌐 **[https://eduguard-dropout-prediction.streamlit.app](https://eduguard-dropout-prediction.streamlit.app)**

---

## Key References

- Ojo & Awonuga (2022). Nigerian secondary school dropout intervention
  taxonomy. *Nigerian Journal of Educational Research and Policy.*
- Hussain et al. (2023). XGBoost for dropout prediction.
  *Indonesian JEECS, 9(2), 447–459.*
- Kizito (2023). Early warning systems. *IJERO, 4, 100231.*
- UNESCO UIS (2023). Out-of-school children, Sub-Saharan Africa.
- Realinho et al. (2022). UCI Student Dropout Dataset. *Data, 7(11).*

---

## Developer

**Barnabas Pelumi Oyewole**
CourageWorld Academy | Ibadan, Nigeria
GitHub: [github.com/Courage97](https://github.com/Courage97)
Email: oyewolebarnabas97@gmail.com

---

*Built for academic research purposes. Dataset is synthetic and
calibrated to published Nigerian educational statistics.*