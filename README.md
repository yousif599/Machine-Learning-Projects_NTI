# 🤖 Machine Learning Projects Portfolio

A collection of end-to-end machine learning projects covering classification and
regression on real-world tabular data — from raw CSV to a cleaned dataset, full EDA,
feature engineering, multi-model comparison, hyperparameter tuning, and a final
evaluated model. Every project follows the same disciplined workflow so the notebooks
stay easy to navigate no matter which one you open first.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![CatBoost](https://img.shields.io/badge/CatBoost-yellow)
![XGBoost](https://img.shields.io/badge/XGBoost-green)
![LightGBM](https://img.shields.io/badge/LightGBM-9cf)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📁 Projects

| # | Project | Task | Dataset Size | Best Model | Key Result | Notebook |
|---|---------|------|---------------|------------|------------|----------|
| 1 | [Hotel Booking Cancellation](#1--hotel-booking-cancellation-prediction) | Classification | 74,361 rows | CatBoost | Accuracy 85.3% · ROC AUC 92.0% | `hotel-booking-cancellation/` |
| 2 | [Adult Census Income](#2--adult-census-income-prediction) | Classification | 32,537 rows | Model comparison (11 algorithms) | F1-optimized selection | `adult-census-income/` |
| 3 | [Bank Marketing / Term Deposit](#3--bank-marketing-term-deposit-prediction) | Classification | 8,912 rows | LightGBM | Accuracy 86.3% · ROC AUC 91.6% | `bank-marketing-deposit/` |
| 4 | [Stroke Prediction](#4--stroke-prediction) | Classification | 5,110 rows | LightGBM | Accuracy 94.6% | `stroke-prediction/` |
| 5 | [Alzheimer's Diagnosis](#5--alzheimers-diagnosis-prediction) | Classification | 2,149 rows | LightGBM | Accuracy 82.3% · ROC AUC 87.4% | `alzheimers-diagnosis/` |
| 6 | [Apartment Price Prediction](#6--apartment-price-prediction-egypt) | Regression | 5,578 rows | Random Forest | R² 0.30 | `apartment-price-prediction/` |
| 7 | [Smartphone Price Prediction](#7--smartphone-price-prediction) | Regression | 1,816 rows | Model comparison (14 algorithms + ensembles) | R²-optimized selection | `smartphone-price-prediction/` |

> Metrics above are the final, held-out test-set results reported inside each notebook.
> Projects marked *"Model comparison"* run a full leaderboard of algorithms and select
> the winner programmatically rather than assuming one model in advance.

---

## 🧭 Common Workflow

Every notebook in this repo follows the same structure, so once you're familiar with
one, you can navigate any of them:

1. **Data Understanding** — shape, types, summary statistics, missing values, duplicates
2. **Train / Test Split** — done *before* any preprocessing to avoid data leakage
3. **Missing Value Handling** — median/most-frequent imputation, fit on train only
4. **Exploratory Data Analysis** — target distribution, feature relationships, outliers, correlation
5. **Feature Engineering** — encoding strategy matched to each feature's cardinality and type (label / ordinal / frequency / one-hot)
6. **Feature Scaling** — `StandardScaler` on numerical columns
7. **Model Comparison** — multiple algorithms trained and scored on the same split
8. **Hyperparameter Tuning** — `RandomizedSearchCV` for the winning model
9. **Feature Selection** — importance-driven trimming of the final feature set
10. **Final Evaluation** — classification report / confusion matrix / ROC & PR curves, or MAE / RMSE / R² / residual plots for regression
11. **Deployment Artifacts** — trained model and every preprocessing object saved with `pickle` for reuse in an app

Classification projects additionally address **class imbalance** explicitly (via
`class_weight="balanced"`, `scale_pos_weight`, `auto_class_weights`, or SMOTE) and rank
models by **F1 score / recall rather than raw accuracy**, so the comparison doesn't
quietly reward a model that just leans on the majority class.

---

## 🗂️ Project Details

### 1 — Hotel Booking Cancellation Prediction
Predicts whether a hotel reservation will be canceled before arrival, using booking,
customer, and reservation-history features. Compares 13 algorithms (including Voting and
Stacking ensembles), tunes the winning **CatBoost** model, and ships as a working
**Streamlit web app** (`app.py`) that mirrors the exact training-time preprocessing at
inference time. Full technical write-up available in
[`Hotel_Booking_Cancellation_Documentation.pdf`](./hotel-booking-cancellation/Hotel_Booking_Cancellation_Documentation.pdf).

- **Target:** `is_canceled` (binary)
- **Result:** 85.27% accuracy · 91.98% ROC AUC · 68.9% recall on cancellations
- **Highlights:** feature-specific encoding (label / ordinal / frequency), class-weighting fix, dynamic best-model selection that adapts to whichever algorithm wins the comparison

### 2 — Adult Census Income Prediction
Predicts whether an individual's income exceeds $50K/year from U.S. Census data.
Compares 11 algorithms with class-imbalance handling built in from the start, and
frequency-encodes `native.country` instead of discarding it.

- **Target:** `income` (`<=50K` / `>50K`)
- **Highlights:** every categorical encoder stored per-column for reuse, explicit target mapping instead of implicit alphabetical ordering, models ranked by F1 rather than accuracy

### 3 — Bank Marketing / Term Deposit Prediction
Predicts whether a bank customer will subscribe to a term deposit following a marketing
campaign, using demographic, financial, and campaign-contact features.

- **Target:** `deposit` (`yes` / `no`)
- **Result:** 86.3% accuracy · 91.6% ROC AUC · 90.5% recall
- **Highlights:** LightGBM feature-importance-driven feature selection, two-stage tuning (`RandomizedSearchCV` → `GridSearchCV`)

### 4 — Stroke Prediction
Predicts stroke risk from patient health and demographic data — a highly imbalanced
medical dataset (~5% positive cases) addressed with **SMOTE** oversampling.

- **Target:** `stroke` (binary)
- **Result:** 94.6% accuracy with LightGBM
- **Highlights:** scaler comparison step before model selection, full model leaderboard sorted by Accuracy/F1

### 5 — Alzheimer's Diagnosis Prediction
Predicts an Alzheimer's diagnosis from clinical, lifestyle, and cognitive-assessment
features (MMSE, functional assessment, memory complaints, etc.).

- **Target:** `Diagnosis` (binary)
- **Result:** 82.3% accuracy · 87.4% ROC AUC
- **Highlights:** correlation-driven feature review, RandomizedSearchCV tuning, before/after tuning comparison table, empirical top-N feature-count selection

### 6 — Apartment Price Prediction (Egypt)
Predicts apartment sale prices (EGP) from area, location, and property features scraped
from a real-estate listings dataset.

- **Target:** `price` (continuous, EGP)
- **Result:** R² 0.30 (Random Forest)
- **Highlights:** data-leakage fix (dropped `price_per_sqm`, which is derived directly
  from the target), scaler comparison, `GridSearchCV` tuning
- **Note:** R² is modest — a good candidate for further feature engineering (e.g.
  location-based grouping, building age) in a future iteration.

### 7 — Smartphone Price Prediction
Predicts a smartphone's final listing price from brand, model, RAM, storage, color, and
shipping terms. Compares 14 regression algorithms plus Voting and Stacking ensembles.

- **Target:** `Final Price` (continuous)
- **Highlights:** single consistent "unknown" bucket for unseen brand/model categories
  at inference time (rather than arbitrary growing IDs), feature names preserved through
  scaling for interpretable feature importance, residual-plot diagnostics

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Data handling:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Modeling:** scikit-learn, XGBoost, LightGBM, CatBoost
- **Imbalance handling:** imbalanced-learn (SMOTE), native class-weighting
- **Deployment:** Streamlit
- **Environment:** Jupyter Notebook / Google Colab

---

## 🚀 Getting Started

Each project is self-contained in its own folder with its notebook and (where
applicable) its dataset and `app.py`. The gradient-boosting libraries aren't part of a
default Python install, so install them before running any notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost imbalanced-learn streamlit
```

**Running on Google Colab** (recommended — no local setup required):
1. Open the notebook in Colab
2. Run this in the first cell:
   ```python
   !pip install catboost xgboost lightgbm imbalanced-learn -q
   ```
3. Upload the project's dataset CSV when prompted, then **Runtime → Run all**

**Running the Hotel Booking Streamlit app locally:**
```bash
cd hotel-booking-cancellation
pip install streamlit pandas numpy scikit-learn catboost
streamlit run app.py
```

---

## 📂 Repository Structure

```
.
├── hotel-booking-cancellation/
│   ├── final_NTI_CANCELLATION_project.ipynb
│   ├── app.py
│   ├── Hotel_Booking_Cancellation_Documentation.pdf
│   └── hotel_bookings.csv
├── adult-census-income/
│   ├── Adult_Census_Income_Professional.ipynb
│   └── Adult_Census_Income.csv
├── bank-marketing-deposit/
│   ├── Bank_Deposit_Prediction.ipynb
│   └── bank.csv
├── stroke-prediction/
│   ├── Stroke_Prediction.ipynb
│   └── healthcare-dataset-stroke-data.csv
├── alzheimers-diagnosis/
│   └── Alzheimers_Diagnosis_Prediction.ipynb
├── apartment-price-prediction/
│   ├── Apartment_Price_Prediction_Project.ipynb
│   └── Apartments_Prices_Dataset.csv
├── smartphone-price-prediction/
│   ├── Smartphones_Price_Professional.ipynb
│   └── smartphones.csv
└── README.md
```

> Folder names above are suggested for a clean repo layout — rename to match however
> you've actually organized the files.

---

## 📌 Notes on Data

Some datasets (e.g. `hotel_bookings.csv`, `Adult_Census_Income.csv`) are large enough
that you may prefer to add them to `.gitignore` and instead document the public source
each notebook downloads from, to keep the repository lightweight. If you'd rather commit
them directly, [Git LFS](https://git-lfs.com/) is worth using for anything over ~50MB.

---

## 📄 License

This repository is available under the [MIT License](./LICENSE).

## 🙋 Author

Maintained as a personal machine-learning portfolio. Feedback and pull requests are welcome.
