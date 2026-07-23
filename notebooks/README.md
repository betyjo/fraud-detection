# Notebooks

Run notebooks in the order listed. Each notebook saves outputs consumed by the next.

---

## 1. `eda-fraud-data.ipynb`
**Purpose:** Exploratory data analysis on the e-commerce fraud dataset.  
**Inputs:** `data/raw/Fraud_Data.csv`, `data/raw/IpAddress_to_Country.csv`  
**Outputs:** Visualizations (class imbalance, purchase value distributions, fraud by browser/source/country, time patterns). No files saved.  
**Key steps:** Load raw data, check nulls and dtypes, visualize fraud rate across categorical features, inspect time-based patterns.

---

## 2. `eda-creditcard.ipynb`
**Purpose:** Exploratory data analysis on the anonymized bank credit card dataset.  
**Inputs:** `data/raw/creditcard.csv`  
**Outputs:** Visualizations (class imbalance, feature distributions, correlation heatmap). No files saved.  
**Key steps:** Inspect PCA-transformed features (V1–V28), check class imbalance (~0.17% fraud), examine Amount and Time distributions.

---

## 3. `feature-engineering.ipynb`
**Purpose:** Build model-ready features from raw data.  
**Inputs:** `data/raw/Fraud_Data.csv`, `data/raw/IpAddress_to_Country.csv`, `data/raw/creditcard.csv`  
**Outputs:**
- `data/processed/fraud_processed.csv` — fully engineered and one-hot encoded
- `data/processed/creditcard_processed.csv` — deduplicated, **unscaled** (scaling deferred to modeling)

**Key steps:**
- IP geolocation merge (`merge_asof`)
- `time_since_signup` with negative-value clipping and flag column
- `transaction_velocity`: rolling 24-hour device transaction count
- User/device transaction count and `users_per_device`
- One-hot encoding for `source`, `browser`, `sex`, `country`
- **No StandardScaler applied to creditcard** — scaling is done in `modeling.ipynb` after the split to prevent leakage

---

## 4. `modeling.ipynb`
**Purpose:** Train and evaluate fraud detection models.  
**Inputs:** `data/processed/fraud_processed.csv`, `data/processed/creditcard_processed.csv`  
**Outputs:**
- `models/xgb_fraud_model.pkl` — best tuned XGBoost model
- `models/scaler_fraud.pkl` — fitted StandardScaler for fraud features
- `models/xgb_credit_model.pkl` — XGBoost for creditcard
- `models/scaler_credit.pkl`
- `data/processed/X_test_fraud.csv`, `data/processed/y_test_fraud.csv` — held-out test set for SHAP

**Key steps:**
- Stratified 80/20 split (`random_state=42`)
- StandardScaler fit on `X_train` only (no leakage)
- SMOTE applied to training set only
- Logistic Regression with `class_weight="balanced"`
- Random Forest (200 trees, max_depth=10)
- XGBoost with `RandomizedSearchCV` (9 iterations, 3-fold CV, `average_precision` scoring)
- StratifiedKFold CV (k=5) for all three models
- Full comparison table: Model, Test_F1, Test_AUC_PR, CV_AUC_PR_Mean, CV_AUC_PR_Std

---

## 5. `shap-explainability.ipynb`
**Purpose:** Explain XGBoost fraud model predictions using SHAP.  
**Inputs:** `models/xgb_fraud_model.pkl`, `data/processed/X_test_fraud.csv`, `data/processed/y_test_fraud.csv`  
**Outputs:**
- `models/feature_importance_top10.png` — built-in feature importance bar chart
- `models/feature_importance_top10.csv`
- `models/shap_summary_bar.png` — SHAP mean |value| bar plot
- `models/shap_summary_beeswarm.png` — SHAP beeswarm plot
- `models/shap_waterfall_true_positive.png`
- `models/shap_waterfall_false_positive.png`
- `models/shap_waterfall_false_negative.png`

**Key steps:**
- SHAP TreeExplainer on test set (subsample 2000 rows)
- Waterfall plots for TP, FP, FN cases with predicted probability in title
- SHAP vs built-in importance comparison table; highlights features in SHAP top 5 but not built-in top 10
- Business recommendations based on SHAP findings
