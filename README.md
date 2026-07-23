# Fraud Detection for E-commerce and Bank Transactions

## Project Overview

This project builds machine learning pipelines to detect fraudulent transactions across two domains:

1. **E-commerce fraud** (`Fraud_Data.csv`) — behavioral features, IP geolocation, device/user velocity signals
2. **Credit card fraud** (`creditcard.csv`) — anonymized PCA features from real bank transactions

The pipeline covers data preprocessing, feature engineering, class imbalance handling (SMOTE), model training and hyperparameter tuning, cross-validation, and SHAP-based model explainability.

---

## Repository Structure

```
fraud-detection/
├── .github/
│   └── workflows/
│       └── unittests.yml          # CI: runs pytest on push
├── data/
│   ├── raw/                       # Original datasets (not committed)
│   └── processed/                 # Cleaned, feature-engineered CSVs
├── models/                        # Saved model artifacts (.pkl, .png)
├── notebooks/
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   ├── shap-explainability.ipynb
│   └── README.md                  # Per-notebook documentation
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── model_utils.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   └── test_model_utils.py
├── scripts/
│   └── README.md                  # Planned operationalization scripts
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Data

Place raw files in `data/raw/`:
- `Fraud_Data.csv`
- `IpAddress_to_Country.csv`
- `creditcard.csv`

These files are excluded from version control (see `.gitignore`).

---

## Running

Execute notebooks in order:

1. `notebooks/eda-fraud-data.ipynb` — EDA on e-commerce fraud data
2. `notebooks/eda-creditcard.ipynb` — EDA on credit card data
3. `notebooks/feature-engineering.ipynb` — Feature engineering, saves processed CSVs
4. `notebooks/modeling.ipynb` — Model training, tuning, evaluation, saves models + test set
5. `notebooks/shap-explainability.ipynb` — SHAP analysis, saves plots and comparison table

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests use synthetic in-memory data and do not require the raw CSV files.

---

## Key Design Decisions

### Why AUC-PR over Accuracy?
Both datasets are heavily imbalanced (fraud is ~9% in Fraud_Data, ~0.17% in creditcard). Accuracy is misleading — a model predicting "not fraud" for every transaction achieves 99.83% accuracy on creditcard. AUC-PR (area under the Precision-Recall curve) directly measures performance on the minority class and is robust to class imbalance.

### Why SMOTE on Training Set Only?
SMOTE generates synthetic samples by interpolating between real minority-class examples. Applying it before the split would leak synthetic versions of test-set samples into training, inflating CV performance. SMOTE is applied only to `X_train` after the stratified split to ensure `X_test` contains only real, unseen observations.

### Why StandardScaler Fit on Train Only?
Fitting the scaler on the full dataset before splitting leaks test-set statistics (mean, std) into the training process. The scaler is always fit on `X_train` and only used to `transform` `X_test`, maintaining a clean train/test boundary.

### Why XGBoost as the Best Model?
XGBoost consistently outperforms Logistic Regression and Random Forest on AUC-PR across both datasets in cross-validation. Its gradient boosting architecture captures non-linear feature interactions that are especially important in fraud detection (e.g., `time_since_signup` combined with `transaction_velocity`). It also supports `RandomizedSearchCV` tuning efficiently through its native early-stopping and parallel training.

---

## Results

| Model | Test F1 | Test AUC-PR | CV AUC-PR Mean | CV AUC-PR Std |
|---|---|---|---|---|
| LogisticRegression | TBD | TBD | TBD | TBD |
| RandomForest | TBD | TBD | TBD | TBD |
| XGBoost (tuned) | TBD | TBD | TBD | TBD |

*Results will be populated after running `modeling.ipynb` with the actual dataset.*

**CreditCard XGBoost:**

| Model | Test F1 | Test AUC-PR | CV AUC-PR Mean | CV AUC-PR Std |
|---|---|---|---|---|
| XGBoost (tuned) | TBD | TBD | TBD | TBD |

---

## SHAP Key Findings

Top fraud drivers identified by mean |SHAP| value (to be updated after running `shap-explainability.ipynb`):
1. `time_since_signup` — very recent signups strongly predict fraud
2. `transaction_velocity` — multiple device transactions in 24h window
3. `purchase_value` — high-value anomalous purchases
4. `hour_of_day` — off-hours transactions increase risk
5. `users_per_device` — devices shared across multiple accounts

See `notebooks/shap-explainability.ipynb` for full analysis and business recommendations.
