# How to Run the Fraud Detection Project

This guide walks you through every step from environment setup to running the full pipeline end-to-end.

---

## Prerequisites

- Python 3.9 or higher (3.10 recommended)
- Git
- ~4 GB free RAM minimum (8 GB recommended for the creditcard dataset)
- The three raw data files (download links in the project brief)

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/betyjo/fraud-detection.git
cd fraud-detection
```

---

## Step 2 — Create a Virtual Environment (Recommended)

A virtual environment keeps the project dependencies isolated from your system Python.

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt after activation.

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs everything: pandas, numpy, scikit-learn, imbalanced-learn, xgboost, shap, matplotlib, seaborn, jupyter, and all other dependencies pinned to the exact versions used in development.

This will take 2–5 minutes on a typical connection.

---

## Step 4 — Add the Raw Data Files

The raw data files are excluded from version control (they are in `.gitignore`). Download them and place them in the `data/raw/` folder:

```
fraud-detection/
└── data/
    └── raw/
        ├── Fraud_Data.csv
        ├── IpAddress_to_Country.csv
        └── creditcard.csv
```

**Download links:**
- `Fraud_Data.csv` — [Kaggle: Fraud E-commerce Dataset](https://www.kaggle.com/datasets/vbinh/fraud-detection-e-commerce)
- `IpAddress_to_Country.csv` — included with the same dataset above
- `creditcard.csv` — [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

If the `data/raw/` folder does not exist yet, create it:

```bash
mkdir -p data/raw
```

**Windows:**
```cmd
mkdir data\raw
```

---

## Step 5 — Launch Jupyter Lab

```bash
jupyter lab
```

This opens Jupyter Lab in your browser at `http://localhost:8888`. All notebooks are in the `notebooks/` folder.

Alternatively, if you prefer classic Jupyter Notebook:
```bash
jupyter notebook
```

---

## Step 6 — Run the Notebooks in Order

**Important:** Run the notebooks strictly in this order. Each notebook saves outputs that the next one depends on.

---

### Notebook 1 — `eda-fraud-data.ipynb`

**Purpose:** Explore the raw e-commerce fraud dataset.

**What it does:**
- Loads `data/raw/Fraud_Data.csv`
- Displays shape, data types, missing values, duplicate count
- Shows class distribution (count + percentage) with countplot
- Plots univariate distributions of `purchase_value` and `age`
- Plots bivariate comparisons (boxplots, countplots) by `class`
- Flags dataset as highly imbalanced

**Inputs:** `data/raw/Fraud_Data.csv`
**Outputs:** Visualizations displayed inline (no files saved)
**Run time:** ~30 seconds

---

### Notebook 2 — `eda-creditcard.ipynb`

**Purpose:** Explore the bank credit card dataset.

**What it does:**
- Loads `data/raw/creditcard.csv`
- Checks shape, types, missing values, duplicates
- Shows class distribution (0.17% fraud rate — extreme imbalance)
- Plots `Amount` distribution
- Generates correlation heatmap for all V1–V28 features vs `Class`
- Plots boxplots for the 5 most correlated features (V4, V10, V12, V14, V17) by class

**Inputs:** `data/raw/creditcard.csv`
**Outputs:** Visualizations displayed inline (no files saved)
**Run time:** ~1–2 minutes (correlation matrix on 284k rows takes a moment)

---

### Notebook 3 — `feature-engineering.ipynb`

**Purpose:** Engineer all features and save processed datasets.

**What it does:**

For `Fraud_Data.csv`:
- Parses datetime columns
- Converts IP addresses to integers
- Merges with `IpAddress_to_Country.csv` using `merge_asof` (fast range lookup)
- Assigns `"Unknown"` to unmatched IPs
- Computes `time_since_signup` (seconds), clipping negative values to 0
- Adds `time_since_signup_flag` for anomalous rows
- Adds `hour_of_day`, `day_of_week`, `is_weekend`
- Adds `user_transaction_count`, `device_transaction_count`, `users_per_device`
- Adds `seconds_since_previous_transaction`
- Adds `transaction_velocity` (24-hour rolling device window)
- Drops raw columns that are no longer needed (ip_address, device_id, timestamps, bounds)
- One-hot encodes `source`, `browser`, `sex`, `country` with `drop_first=True`
- Saves to `data/processed/fraud_processed.csv`

For `creditcard.csv`:
- Removes duplicates only
- Does NOT scale — scaling happens in `modeling.ipynb` after the split to prevent leakage
- Saves to `data/processed/creditcard_processed.csv`

**Inputs:**
- `data/raw/Fraud_Data.csv`
- `data/raw/IpAddress_to_Country.csv`
- `data/raw/creditcard.csv`

**Outputs:**
- `data/processed/fraud_processed.csv`
- `data/processed/creditcard_processed.csv`

**Run time:** ~2–4 minutes (the `transaction_velocity` rolling window is the slowest cell)

> **Note:** If the `transaction_velocity` cell is slow, this is expected. It iterates over device groups to compute a 24-hour rolling count. On the full dataset (~150k rows) it takes 1–3 minutes depending on your machine.

---

### Notebook 4 — `modeling.ipynb`

**Purpose:** Train all models, tune XGBoost, evaluate, and save artifacts.

**What it does:**

**Part 1 — Fraud_Data:**
1. Loads `data/processed/fraud_processed.csv`
2. Separates features (`X`) and target (`y = class`)
3. Stratified 80/20 train-test split (`random_state=42`)
4. Fits `StandardScaler` on `X_train` only, transforms both sets
5. Applies SMOTE to `X_train` only — prints before/after class counts
6. Trains **Logistic Regression** (`class_weight="balanced"`, `max_iter=1000`)
7. Trains **Random Forest** (`n_estimators=200`, `max_depth=10`)
8. Trains **XGBoost** with `RandomizedSearchCV` (9 iterations, scoring=AUC-PR)
9. Prints best hyperparameters and best CV score from the search
10. Evaluates all 3 models: F1, AUC-PR, labeled confusion matrix heatmap
11. Runs StratifiedKFold k=5 CV on all 3 models — prints mean ± std AUC-PR
12. Prints the full comparison table
13. Prints model selection justification
14. Saves: `models/xgb_fraud_model.pkl`, `models/scaler_fraud.pkl`
15. Saves: `data/processed/X_test_fraud.csv`, `data/processed/y_test_fraud.csv`

**Part 2 — CreditCard:**
1. Loads `data/processed/creditcard_processed.csv`
2. Same pipeline: stratified split → scale on train → SMOTE on train
3. Trains XGBoost with the best params found in Part 1
4. Evaluates: F1, AUC-PR, confusion matrix
5. Runs StratifiedKFold k=5 CV
6. Saves: `models/xgb_credit_model.pkl`, `models/scaler_credit.pkl`

**Inputs:**
- `data/processed/fraud_processed.csv`
- `data/processed/creditcard_processed.csv`

**Outputs:**
- `models/xgb_fraud_model.pkl`
- `models/xgb_credit_model.pkl`
- `models/scaler_fraud.pkl`
- `models/scaler_credit.pkl`
- `data/processed/X_test_fraud.csv`
- `data/processed/y_test_fraud.csv`

**Run time:** ~10–20 minutes total
- Logistic Regression: ~30 seconds
- Random Forest: ~2–3 minutes
- XGBoost RandomizedSearchCV (9 iter × 3 folds): ~5–8 minutes
- StratifiedKFold CV for all 3 models: ~5–8 minutes
- CreditCard pipeline: ~3–5 minutes

> **Tip:** You can reduce run time by lowering `n_iter` in `RandomizedSearchCV` from 9 to 3, or by reducing `n_splits` in `StratifiedKFold` from 5 to 3. The results will be less thorough but faster.

---

### Notebook 5 — `shap-explainability.ipynb`

**Purpose:** Explain the model using SHAP and produce business insights.

**What it does:**
1. Loads `models/xgb_fraud_model.pkl` and the saved test set
2. Computes predictions on the test set
3. **Built-in feature importance:** Extracts `model.feature_importances_`, plots top 10 as horizontal bar chart
4. Saves top 10 chart as `models/feature_importance_top10.png` and table as `models/feature_importance_top10.csv`
5. **SHAP TreeExplainer:** Subsamples up to 2,000 test rows, computes SHAP values
6. **Summary bar plot:** Top 20 features by mean |SHAP| — saved as `models/shap_summary_bar.png`
7. **Beeswarm plot:** Distribution of SHAP values per feature — saved as `models/shap_summary_beeswarm.png`
8. Prints top 5 fraud drivers by mean |SHAP|
9. **Waterfall plots for 3 cases:**
   - True Positive (fraud correctly caught): `models/shap_waterfall_true_positive.png`
   - False Positive (legitimate flagged): `models/shap_waterfall_false_positive.png`
   - False Negative (fraud missed): `models/shap_waterfall_false_negative.png`
10. **Comparison table:** SHAP rank vs built-in rank for all features, flags discrepancies
11. **Business recommendations:** 5 concrete recommendations in a markdown cell

**Inputs:**
- `models/xgb_fraud_model.pkl`
- `data/processed/X_test_fraud.csv`
- `data/processed/y_test_fraud.csv`

**Outputs:**
- `models/feature_importance_top10.png`
- `models/feature_importance_top10.csv`
- `models/shap_summary_bar.png`
- `models/shap_summary_beeswarm.png`
- `models/shap_waterfall_true_positive.png`
- `models/shap_waterfall_false_positive.png`
- `models/shap_waterfall_false_negative.png`

**Run time:** ~3–8 minutes (SHAP value computation on 2,000 rows is the slow step)

---

## Step 7 — Run the Unit Tests

The unit tests verify the core logic of all source modules without requiring the raw data files. They use synthetic in-memory DataFrames.

```bash
python -m pytest tests/ -v
```

Expected output: **35 passed** in ~12 seconds.

```
tests/test_feature_engineering.py::TestAddTimeFeatures::test_add_time_features_columns_exist PASSED
tests/test_feature_engineering.py::TestAddTimeFeatures::test_time_since_signup_negative_clipped PASSED
... (35 total)
35 passed in 12.03s
```

To run a specific test file only:
```bash
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_feature_engineering.py -v
python -m pytest tests/test_model_utils.py -v
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'xgboost'"
You forgot to activate the virtual environment, or `pip install -r requirements.txt` did not complete successfully.
```bash
# Activate venv first, then:
pip install -r requirements.txt
```

### "FileNotFoundError: data/raw/Fraud_Data.csv"
The raw data files are not in `data/raw/`. Download them from Kaggle and place them there. See Step 4.

### "FileNotFoundError: data/processed/fraud_processed.csv"
You need to run `feature-engineering.ipynb` before `modeling.ipynb`. Run notebooks in order.

### "FileNotFoundError: models/xgb_fraud_model.pkl"
You need to run `modeling.ipynb` before `shap-explainability.ipynb`. Run notebooks in order.

### The `transaction_velocity` cell takes too long
This is expected. The rolling 24h window iterates over device groups. On the full Fraud_Data dataset it takes 1–3 minutes. If you need it faster, you can skip this cell — the rest of the notebook will still work (the model will just not have `transaction_velocity` as a feature).

### Jupyter Lab won't start
Make sure your virtual environment is activated and jupyter is installed:
```bash
pip install jupyterlab
jupyter lab
```

### SMOTE raises a ValueError about sample counts
This happens if a class has fewer than `k_neighbors` (default 5) samples. Unlikely with the provided datasets but can occur if you filter the data. Use `SMOTE(k_neighbors=3)` if this happens.

### XGBoost warns "use_label_encoder is deprecated"
This warning is safe to ignore — it is a version compatibility notice and does not affect results.

---

## Quick Reference — File Dependency Map

```
data/raw/Fraud_Data.csv ──────────────────────┐
data/raw/IpAddress_to_Country.csv ────────────┤──► feature-engineering.ipynb
data/raw/creditcard.csv ──────────────────────┘
                                               │
                          data/processed/fraud_processed.csv ──┐
                          data/processed/creditcard_processed.csv ──┤──► modeling.ipynb
                                                               │
                          models/xgb_fraud_model.pkl ─────────┐
                          models/scaler_fraud.pkl ─────────────┤──► shap-explainability.ipynb
                          data/processed/X_test_fraud.csv ─────┤
                          data/processed/y_test_fraud.csv ─────┘
```

---

## CI/CD — GitHub Actions

The repository includes `.github/workflows/unittests.yml` which automatically runs `pytest tests/` on every push to GitHub. You can see the test results in the **Actions** tab of your repository at `https://github.com/betyjo/fraud-detection/actions`.
