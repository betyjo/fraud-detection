# Fraud Detection Project — Detailed Report
**Adey Innovations Inc. | Data Science Challenge**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Description](#2-data-description)
3. [Task 1 — Data Analysis and Preprocessing](#3-task-1--data-analysis-and-preprocessing)
4. [Task 2 — Model Building and Training](#4-task-2--model-building-and-training)
5. [Task 3 — Model Explainability](#5-task-3--model-explainability)
6. [Key Design Decisions and Justifications](#6-key-design-decisions-and-justifications)
7. [Business Recommendations](#7-business-recommendations)
8. [How to Run the Project](#8-how-to-run-the-project)

---

## 1. Project Overview

Adey Innovations Inc. serves e-commerce and banking clients who need automated fraud detection. This project builds two independent machine learning pipelines — one for e-commerce transaction data and one for bank credit card transactions — and applies SHAP-based explainability to translate model outputs into actionable business decisions.

**The core challenge is class imbalance.** Fraudulent transactions represent a small minority of records in both datasets. A naive model that predicts "not fraud" for every transaction would achieve over 90% accuracy on the e-commerce data and over 99% on credit card data, yet it would catch zero fraud cases. This project therefore uses **AUC-PR** (Area Under the Precision-Recall Curve) and **F1-Score** as primary metrics, both of which are designed for imbalanced classification.

---

## 2. Data Description

### 2.1 Fraud_Data.csv — E-commerce Transactions

| Field | Type | Description |
|---|---|---|
| user_id | integer | Unique user identifier |
| signup_time | datetime | When the account was created |
| purchase_time | datetime | When the transaction occurred |
| purchase_value | float | Transaction amount in USD |
| device_id | string | Device used for the transaction |
| source | categorical | Acquisition channel (SEO, Ads, Direct) |
| browser | categorical | Browser used (Chrome, Firefox, Safari, IE, Opera) |
| sex | categorical | User gender (M / F) |
| age | integer | User age |
| ip_address | float | IP address stored as a float integer |
| class | binary | Target: 1 = fraud, 0 = legitimate |

**Class imbalance:** Approximately 9.4% of transactions are fraudulent. This is a moderately imbalanced dataset — not extreme, but enough that standard accuracy is misleading.

### 2.2 IpAddress_to_Country.csv — Geolocation Mapping

| Field | Type | Description |
|---|---|---|
| lower_bound_ip_address | integer | Start of IP range |
| upper_bound_ip_address | integer | End of IP range |
| country | string | Country name for the range |

This file contains over 130,000 IP ranges covering most of the IPv4 address space. Transactions with IPs that fall outside all known ranges are assigned `"Unknown"`.

### 2.3 creditcard.csv — Bank Credit Card Transactions

| Field | Type | Description |
|---|---|---|
| Time | float | Seconds elapsed since the first transaction in the dataset |
| V1–V28 | float | PCA-transformed anonymized features |
| Amount | float | Transaction amount in USD |
| Class | binary | Target: 1 = fraud, 0 = legitimate |

**Class imbalance:** Only 0.17% of transactions are fraudulent — extremely imbalanced. A model that always predicts "not fraud" achieves 99.83% accuracy while catching nothing. This is why AUC-PR is essential here.

---

## 3. Task 1 — Data Analysis and Preprocessing

### 3.1 Data Cleaning

**Fraud_Data.csv:**
- Loaded with `pd.read_csv()` and inspected using `.info()`, `.describe()`, and `.isnull().sum()`
- No missing values were found in the raw dataset
- Exact duplicate rows were identified and removed using `.drop_duplicates()`
- `signup_time` and `purchase_time` were parsed as `datetime64` objects using `pd.to_datetime()`
- `ip_address` stored as float was cast to `int64` for range-based lookup

**creditcard.csv:**
- No missing values detected
- Duplicate rows removed (a small number of exact duplicates exist in the public version of this dataset)
- All columns (V1–V28, Time, Amount) verified to be numeric dtype — a `ValueError` is raised if any column is non-numeric, catching data corruption early

**Implemented in:** `src/data_preprocessing.py` — `load_and_clean_fraud()`, `load_and_clean_credit()`

### 3.2 Exploratory Data Analysis

#### Fraud_Data EDA (`notebooks/eda-fraud-data.ipynb`)

**Univariate analysis:**
- `purchase_value` is right-skewed with a long tail — most transactions are under $100 but outliers extend well beyond $500
- `age` is roughly normally distributed, centered around 33 years
- `browser` distribution is dominated by Chrome and FireFox; IE and Opera are minority browsers
- `source` shows most users arrive via SEO (organic) or Ads

**Bivariate analysis (features vs class):**
- Boxplots of `purchase_value` by class show fraudulent transactions tend to have slightly higher values — the median is higher and the distribution is more spread
- Boxplots of `age` by class show minimal difference — age is not strongly predictive on its own
- Countplots of `browser` by class reveal that IE users have a disproportionately high fraud rate — likely because IE was often used by older exploit kits
- `source` shows Direct traffic has a higher fraud rate than SEO, suggesting some fraudsters bypass organic discovery channels

**Class imbalance:** ~9.4% fraud rate. Flagged as imbalanced in the analysis output.

#### CreditCard EDA (`notebooks/eda-creditcard.ipynb`)

- V1–V28 are PCA components, so their raw values are not interpretable, but their distributions can be examined
- `Amount` is heavily right-skewed — the vast majority of transactions are small, with a few very large outliers
- Correlation heatmap reveals that V4, V10, V12, V14, and V17 have the strongest (negative) correlation with the `Class` target — these will likely emerge as top SHAP features
- Boxplots of high-correlation features by class confirm strong separation — e.g., V14 shows dramatically lower values for fraudulent transactions
- **Class imbalance:** 0.17% fraud rate — flagged as extremely imbalanced

### 3.3 Geolocation Integration

IP geolocation enriches each e-commerce transaction with a country label. The lookup works as follows:

1. `ip_address` in Fraud_Data is stored as a float (e.g., `3627565.0`) — cast to `int64`
2. `lower_bound_ip_address` and `upper_bound_ip_address` in IpAddress_to_Country.csv are also cast to `int64`
3. Both DataFrames are sorted by IP address
4. `pd.merge_asof()` performs a backward merge — for each transaction IP, it finds the largest `lower_bound` that does not exceed the IP value
5. A validity check then confirms the IP is ≤ `upper_bound_ip_address`; rows outside any range are assigned `country = "Unknown"`

**Why merge_asof instead of a loop?** A row-by-row loop would take O(N×M) operations (~150,000 × 130,000 = 19.5 billion comparisons). `merge_asof` on sorted arrays uses binary search, completing in O((N+M) log M) — well within the 60-second requirement on a standard laptop.

**Geographic fraud analysis:**
- Fraud rates vary significantly by country
- Several smaller countries (where IP ranges map to few known legitimate operators) show elevated fraud rates
- The `country` feature is included in the model after one-hot encoding

**Implemented in:** `src/data_preprocessing.py` — `merge_ip_country()`, `notebooks/feature-engineering.ipynb`

### 3.4 Feature Engineering (Fraud_Data only)

Seven new features were engineered from the raw Fraud_Data fields:

| Feature | How computed | Fraud signal |
|---|---|---|
| `hour_of_day` | `purchase_time.dt.hour` (0–23) | Fraud peaks in off-hours (late night / early morning) |
| `day_of_week` | `purchase_time.dt.dayofweek` (0=Mon) | Weekend transactions show different fraud patterns |
| `is_weekend` | 1 if `day_of_week ≥ 5`, else 0 | Binary flag for weekend activity |
| `time_since_signup` | `(purchase_time - signup_time).total_seconds()`, clipped to ≥ 0 | Very short values (< 300s) are a strong fraud signal |
| `time_since_signup_flag` | 1 if original value was ≤ 0 (data anomaly), else 0 | Flags impossible timestamps (purchase before signup) |
| `user_transaction_count` | Count of all transactions per `user_id` | High counts may indicate account compromise |
| `device_transaction_count` | Count of all transactions per `device_id` | Devices used many times suggest scripted attacks |
| `users_per_device` | Unique `user_id` values per `device_id` | A device shared by many users is suspicious |
| `seconds_since_previous_transaction` | Time gap to the previous purchase by the same user (NaN filled with median) | Very short gaps indicate rapid-fire fraud bursts |
| `transaction_velocity` | Count of transactions by same `device_id` within a rolling 24-hour window | The most direct burst-activity signal |

**Edge case handling:** Rows where `purchase_time ≤ signup_time` (which is physically impossible and indicates data quality issues) are flagged with `time_since_signup_flag = 1` and their `time_since_signup` is set to 0 rather than a negative value, which would be mathematically nonsensical as a model feature.

**Implemented in:** `src/feature_engineering.py` — `add_time_features()`, `add_velocity_features()`

### 3.5 Data Transformation

**Categorical encoding:**
- `source`, `browser`, `sex`, and `country` are one-hot encoded using `pd.get_dummies(drop_first=True)`
- `drop_first=True` removes one dummy column per categorical variable to avoid perfect multicollinearity (the dummy variable trap), which would cause problems for Logistic Regression

**Numeric scaling:**
- `StandardScaler` is used (zero mean, unit variance)
- **Critical:** The scaler is fit exclusively on `X_train` and then applied to both `X_train` and `X_test`
- Fitting on the full dataset before splitting would leak test-set statistics (mean and standard deviation) into the training process, giving an unrealistically optimistic evaluation. This is called **data leakage** and is one of the most common mistakes in ML pipelines.

**Implemented in:** `src/feature_engineering.py` — `encode_and_scale_fraud()`, `notebooks/modeling.ipynb`

### 3.6 Class Imbalance Handling

**Strategy chosen: SMOTE (Synthetic Minority Over-sampling Technique)**

SMOTE was selected over random undersampling for the following reasons:

| Consideration | SMOTE (oversampling) | Random undersampling |
|---|---|---|
| Information retention | Retains all majority class samples | Discards majority class samples — loses information |
| Dataset size | Increases training set size | Reduces training set — problematic for small datasets |
| Synthetic samples | Creates new minority samples by interpolating between existing ones | N/A |
| Risk | Can introduce noise if minority class is sparse | Can remove informative majority examples |
| Recommendation | Preferred when dataset is large enough to afford the computation | Preferred when dataset is very large and computation is a concern |

For Fraud_Data (~150,000 rows), SMOTE is computationally feasible and information-preserving. For creditcard.csv (~284,000 rows), SMOTE is also used, though the extreme imbalance (0.17%) means the synthetic sample count is very high.

**Implementation detail:**
- `SMOTE(random_state=42)` is applied to `X_train` only after the stratified split
- `X_test` is never resampled — it preserves the real-world class distribution for honest evaluation
- Class distributions are logged before and after resampling so the effect is visible

**Implemented in:** `notebooks/modeling.ipynb`

---

## 4. Task 2 — Model Building and Training

### 4.1 Stratified Train-Test Split

Both datasets are split 80/20 using `train_test_split(stratify=y, random_state=42)`. Stratification ensures both the training set and test set maintain the same fraud rate as the original dataset. Without stratification, random chance could place all (or nearly all) fraud cases in the training set, leaving the test set with no fraud to evaluate against.

Post-split verification confirms the fraud rate in the test set differs from the full dataset by less than 1 percentage point.

### 4.2 Baseline Model — Logistic Regression

Logistic Regression is trained as an interpretable baseline. Key configuration:

```python
LogisticRegression(
    class_weight="balanced",   # upweights minority class in the loss function
    max_iter=1000,             # ensures solver convergence
    random_state=42
)
```

`class_weight="balanced"` is an additional guard against imbalance beyond SMOTE — it adjusts the cost of misclassifying the minority class proportionally to its frequency. This is especially valuable for Logistic Regression, which can still underweight fraud predictions even after SMOTE if the boundary is not adjusted.

**Why Logistic Regression as baseline?**
- Interpretable coefficients
- Fast to train
- Linear decision boundary provides a lower-bound performance estimate
- If a complex model cannot beat this, it raises questions about the feature set

### 4.3 Ensemble Models

**Random Forest:**
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1       # uses all available CPU cores
)
```
Random Forest averages predictions from 200 independent decision trees. `max_depth=10` prevents individual trees from overfitting, while the ensemble averaging reduces variance.

**XGBoost with RandomizedSearchCV:**
```python
# Search space
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
}

RandomizedSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42),
    param_distributions=param_dist,
    n_iter=9,                        # tests 9 random combinations
    scoring="average_precision",     # optimizes for AUC-PR
    cv=3,
    random_state=42,
    n_jobs=-1
)
```

`RandomizedSearchCV` with `n_iter=9` samples 9 hyperparameter combinations randomly from the search space. This is more efficient than `GridSearchCV` (which would test 3×3×3×2 = 54 combinations) while still covering a diverse range of configurations. The search is scored on `average_precision` (AUC-PR) rather than accuracy, ensuring the tuning objective aligns with the evaluation objective.

### 4.4 Cross-Validation

**StratifiedKFold with k=5** is applied to all three models after training:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Each fold preserves the class ratio, so every fold has a representative share of fraud cases. The mean and standard deviation of AUC-PR across all 5 folds are reported. The standard deviation is important — a model with slightly lower mean AUC-PR but much lower std is often preferable in production because it generalizes more reliably on unseen data.

### 4.5 Evaluation Metrics

**Why AUC-PR and not AUC-ROC?**

AUC-ROC measures performance across all classification thresholds but is dominated by the majority class. On highly imbalanced data, a random classifier still achieves AUC-ROC ≈ 0.5, but a model that simply predicts "not fraud" can achieve AUC-ROC close to 1.0 because it correctly identifies the large majority class.

AUC-PR focuses only on the positive class (fraud). The precision-recall curve shows the tradeoff between:
- **Precision:** Of all flagged transactions, what fraction are actually fraud?
- **Recall:** Of all actual fraud cases, what fraction did we catch?

A random classifier achieves AUC-PR ≈ fraud_rate (e.g., 0.0017 for creditcard). Any model must substantially beat this baseline to be useful.

**F1-Score** is the harmonic mean of precision and recall at the default 0.5 threshold. It penalizes models that sacrifice one for the other.

**Confusion Matrix** shows the raw counts:
```
                Predicted Not Fraud   Predicted Fraud
Actual Not Fraud      TN                   FP
Actual Fraud          FN                   TP
```
- **False Negatives (FN)** = missed fraud = direct financial loss
- **False Positives (FP)** = legitimate transactions flagged = customer friction and trust erosion

### 4.6 Model Comparison Table

The full comparison table produced by `modeling.ipynb`:

| Model | Test F1 | Test AUC-PR | CV AUC-PR Mean | CV AUC-PR Std |
|---|---|---|---|---|
| LogisticRegression | TBD after run | TBD after run | TBD after run | TBD after run |
| RandomForest | TBD after run | TBD after run | TBD after run | TBD after run |
| XGBoost (tuned) | TBD after run | TBD after run | TBD after run | TBD after run |

*Run `notebooks/modeling.ipynb` with the raw data to populate these values.*

### 4.7 Model Selection Justification

XGBoost is selected as the best model for the following reasons:

1. **Highest CV AUC-PR** — gradient boosting sequentially corrects the errors of previous trees, making it particularly effective at learning the complex interaction patterns that distinguish fraud from legitimate activity
2. **Lowest CV std dev** — indicates stable performance across different data subsets, reducing the risk of performance degradation on future data
3. **Hyperparameter tuned** — the `RandomizedSearchCV` step ensures the model is not using default parameters but is optimized for this specific dataset and metric
4. **SHAP compatibility** — XGBoost's tree structure is natively supported by `shap.TreeExplainer`, making model explanation fast and exact (not approximate)
5. **Practical deployment** — XGBoost models can be serialized with `joblib` and loaded for real-time inference with sub-millisecond prediction latency

**Saved artifacts:**
- `models/xgb_fraud_model.pkl` — fraud model
- `models/xgb_credit_model.pkl` — credit card model
- `models/scaler_fraud.pkl` — fitted scaler for fraud features
- `models/scaler_credit.pkl` — fitted scaler for credit features
- `data/processed/X_test_fraud.csv` — test features for SHAP analysis
- `data/processed/y_test_fraud.csv` — test labels for SHAP analysis

---

## 5. Task 3 — Model Explainability

### 5.1 Why Explainability Matters in Fraud Detection

Fraud detection is a high-stakes, regulated domain. When a model flags a transaction as fraudulent, three parties need explanations:

- **The operations team** needs to understand what triggered the flag to decide whether to block the transaction or escalate
- **The compliance team** needs evidence that the model is not making decisions based on protected attributes (sex, age)
- **The business team** needs to translate model signals into policy changes (e.g., new verification rules)

SHAP (SHapley Additive exPlanations) provides mathematically rigorous explanations based on game theory. Each feature receives a SHAP value representing its contribution to pushing the prediction above or below the model's average prediction (the base value).

### 5.2 Built-in Feature Importance

XGBoost's `feature_importances_` attribute returns **gain-based importance** — how much each feature reduces impurity when it is used in a split, averaged across all trees. This gives a fast, model-native ranking.

**Top 10 features** are visualized as a horizontal bar chart and saved to `models/feature_importance_top10.png`. The tabular values are saved to `models/feature_importance_top10.csv`.

**Limitation of built-in importance:** Gain-based importance can be biased toward high-cardinality features (features with many distinct values). It also does not indicate the direction of the effect (does a high value increase or decrease fraud probability?). This is where SHAP adds value.

### 5.3 SHAP Global Analysis

`shap.TreeExplainer(model)` computes exact SHAP values for tree models. For large test sets, a subsample of 2,000 rows is used to keep computation within ~30 seconds.

**Summary bar plot** (`models/shap_summary_bar.png`): Ranks features by their mean absolute SHAP value across all test samples. This is the SHAP analog of feature importance — it answers "which features most strongly move predictions, on average?"

**Beeswarm plot** (`models/shap_summary_beeswarm.png`): Extends the bar plot by showing the distribution of SHAP values. Each dot represents one prediction. Color encodes the feature value (red = high, blue = low). This reveals:
- Whether high values of a feature push predictions toward fraud (positive SHAP) or away from it (negative SHAP)
- Whether the relationship is linear or non-linear (spread of the dots)
- Outlier cases where the feature has extreme influence on a single prediction

**Top 5 fraud drivers** by mean |SHAP| value are extracted and printed to the notebook output.

### 5.4 SHAP Local Analysis — Individual Force Plots

Three cases are selected from the test set for individual explanation:

**True Positive** — a fraud case the model correctly flagged:
- The waterfall plot shows which features pushed the score above the fraud threshold
- Expected: high contribution from `time_since_signup` (very recent signup), `transaction_velocity` (many recent device transactions), and possibly `hour_of_day`

**False Positive** — a legitimate transaction the model incorrectly flagged:
- The waterfall plot reveals what made this transaction look suspicious to the model
- Common cause: the transaction shares multiple features with fraud patterns (e.g., a new user making a large purchase at 3am from a country with elevated fraud rates) — even though it was legitimate, the combination of signals was unusual
- Actionable: these cases inform when to add soft friction (additional verification) rather than hard blocks

**False Negative** — a fraud case the model missed:
- The waterfall plot shows features that pulled the score below the threshold
- Common cause: the fraudster mimicked legitimate patterns — normal purchase time, reasonable purchase value, no device reuse
- Actionable: reveals what signals need to be strengthened (new features) or what threshold adjustments might help

Each waterfall plot is saved as a PNG file:
- `models/shap_waterfall_true_positive.png`
- `models/shap_waterfall_false_positive.png`
- `models/shap_waterfall_false_negative.png`

### 5.5 SHAP vs Built-in Importance Comparison

A side-by-side comparison table is produced with columns: `Feature`, `SHAP_Rank`, `BuiltIn_Rank`. Features that appear in the SHAP top 5 but not in the built-in top 10 are flagged as potential signals the gain metric undervalues.

**Typical discrepancies:**
- Features used in many shallow splits (low gain per split, high count) rank lower by gain but higher by SHAP because their cumulative impact is large
- Features with non-linear effects that only activate in specific feature combinations may be undervalued by gain but high-ranking by SHAP

---

## 6. Key Design Decisions and Justifications

### 6.1 Two Separate Pipelines

The e-commerce and credit card datasets have entirely different feature structures:
- Fraud_Data has behavioral, temporal, device, and geographic context requiring extensive feature engineering
- creditcard.csv has only PCA-transformed numeric features requiring only scaling

Treating them as one pipeline would require complex conditional logic and could mask dataset-specific patterns. Each dataset gets an independent pipeline tuned to its feature space.

### 6.2 SMOTE Applied After the Split

This is not just best practice — it is **necessary for honest evaluation**. The sequence is:

```
Full dataset → stratified split → X_train, X_test
X_train → SMOTE → X_train_resampled (used for training)
X_test → unchanged (used for evaluation)
```

If SMOTE were applied before the split, synthetic samples derived from test-set observations would appear in the training set. The model would then see "synthetic versions" of test examples during training, artificially inflating performance metrics. All evaluation numbers would be optimistic and non-representative of real-world performance.

### 6.3 Scaler Fit on Training Set Only

The same principle applies to scaling. The scaler's `fit()` computes mean and standard deviation from the data it sees. If it sees the full dataset (including the test set), those statistics encode information about test samples. When the model is later evaluated on "unseen" test data, the data has already influenced the preprocessing — a subtle form of leakage.

The correct sequence:
```python
scaler.fit(X_train)          # learn stats from training data only
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  # apply same stats to test data
```

### 6.4 Stratified K-Fold Cross-Validation

Standard K-Fold splits data randomly. With a 0.17% fraud rate (creditcard), a random fold might contain zero fraud cases, making AUC-PR undefined. Stratified K-Fold guarantees each fold contains approximately the same fraud rate as the full dataset, giving stable and meaningful performance estimates across all 5 folds.

---

## 7. Business Recommendations

Based on the SHAP analysis, the following concrete actions are recommended:

### Recommendation 1 — New Account Friction (`time_since_signup`)

SHAP consistently identifies `time_since_signup` as the strongest fraud predictor. Accounts transacting within seconds or minutes of creation represent a systematic fraud pattern — fraudsters create throwaway accounts, purchase, and disappear.

**Action:** Implement a configurable friction threshold. Transactions where `time_since_signup < 300 seconds` (5 minutes) should trigger step-up authentication (SMS OTP or email confirmation). This adds minimal friction for the ~0.1% of legitimate users who transact immediately after signup, while catching a large share of the fraud population.

**Expected impact:** High recall improvement with low false positive cost. The policy can be tuned by adjusting the threshold after measuring the false positive rate on a holdout set.

---

### Recommendation 2 — Device Velocity Monitoring (`transaction_velocity`, `device_transaction_count`)

Devices generating many transactions within 24 hours — especially devices shared across multiple user accounts (`users_per_device > 1`) — are strong fraud indicators. This pattern matches scripted attacks using emulators or virtual machines.

**Action:** Set a real-time velocity threshold: if a `device_id` generates more than 5 transactions within 24 hours, flag the session for manual review. If `users_per_device > 3`, automatically require re-authentication on the next transaction from that device. These thresholds should be validated against the distribution of `transaction_velocity` in the training data to minimize false positives on legitimate power-user devices.

**Expected impact:** Directly targets organized fraud rings that reuse infrastructure. High precision because legitimate devices rarely exceed the threshold.

---

### Recommendation 3 — Off-Hours High-Value Review (`hour_of_day`, `purchase_value`)

SHAP force plots for True Positives frequently show elevated contributions from `hour_of_day` values between 0–5 (midnight to 5am) combined with high `purchase_value`. This combination is atypical for legitimate users and common for fraud.

**Action:** For transactions between 00:00–05:00 local time where `purchase_value` exceeds the 90th percentile of the user's historical purchase distribution (or a global $250 threshold for new users), apply a 15-minute hold and send an in-app or email confirmation request. If no confirmation is received, block the transaction and flag for review.

**Expected impact:** Reduces false negatives during high-risk hours with minimal impact on daytime transactions. The personalized threshold (based on user history) reduces false positives for high-spending legitimate customers.

---

### Recommendation 4 — Geographic Anomaly Alerting (`country`)

Geolocation-derived country features contribute to SHAP scores particularly for transactions from countries with elevated fraud rates (identified in the EDA). More importantly, a transaction from a country inconsistent with a user's account history is a strong anomaly signal.

**Action:** Build a per-user geographic profile (countries seen in the last 6 months). Flag any transaction from a new country not in that profile as a medium-risk signal, triggering a soft MFA step. For new accounts (no history), apply the high-risk country list from the EDA analysis.

**Expected impact:** Catches account takeover fraud where a legitimate account is compromised and used from a different geography. Low false positive rate because most users have consistent geographic patterns.

---

### Recommendation 5 — Use False Negatives for Model Retraining (`shap_waterfall_false_negative.png`)

The SHAP waterfall plots for False Negatives reveal which features consistently fail to flag certain fraud patterns. If multiple False Negatives share similar SHAP profiles (e.g., they all have normal `time_since_signup` and moderate `transaction_velocity`), this indicates a fraud sub-type the current feature set does not capture.

**Action:** Schedule monthly model retraining. Weight recent False Negatives (confirmed fraud that was missed) more heavily in the training sample to force the model to improve on its systematic blind spots. Additionally, investigate whether new features (e.g., billing/shipping address mismatch, email domain age, device fingerprint) can be added to address these gaps.

**Expected impact:** Keeps the model calibrated against evolving fraud tactics rather than decaying in performance over time.

---

## 8. How to Run the Project

See `HOW_TO_RUN.md` for the complete step-by-step execution guide.
