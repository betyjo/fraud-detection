# Scripts

This folder is intended for standalone Python scripts that operationalize the notebook pipeline.

---

## Planned Scripts

### `train_fraud.py`
End-to-end training script for the Fraud_Data model:
- Loads and preprocesses `data/raw/Fraud_Data.csv` and `data/raw/IpAddress_to_Country.csv`
- Runs feature engineering (via `src/feature_engineering.py`)
- Stratified split, SMOTE, XGBoost training
- Saves model to `models/xgb_fraud_model.pkl` and test set CSVs

### `train_credit.py`
Same pipeline for `data/raw/creditcard.csv` → `models/xgb_credit_model.pkl`.

### `predict.py`
Batch inference script:
- Accepts a CSV path as input
- Loads the saved model and scaler
- Outputs predictions and fraud probabilities to a new CSV

### `evaluate.py`
Standalone evaluation script:
- Loads a saved model and a labeled CSV
- Prints F1, AUC-PR, confusion matrix
- Uses `src/model_utils.evaluate_model()`

---

## Usage Pattern

```bash
python scripts/train_fraud.py --data-dir data/raw --output-dir models/
python scripts/predict.py --model models/xgb_fraud_model.pkl --input data/new_transactions.csv --output data/predictions.csv
```

Scripts accept command-line arguments via `argparse` and log progress to stdout.
