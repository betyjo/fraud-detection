# Fraud Detection for E-commerce and Bank Transactions

## Overview

This project develops machine learning models for detecting fraudulent transactions in:

1. E-commerce transactions
2. Credit card transactions

The project includes:

- Data preprocessing
- Feature engineering
- Geolocation enrichment
- Class imbalance handling
- Machine learning modeling
- SHAP explainability

## Repository Structure

fraud-detection/

├── .vscode/

│ └── settings.json

├── .github/

│ └── workflows/

│ └── unittests.yml

├── data/ # Add to .gitignore

│ ├── raw/ # Original datasets

│ └── processed/ # Cleaned and feature-engineered data

├── notebooks/

│ ├── **init**.py

│ ├── eda-fraud-data.ipynb

│ ├── eda-creditcard.ipynb

│ ├── feature-engineering.ipynb

│ ├── modeling.ipynb

│ ├── shap-explainability.ipynb

│ └── README.md

├── src/

│ └── **init**.py

├── tests/

│ └── **init**.py

├── models/ # Saved model artifacts

├── scripts/

│ ├── **init**.py

│ └── README.md

├── requirements.txt

├── README.md

└── .gitignore
