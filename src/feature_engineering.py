"""
feature_engineering.py
-----------------------
Utility functions for creating features from raw fraud detection data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import List, Tuple


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features from purchase_time and signup_time.

    Features added:
    - hour_of_day: hour of the purchase (0–23)
    - day_of_week: day of week (0=Monday … 6=Sunday)
    - is_weekend: 1 if Saturday/Sunday, else 0
    - time_since_signup: seconds between signup and purchase, clipped to >= 0
    - time_since_signup_flag: 1 if the original time_since_signup was <= 0, else 0

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'purchase_time' and 'signup_time' as datetime columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with new time features appended.
    """
    df = df.copy()

    raw_seconds = (df["purchase_time"] - df["signup_time"]).dt.total_seconds()

    # Flag anomalous rows before clipping
    df["time_since_signup_flag"] = (raw_seconds <= 0).astype(int)

    # Clip negative values to 0
    df["time_since_signup"] = raw_seconds.clip(lower=0)

    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add transaction velocity features.

    Features added:
    - user_transaction_count: total transactions per user_id
    - device_transaction_count: total transactions per device_id
    - users_per_device: unique users per device_id
    - seconds_since_previous_transaction: seconds since same user's last transaction
      (NaN for first transaction filled with median)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'user_id', 'device_id', and 'purchase_time' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with velocity features appended.
    """
    df = df.copy()

    # User transaction count
    user_tx = df.groupby("user_id").size().reset_index(name="user_transaction_count")
    df = df.merge(user_tx, on="user_id", how="left")

    # Device transaction count
    device_tx = df.groupby("device_id").size().reset_index(name="device_transaction_count")
    df = df.merge(device_tx, on="device_id", how="left")

    # Unique users per device
    device_users = (
        df.groupby("device_id")["user_id"]
        .nunique()
        .reset_index(name="users_per_device")
    )
    df = df.merge(device_users, on="device_id", how="left")

    # Seconds since previous transaction (per user)
    df = df.sort_values(["user_id", "purchase_time"])
    df["_prev_purchase_time"] = df.groupby("user_id")["purchase_time"].shift()
    df["seconds_since_previous_transaction"] = (
        df["purchase_time"] - df["_prev_purchase_time"]
    ).dt.total_seconds()

    median_val = df["seconds_since_previous_transaction"].median()
    df["seconds_since_previous_transaction"] = (
        df["seconds_since_previous_transaction"].fillna(median_val)
    )

    df = df.drop(columns=["_prev_purchase_time"])

    return df


def encode_and_scale_fraud(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: List[str],
    num_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, object]:
    """One-hot encode categorical columns and StandardScale numeric columns.

    Encoding and scaling are fit on X_train only to prevent leakage.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature DataFrame.
    X_test : pd.DataFrame
        Test feature DataFrame.
    cat_cols : list of str
        Categorical column names to one-hot encode (drop_first=True).
    num_cols : list of str
        Numeric column names to scale.

    Returns
    -------
    tuple
        (X_train_transformed, X_test_transformed, fitted_scaler, fitted_encoder)
        Both transformed DataFrames are pandas DataFrames with named columns.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    # One-hot encode on train, align test
    X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

    # Align columns — test may be missing some dummy columns
    X_train_encoded, X_test_encoded = X_train_encoded.align(
        X_test_encoded, join="left", axis=1, fill_value=0
    )

    # Scale numeric columns — fit on train only
    scaler = StandardScaler()
    num_cols_present = [c for c in num_cols if c in X_train_encoded.columns]

    X_train_encoded[num_cols_present] = scaler.fit_transform(
        X_train_encoded[num_cols_present]
    )
    X_test_encoded[num_cols_present] = scaler.transform(
        X_test_encoded[num_cols_present]
    )

    return X_train_encoded, X_test_encoded, scaler, None
