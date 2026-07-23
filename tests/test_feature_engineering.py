"""
test_feature_engineering.py
----------------------------
Unit tests for src/feature_engineering.py.
All tests use synthetic in-memory DataFrames — no real CSV files required.
"""

import pytest
import pandas as pd
import numpy as np

from src.feature_engineering import (
    add_time_features,
    add_velocity_features,
    encode_and_scale_fraud,
)


def _make_base_df(n=5):
    """Create a minimal synthetic DataFrame for time feature tests."""
    base_time = pd.Timestamp("2020-01-15 10:30:00")
    return pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5],
        "device_id": ["D1", "D2", "D1", "D3", "D2"],
        "signup_time": [
            pd.Timestamp("2020-01-01 08:00:00"),
            pd.Timestamp("2020-01-10 09:00:00"),
            pd.Timestamp("2020-01-15 10:30:01"),  # purchase before signup (negative)
            pd.Timestamp("2020-01-15 10:30:00"),  # exactly same (zero)
            pd.Timestamp("2019-12-01 00:00:00"),
        ],
        "purchase_time": [
            base_time,
            base_time,
            base_time,
            base_time,
            base_time,
        ],
        "purchase_value": [10.0, 20.0, 30.0, 40.0, 50.0],
        "class": [0, 0, 1, 0, 1],
    })


# ---------------------------------------------------------------------------
# add_time_features
# ---------------------------------------------------------------------------

class TestAddTimeFeatures:
    def test_add_time_features_columns_exist(self):
        """All expected time feature columns are added."""
        df = _make_base_df()
        result = add_time_features(df)

        expected_cols = [
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "time_since_signup",
            "time_since_signup_flag",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_time_since_signup_negative_clipped(self):
        """Row where purchase_time < signup_time gives time_since_signup=0 and flag=1."""
        df = _make_base_df()
        result = add_time_features(df)

        # Row index 2 has signup_time after purchase_time (1 second after)
        negative_row = result.iloc[2]
        assert negative_row["time_since_signup"] == 0.0, (
            f"Expected 0, got {negative_row['time_since_signup']}"
        )
        assert negative_row["time_since_signup_flag"] == 1, (
            f"Expected flag=1 for negative time, got {negative_row['time_since_signup_flag']}"
        )

    def test_time_since_signup_zero_flagged(self):
        """Row where purchase_time == signup_time gives time_since_signup=0 and flag=1."""
        df = _make_base_df()
        result = add_time_features(df)

        # Row index 3 has exactly the same timestamp
        zero_row = result.iloc[3]
        assert zero_row["time_since_signup"] == 0.0
        assert zero_row["time_since_signup_flag"] == 1

    def test_time_since_signup_positive_not_flagged(self):
        """Row where purchase_time > signup_time has flag=0 and positive time."""
        df = _make_base_df()
        result = add_time_features(df)

        # Row index 0 has purchase 14 days after signup
        positive_row = result.iloc[0]
        assert positive_row["time_since_signup"] > 0
        assert positive_row["time_since_signup_flag"] == 0

    def test_is_weekend_values(self):
        """is_weekend is 1 only for Saturday/Sunday transactions."""
        df = _make_base_df()
        result = add_time_features(df)

        # 2020-01-15 is a Wednesday (day_of_week=2), so is_weekend should be 0
        assert result["is_weekend"].iloc[0] == 0

    def test_hour_of_day_range(self):
        """hour_of_day values are within [0, 23]."""
        df = _make_base_df()
        result = add_time_features(df)
        assert result["hour_of_day"].between(0, 23).all()

    def test_original_df_not_mutated(self):
        """Original DataFrame is not modified (function uses copy)."""
        df = _make_base_df()
        _ = add_time_features(df)
        assert "time_since_signup" not in df.columns


# ---------------------------------------------------------------------------
# add_velocity_features
# ---------------------------------------------------------------------------

class TestAddVelocityFeatures:
    def _make_velocity_df(self):
        """Three rows from the same user, two devices."""
        return pd.DataFrame({
            "user_id": [10, 10, 10],
            "device_id": ["DA", "DA", "DB"],
            "purchase_time": pd.to_datetime([
                "2020-03-01 10:00:00",
                "2020-03-01 11:00:00",
                "2020-03-01 12:00:00",
            ]),
            "class": [0, 0, 1],
        })

    def test_add_velocity_features_count(self):
        """Three rows from the same user gives user_transaction_count=3."""
        df = self._make_velocity_df()
        result = add_velocity_features(df)

        assert (result["user_transaction_count"] == 3).all(), (
            f"Expected all 3, got {result['user_transaction_count'].tolist()}"
        )

    def test_device_transaction_count(self):
        """Device DA appears twice, device DB once."""
        df = self._make_velocity_df()
        result = add_velocity_features(df)

        da_rows = result[result["device_id"] == "DA"]
        db_rows = result[result["device_id"] == "DB"]

        assert (da_rows["device_transaction_count"] == 2).all()
        assert (db_rows["device_transaction_count"] == 1).all()

    def test_users_per_device(self):
        """Single user using DA → users_per_device = 1 for DA."""
        df = self._make_velocity_df()
        result = add_velocity_features(df)

        da_rows = result[result["device_id"] == "DA"]
        assert (da_rows["users_per_device"] == 1).all()

    def test_seconds_since_previous_transaction_first_is_filled(self):
        """First transaction per user has NaN filled with median (not NaN)."""
        df = self._make_velocity_df()
        result = add_velocity_features(df)

        assert result["seconds_since_previous_transaction"].isna().sum() == 0

    def test_velocity_features_columns_exist(self):
        """All expected velocity columns are present in output."""
        df = self._make_velocity_df()
        result = add_velocity_features(df)

        expected = [
            "user_transaction_count",
            "device_transaction_count",
            "users_per_device",
            "seconds_since_previous_transaction",
        ]
        for col in expected:
            assert col in result.columns, f"Missing: {col}"


# ---------------------------------------------------------------------------
# encode_and_scale_fraud
# ---------------------------------------------------------------------------

class TestEncodeAndScaleFraud:
    def _make_encode_dfs(self):
        train = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "purchase_value": [100.0, 200.0, 150.0, 300.0],
            "source": ["SEO", "Ads", "SEO", "Direct"],
            "browser": ["Chrome", "Firefox", "Chrome", "Safari"],
        })
        test = pd.DataFrame({
            "age": [28, 33],
            "purchase_value": [120.0, 250.0],
            "source": ["Ads", "SEO"],
            "browser": ["Firefox", "Chrome"],
        })
        return train, test

    def test_encode_and_scale_output_shapes(self):
        """Output DataFrames have the same number of rows as inputs."""
        train, test = self._make_encode_dfs()
        X_train_t, X_test_t, scaler, _ = encode_and_scale_fraud(
            train, test,
            cat_cols=["source", "browser"],
            num_cols=["age", "purchase_value"],
        )
        assert len(X_train_t) == len(train)
        assert len(X_test_t) == len(test)

    def test_encode_and_scale_no_cat_cols_remaining(self):
        """Original categorical columns are removed from output."""
        train, test = self._make_encode_dfs()
        X_train_t, X_test_t, scaler, _ = encode_and_scale_fraud(
            train, test,
            cat_cols=["source", "browser"],
            num_cols=["age", "purchase_value"],
        )
        assert "source" not in X_train_t.columns
        assert "browser" not in X_train_t.columns

    def test_encode_and_scale_scaler_fitted_on_train(self):
        """Scaler mean is computed from train data only."""
        train, test = self._make_encode_dfs()
        X_train_t, X_test_t, scaler, _ = encode_and_scale_fraud(
            train, test,
            cat_cols=["source", "browser"],
            num_cols=["age", "purchase_value"],
        )
        # Scaled train numeric columns should have mean ≈ 0
        train_numeric_means = X_train_t[["age", "purchase_value"]].mean()
        assert abs(train_numeric_means["age"]) < 1e-9
        assert abs(train_numeric_means["purchase_value"]) < 1e-9
