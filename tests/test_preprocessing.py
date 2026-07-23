"""
test_preprocessing.py
---------------------
Unit tests for src/data_preprocessing.py.
All tests use synthetic data — no real CSV files required.
"""

import pytest
import pandas as pd
import numpy as np
from io import StringIO
from unittest.mock import patch, MagicMock

from src.data_preprocessing import (
    ip_to_int,
    load_and_clean_fraud,
    load_and_clean_credit,
    merge_ip_country,
)


# ---------------------------------------------------------------------------
# ip_to_int
# ---------------------------------------------------------------------------

class TestIpToInt:
    def test_ip_to_int_known_value(self):
        """A known float IP converts correctly to its integer representation."""
        result = ip_to_int(3627565.0)
        assert result == 3627565
        assert isinstance(result, int)

    def test_ip_to_int_zero(self):
        """Zero IP address converts to 0."""
        assert ip_to_int(0.0) == 0

    def test_ip_to_int_large_value(self):
        """Large float IP converts without overflow."""
        result = ip_to_int(4294967295.0)  # max IPv4 as int
        assert result == 4294967295

    def test_ip_to_int_from_int(self):
        """Integer input also works (idempotent)."""
        assert ip_to_int(12345) == 12345


# ---------------------------------------------------------------------------
# load_and_clean_fraud
# ---------------------------------------------------------------------------

class TestLoadAndCleanFraud:
    def _make_fraud_csv(self, include_duplicate=False) -> str:
        rows = [
            "user_id,signup_time,purchase_time,purchase_value,device_id,source,browser,sex,age,ip_address,class",
            "1,2015-01-01 00:00:00,2015-01-02 00:00:00,50.0,DEV1,SEO,Chrome,M,30,3627565.0,0",
            "2,2015-02-01 00:00:00,2015-02-02 00:00:00,75.0,DEV2,Ads,Firefox,F,25,1234567.0,1",
        ]
        if include_duplicate:
            # Exact duplicate of row 1
            rows.append("1,2015-01-01 00:00:00,2015-01-02 00:00:00,50.0,DEV1,SEO,Chrome,M,30,3627565.0,0")
        return "\n".join(rows)

    def test_load_and_clean_fraud_removes_duplicates(self, tmp_path):
        """Duplicate rows are removed after cleaning."""
        csv_content = self._make_fraud_csv(include_duplicate=True)
        csv_file = tmp_path / "Fraud_Data.csv"
        csv_file.write_text(csv_content)

        df = load_and_clean_fraud(str(csv_file))

        assert df.duplicated().sum() == 0
        assert len(df) == 2  # only 2 unique rows

    def test_load_and_clean_fraud_datetime_parsing(self, tmp_path):
        """signup_time and purchase_time are parsed as datetime."""
        csv_content = self._make_fraud_csv()
        csv_file = tmp_path / "Fraud_Data.csv"
        csv_file.write_text(csv_content)

        df = load_and_clean_fraud(str(csv_file))

        assert pd.api.types.is_datetime64_any_dtype(df["signup_time"])
        assert pd.api.types.is_datetime64_any_dtype(df["purchase_time"])

    def test_load_and_clean_fraud_no_duplicates_unchanged(self, tmp_path):
        """DataFrame with no duplicates is returned unchanged in length."""
        csv_content = self._make_fraud_csv(include_duplicate=False)
        csv_file = tmp_path / "Fraud_Data.csv"
        csv_file.write_text(csv_content)

        df = load_and_clean_fraud(str(csv_file))
        assert len(df) == 2


# ---------------------------------------------------------------------------
# load_and_clean_credit
# ---------------------------------------------------------------------------

class TestLoadAndCleanCredit:
    def _make_credit_csv(self, include_duplicate=False, include_non_numeric=False) -> str:
        header = "Time,V1,V2,Amount,Class"
        rows = [
            header,
            "0.0,1.1,-2.2,100.0,0",
            "1.0,0.5,3.3,50.5,1",
        ]
        if include_duplicate:
            rows.append("0.0,1.1,-2.2,100.0,0")
        if include_non_numeric:
            rows.append("bad,data,here,oops,x")
        return "\n".join(rows)

    def test_load_and_clean_credit_types(self, tmp_path):
        """All columns in a clean creditcard CSV are numeric."""
        csv_content = self._make_credit_csv()
        csv_file = tmp_path / "creditcard.csv"
        csv_file.write_text(csv_content)

        df = load_and_clean_credit(str(csv_file))

        for col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), (
                f"Column '{col}' should be numeric, got {df[col].dtype}"
            )

    def test_load_and_clean_credit_removes_duplicates(self, tmp_path):
        """Duplicate rows are removed."""
        csv_content = self._make_credit_csv(include_duplicate=True)
        csv_file = tmp_path / "creditcard.csv"
        csv_file.write_text(csv_content)

        df = load_and_clean_credit(str(csv_file))

        assert df.duplicated().sum() == 0
        assert len(df) == 2
