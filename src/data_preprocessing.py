"""
data_preprocessing.py
---------------------
Utility functions for loading and cleaning raw fraud detection datasets.
"""

import pandas as pd
import numpy as np


def load_and_clean_fraud(path: str) -> pd.DataFrame:
    """Load Fraud_Data.csv, drop duplicates, parse datetime columns, and return cleaned DataFrame.

    Parameters
    ----------
    path : str
        Path to Fraud_Data.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with parsed signup_time and purchase_time columns.
    """
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    return df.reset_index(drop=True)


def load_and_clean_credit(path: str) -> pd.DataFrame:
    """Load creditcard.csv, drop duplicates, verify numeric columns, and return cleaned DataFrame.

    Parameters
    ----------
    path : str
        Path to creditcard.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with all numeric columns verified.

    Raises
    ------
    ValueError
        If any expected numeric column contains non-numeric data.
    """
    df = pd.read_csv(path)
    df = df.drop_duplicates()

    # Verify that all columns except 'Class' are numeric
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"Column '{col}' in creditcard data is not numeric. "
                f"Got dtype: {df[col].dtype}"
            )

    return df.reset_index(drop=True)


def ip_to_int(ip_float) -> int:
    """Convert a float IP address to an integer.

    The raw Fraud_Data IP addresses are stored as floats (e.g., 3627565.0).
    This function safely casts them to int for range lookups.

    Parameters
    ----------
    ip_float : float or int
        Float representation of an IP address.

    Returns
    -------
    int
        Integer representation of the IP address.

    Examples
    --------
    >>> ip_to_int(3627565.0)
    3627565
    """
    return int(ip_float)


def merge_ip_country(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """Merge fraud transactions with IP geolocation data using merge_asof.

    Performs a backward sorted merge to find the country corresponding to
    each IP address. Unmatched IPs (outside any known range) receive 'Unknown'.

    Parameters
    ----------
    fraud_df : pd.DataFrame
        Fraud data with an 'ip_address' column (integer type).
    ip_df : pd.DataFrame
        IP-to-country mapping with 'lower_bound_ip_address',
        'upper_bound_ip_address', and 'country' columns (integer type).

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with a 'country' column added.
        Rows with IPs outside any range have country='Unknown'.
    """
    # Ensure integer types
    fraud_df = fraud_df.copy()
    ip_df = ip_df.copy()

    fraud_df["ip_address"] = fraud_df["ip_address"].astype(np.int64)
    ip_df["lower_bound_ip_address"] = ip_df["lower_bound_ip_address"].astype(np.int64)
    ip_df["upper_bound_ip_address"] = ip_df["upper_bound_ip_address"].astype(np.int64)

    fraud_df = fraud_df.sort_values("ip_address")
    ip_df = ip_df.sort_values("lower_bound_ip_address")

    merged = pd.merge_asof(
        fraud_df,
        ip_df,
        left_on="ip_address",
        right_on="lower_bound_ip_address",
        direction="backward",
    )

    # Invalidate rows where IP exceeds the upper bound of matched range
    mask_invalid = merged["ip_address"] > merged["upper_bound_ip_address"]
    merged.loc[mask_invalid, "country"] = "Unknown"
    merged["country"] = merged["country"].fillna("Unknown")

    return merged
