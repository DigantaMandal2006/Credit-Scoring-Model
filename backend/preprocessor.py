"""
Shared preprocessing utilities for the Credit Scoring API.
Used by both the Flask backend and the training pipeline.
"""

import numpy as np
import pandas as pd

DROP_COLS = ["ID", "Customer_ID", "Name", "SSN", "Month"]

NUM_COLS = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio",
    "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
]

CAT_COLS = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]

FEATURE_ORDER = [
    "Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Credit_Mix", "Outstanding_Debt", "Credit_Utilization_Ratio",
    "Payment_of_Min_Amount", "Total_EMI_per_month",
    "Amount_invested_monthly", "Payment_Behaviour", "Monthly_Balance",
    "Credit_History_Months", "Num_Loan_Types",
    "Debt_to_Income", "EMI_to_Salary", "Savings_Rate",
    "Balance_Buffer", "Cards_per_Account",
]


def clean_numeric(series: pd.Series) -> pd.Series:
    """Strip non-numeric characters and coerce to float."""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce",
    )


def parse_credit_history_age(series: pd.Series) -> pd.Series:
    """Convert 'X Years and Y Months' text → total months (float)."""
    years  = series.astype(str).str.extract(r"(\d+)\s*Year",  expand=False)
    months = series.astype(str).str.extract(r"(\d+)\s*Month", expand=False)
    y = pd.to_numeric(years,  errors="coerce").fillna(0)
    m = pd.to_numeric(months, errors="coerce").fillna(0)
    result = y * 12 + m
    result[result == 0] = np.nan
    return result


def preprocess_dataframe(df: pd.DataFrame, scaler, cat_encoders: dict) -> pd.DataFrame:
    """
    Apply the full preprocessing pipeline to a raw dataframe.
    Expects the scaler and cat_encoders fitted on training data.
    Returns a dataframe with exactly the columns in FEATURE_ORDER, scaled.
    """
    df = df.copy()

    # Drop identifiers
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Clean numeric columns
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    # Credit history age → months
    if "Credit_History_Age" in df.columns:
        df["Credit_History_Months"] = parse_credit_history_age(df["Credit_History_Age"])
        df.drop(columns=["Credit_History_Age"], inplace=True)
    elif "Credit_History_Months" not in df.columns:
        df["Credit_History_Months"] = np.nan

    # Type_of_Loan → count
    if "Type_of_Loan" in df.columns:
        df["Num_Loan_Types"] = df["Type_of_Loan"].fillna("").astype(str).apply(
            lambda x: len([t for t in x.split(",") if t.strip() not in ("", "nan", "none")])
        )
        df.drop(columns=["Type_of_Loan"], inplace=True)
    elif "Num_Loan_Types" not in df.columns:
        df["Num_Loan_Types"] = 0

    # Engineered features
    eps = 1e-6
    df["Debt_to_Income"]  = df.get("Outstanding_Debt", 0)    / (df.get("Annual_Income", eps) + eps)
    df["EMI_to_Salary"]   = df.get("Total_EMI_per_month", 0) / (df.get("Monthly_Inhand_Salary", eps) + eps)
    df["Savings_Rate"]    = df.get("Amount_invested_monthly", 0) / (df.get("Monthly_Inhand_Salary", eps) + eps)
    df["Balance_Buffer"]  = df.get("Monthly_Balance", 0)     / (df.get("Monthly_Inhand_Salary", eps) + eps)
    df["Cards_per_Account"] = df.get("Num_Credit_Card", 0)   / (df.get("Num_Bank_Accounts", eps) + eps)

    # Encode categoricals
    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = "Unknown"
        else:
            df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "_": np.nan}).fillna("Unknown")
        if col in cat_encoders:
            le = cat_encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])

    # Drop any remaining object columns
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    df.drop(columns=obj_cols, inplace=True)

    # Ensure all expected features exist
    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0

    df = df[FEATURE_ORDER]

    # Replace inf/NaN → 0 before scaling
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # Scale
    df[FEATURE_ORDER] = scaler.transform(df[FEATURE_ORDER])

    # Final safety
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    return df


def preprocess_single(record: dict, scaler, cat_encoders: dict) -> pd.DataFrame:
    """Wrap a single dict record into a 1-row preprocessed DataFrame."""
    return preprocess_dataframe(pd.DataFrame([record]), scaler, cat_encoders)
