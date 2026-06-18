"""Data cleaning for Titanic dataset."""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def drop_irrelevant(df: pd.DataFrame) -> pd.DataFrame:
    """Drop PassengerId, Ticket — non-predictive."""
    cols = ["PassengerId", "Ticket"]
    df = df.drop(columns=[c for c in cols if c in df.columns])
    return df


def fill_age(df: pd.DataFrame) -> pd.DataFrame:
    """Fill Age with median per Pclass+Sex group — more accurate than global median."""
    if "Age" in df.columns:
        df["Age"] = df.groupby(["Pclass","Sex"])["Age"].transform(
            lambda x: x.fillna(x.median())
        )
        df["Age"] = df["Age"].fillna(df["Age"].median())
    return df


def fill_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Fill Embarked with mode (S is most common)."""
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    return df


def fill_fare(df: pd.DataFrame) -> pd.DataFrame:
    """Fill Fare with median per Pclass."""
    if "Fare" in df.columns:
        df["Fare"] = df.groupby("Pclass")["Fare"].transform(
            lambda x: x.fillna(x.median())
        )
    return df


def fill_cabin(df: pd.DataFrame) -> pd.DataFrame:
    """Fill Cabin NaN with 'Unknown'."""
    if "Cabin" in df.columns:
        df["Cabin"] = df["Cabin"].fillna("Unknown")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    if before - len(df) > 0:
        logger.info(f"Removed {before - len(df)} duplicates")
    return df


def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    if "Pclass" in df.columns:
        df["Pclass"] = df["Pclass"].astype(int)
    if "SibSp" in df.columns:
        df["SibSp"] = df["SibSp"].astype(int)
    if "Parch" in df.columns:
        df["Parch"] = df["Parch"].astype(int)
    return df


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    m = df.isnull().sum()
    pct = m / len(df) * 100
    return pd.DataFrame({"Count": m, "Pct": pct.round(2)}).query("Count > 0").sort_values("Pct", ascending=False)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = drop_irrelevant(df)
    df = fill_age(df)
    df = fill_embarked(df)
    df = fill_fare(df)
    df = fill_cabin(df)
    df = remove_duplicates(df)
    df = fix_types(df)
    logger.info(f"Cleaned. Shape: {df.shape}. Nulls: {df.isnull().sum().sum()}")
    return df
