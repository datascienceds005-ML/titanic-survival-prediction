"""
Advanced feature engineering for Titanic survival prediction.
Every feature is designed to improve predictive power.
"""
import pandas as pd
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)


def add_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """FamilySize = SibSp + Parch + 1. Large families had lower survival."""
    if "SibSp" in df.columns and "Parch" in df.columns:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df


def add_is_alone(df: pd.DataFrame) -> pd.DataFrame:
    """Traveling alone = higher risk. Women alone still survived more."""
    if "FamilySize" in df.columns:
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def add_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract title from Name. Title is a strong proxy for:
    - Gender (Mr/Mrs/Miss)
    - Age (Master = young boy)
    - Social status (Dr/Lady/Sir)
    """
    if "Name" in df.columns:
        df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
        if df["Title"].isnull().all():
            # Fallback for generated names
            df["Title"] = df["Name"].str.extract(r"^([^_]+)_", expand=False)
        rare = ["Lady","Countess","Capt","Col","Don","Dr","Major",
                "Rev","Sir","Jonkheer","Dona","Rev","Ms","Mlle","Mme"]
        df["Title"] = df["Title"].replace(rare, "Rare")
        df["Title"] = df["Title"].replace({"Mlle":"Miss","Ms":"Miss","Mme":"Mrs"})
        df["Title"] = df["Title"].fillna("Mr")
    return df


def add_fare_per_person(df: pd.DataFrame) -> pd.DataFrame:
    """Fare per family member — normalises group ticket prices."""
    if "Fare" in df.columns and "FamilySize" in df.columns:
        df["FarePerPerson"] = (df["Fare"] / df["FamilySize"]).round(4)
    return df


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Age groups — children had higher survival (women and children first)."""
    if "Age" in df.columns:
        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[0, 12, 18, 35, 60, 100],
            labels=["Child","Teen","Adult","MiddleAge","Senior"]
        ).astype(str)
    return df


def add_fare_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Fare quartile bins — proxy for wealth/class."""
    if "Fare" in df.columns:
        df["FareBin"] = pd.qcut(
            df["Fare"], q=4, labels=["Low","Mid","High","VeryHigh"],
            duplicates="drop"
        ).astype(str)
    return df


def add_cabin_known(df: pd.DataFrame) -> pd.DataFrame:
    """Was cabin information recorded? Known cabin = higher class = higher survival."""
    if "Cabin" in df.columns:
        df["CabinKnown"] = (df["Cabin"] != "Unknown").astype(int)
    return df


def add_deck(df: pd.DataFrame) -> pd.DataFrame:
    """Extract deck letter from Cabin (A-G). Higher decks = first class."""
    if "Cabin" in df.columns:
        df["Deck"] = df["Cabin"].str[0]
        df["Deck"] = df["Deck"].replace("U", "Unknown")
        df["Deck"] = df["Deck"].fillna("Unknown")
    return df


def add_family_type(df: pd.DataFrame) -> pd.DataFrame:
    """Family size category — solo/small/medium/large."""
    if "FamilySize" in df.columns:
        df["FamilyType"] = pd.cut(
            df["FamilySize"],
            bins=[0, 1, 3, 5, 20],
            labels=["Solo","Small","Medium","Large"]
        ).astype(str)
    return df


def add_pclass_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction: Pclass × Sex — 1st class female had ~97% survival."""
    if "Pclass" in df.columns and "Sex" in df.columns:
        df["PclassSex"] = df["Pclass"].astype(str) + "_" + df["Sex"]
    return df


def add_age_pclass(df: pd.DataFrame) -> pd.DataFrame:
    """Age × Pclass interaction — young 3rd class passengers had poor odds."""
    if "Age" in df.columns and "Pclass" in df.columns:
        df["AgePclass"] = (df["Age"] * df["Pclass"]).round(2)
    return df


def add_is_child(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: age under 12. Children prioritised in evacuation."""
    if "Age" in df.columns:
        df["IsChild"] = (df["Age"] < 12).astype(int)
    return df


def add_is_female(df: pd.DataFrame) -> pd.DataFrame:
    """Binary sex flag — sex is the #1 survival predictor."""
    if "Sex" in df.columns:
        df["IsFemale"] = (df["Sex"] == "female").astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run complete feature engineering pipeline."""
    df = df.copy()
    df = add_family_size(df)
    df = add_is_alone(df)
    df = add_title(df)
    df = add_fare_per_person(df)
    df = add_age_group(df)
    df = add_fare_bin(df)
    df = add_cabin_known(df)
    df = add_deck(df)
    df = add_family_type(df)
    df = add_pclass_sex(df)
    df = add_age_pclass(df)
    df = add_is_child(df)
    df = add_is_female(df)
    added = ["FamilySize","IsAlone","Title","FarePerPerson","AgeGroup",
             "FareBin","CabinKnown","Deck","FamilyType","PclassSex",
             "AgePclass","IsChild","IsFemale"]
    logger.info(f"Feature engineering done. Added: {[c for c in added if c in df.columns]}")
    return df
