import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, ".")
from src.data.clean import (
    fill_age, fill_embarked, remove_duplicates,
    drop_irrelevant, clean_data
)


def test_drop_irrelevant(sample_df):
    result = drop_irrelevant(sample_df)
    assert "PassengerId" not in result.columns
    assert "Ticket" not in result.columns


def test_fill_age(sample_df):
    df = drop_irrelevant(sample_df)
    result = fill_age(df)
    assert result["Age"].isnull().sum() == 0


def test_fill_embarked(sample_df):
    df = sample_df.copy()
    df.loc[0, "Embarked"] = np.nan
    result = fill_embarked(df)
    assert result["Embarked"].isnull().sum() == 0


def test_remove_duplicates():
    df = pd.DataFrame({"A":[1,1,2],"B":["x","x","y"],"Survived":[0,0,1]})
    result = remove_duplicates(df)
    assert len(result) == 2


def test_clean_data_no_nulls(sample_df):
    result = clean_data(sample_df)
    assert "PassengerId" not in result.columns
    assert result["Age"].isnull().sum() == 0
    assert result["Embarked"].isnull().sum() == 0
