import pytest
import sys
sys.path.insert(0, ".")
from src.data.clean import clean_data
from src.features.engineer import (
    add_family_size, add_is_alone, add_title,
    add_fare_per_person, add_is_female, engineer_features
)


def test_family_size(sample_df):
    df = clean_data(sample_df)
    result = add_family_size(df)
    assert "FamilySize" in result.columns
    assert (result["FamilySize"] >= 1).all()


def test_is_alone(sample_df):
    df = clean_data(sample_df)
    df = add_family_size(df)
    result = add_is_alone(df)
    assert "IsAlone" in result.columns
    assert result["IsAlone"].isin([0, 1]).all()


def test_is_female(sample_df):
    df = clean_data(sample_df)
    result = add_is_female(df)
    assert "IsFemale" in result.columns
    assert result["IsFemale"].isin([0, 1]).all()


def test_fare_per_person(sample_df):
    df = clean_data(sample_df)
    df = add_family_size(df)
    result = add_fare_per_person(df)
    assert "FarePerPerson" in result.columns
    assert (result["FarePerPerson"] > 0).all()


def test_engineer_all(sample_df):
    df = clean_data(sample_df)
    result = engineer_features(df)
    for col in ["FamilySize","IsAlone","IsFemale","CabinKnown","AgePclass"]:
        assert col in result.columns
