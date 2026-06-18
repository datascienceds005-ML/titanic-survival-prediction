import pytest
import numpy as np
import sys
sys.path.insert(0, ".")
from src.data.clean import clean_data
from src.features.engineer import engineer_features
from src.models.train import (
    get_feature_types, build_preprocessor,
    build_all_pipelines, compute_metrics
)


def test_feature_types(sample_df):
    df = engineer_features(clean_data(sample_df))
    num, cat = get_feature_types(df)
    assert "Age" in num
    assert "Sex" in cat
    assert "Survived" not in num


def test_preprocessor(sample_df):
    df = engineer_features(clean_data(sample_df))
    num, cat = get_feature_types(df)
    pre = build_preprocessor(num, cat)
    X = df.drop(columns=["Survived","Name","Cabin"], errors="ignore")
    X_t = pre.fit_transform(X)
    assert not np.isnan(X_t).any()


def test_logistic_pipeline(sample_df):
    df = engineer_features(clean_data(sample_df))
    num, cat = get_feature_types(df)
    pipes = build_all_pipelines(num, cat)
    pipe = pipes["LogisticRegression"]
    X = df.drop(columns=["Survived","Name","Cabin"], errors="ignore")
    y = df["Survived"]
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert set(preds).issubset({0, 1})


def test_compute_metrics():
    y_true = np.array([1,0,1,0,1,0,1,1,0,0])
    y_pred = np.array([1,0,1,0,0,0,1,1,0,1])
    y_prob = np.array([0.9,0.1,0.8,0.2,0.4,0.3,0.7,0.85,0.15,0.6])
    m = compute_metrics(y_true, y_pred, y_prob)
    assert 0 <= m["Accuracy"] <= 1
    assert 0 <= m["ROC_AUC"] <= 1
