import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    age = np.random.normal(30, 12, n).clip(1, 80)
    age[::10] = np.nan
    return pd.DataFrame({
        "PassengerId": range(1, n+1),
        "Survived":    np.random.choice([0,1], n, p=[0.62,0.38]),
        "Pclass":      np.random.choice([1,2,3], n, p=[0.24,0.21,0.55]),
        "Name":        [f"Mr_Test_{i}" for i in range(n)],
        "Sex":         np.random.choice(["male","female"], n, p=[0.65,0.35]),
        "Age":         age.round(1),
        "SibSp":       np.random.choice([0,1,2,3], n, p=[0.68,0.23,0.07,0.02]),
        "Parch":       np.random.choice([0,1,2], n, p=[0.76,0.13,0.11]),
        "Ticket":      [f"T{i}" for i in range(n)],
        "Fare":        np.random.lognormal(3.0, 0.9, n).clip(5, 512).round(2),
        "Cabin":       np.where(np.random.random(n) < 0.77, np.nan, "C23"),
        "Embarked":    np.random.choice(["S","C","Q"], n, p=[0.72,0.19,0.09]),
    })
