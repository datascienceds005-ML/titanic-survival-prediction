"""Load and generate Titanic dataset."""
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_titanic_data(n_samples: int = 891, seed: int = 42) -> pd.DataFrame:
    """Generate realistic Titanic-like dataset with correct probabilities."""
    np.random.seed(seed)
    n = n_samples

    pclass = np.random.choice([1,2,3], n, p=[0.24,0.21,0.55])
    sex    = np.random.choice(["male","female"], n, p=[0.65,0.35])

    age_raw = np.random.normal(29.7, 14.5, n).clip(0.17, 80)
    age = np.where(np.random.random(n) < 0.05, np.nan, age_raw)

    sibsp = np.random.choice([0,1,2,3,4,5,8], n,
                              p=[0.68,0.23,0.05,0.02,0.01,0.005,0.005])

    p_parch = np.array([0.76,0.13,0.09,0.01,0.006,0.002,0.002])
    p_parch = p_parch / p_parch.sum()
    parch   = np.random.choice([0,1,2,3,4,5,6], n, p=p_parch)

    fare = np.where(
        pclass==1, np.random.lognormal(4.2,0.8,n).clip(5,512),
        np.where(pclass==2, np.random.lognormal(2.9,0.5,n).clip(5,73),
                 np.random.lognormal(2.2,0.6,n).clip(5,69))
    )

    embarked = np.random.choice(["S","C","Q"], n, p=[0.722,0.189,0.089])

    cabin = np.where(
        np.random.random(n) < 0.77, np.nan,
        np.random.choice(["A","B","C","D","E","F","G"], n)
    )

    titles = np.where(
        sex=="male",
        np.random.choice(["Mr","Dr","Rev","Master"], n, p=[0.94,0.02,0.02,0.02]),
        np.random.choice(["Mrs","Miss","Ms","Dr"],   n, p=[0.47,0.48,0.03,0.02])
    )
    names   = [f"{titles[i]}, Passenger_{i}" for i in range(n)]
    tickets = [f"PC{np.random.randint(10000,99999)}" for _ in range(n)]

    age_safe = np.where(np.isnan(age), 29.7, age)
    surv_prob = (
        0.15 +
        0.40 * (sex == "female") +
        0.20 * (pclass == 1) +
        0.10 * (pclass == 2) -
        0.10 * (pclass == 3) +
        0.10 * (age_safe < 16) -
        0.05 * (age_safe > 50) +
        0.05 * (sibsp == 1) -
        0.05 * (sibsp > 3)
    ).clip(0.02, 0.98)

    survived = (np.random.random(n) < surv_prob).astype(int)

    df = pd.DataFrame({
        "PassengerId": range(1, n+1),
        "Survived":    survived,
        "Pclass":      pclass,
        "Name":        names,
        "Sex":         sex,
        "Age":         age.round(1),
        "SibSp":       sibsp,
        "Parch":       parch,
        "Ticket":      tickets,
        "Fare":        fare.round(4),
        "Cabin":       cabin,
        "Embarked":    embarked,
    })
    return df


def load_data(path: str = "data/raw/titanic.csv") -> pd.DataFrame:
    """Load or generate Titanic dataset."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        logger.info(f"Loaded: {df.shape}")
    else:
        logger.info("Generating Titanic dataset...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = generate_titanic_data()
        df.to_csv(path, index=False)
        logger.info(f"Saved: {path} {df.shape}")
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(f"\nSurvival rate: {df['Survived'].mean():.2%}")
