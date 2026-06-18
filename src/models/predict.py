"""Load saved model and predict Titanic survival."""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


def load_model(name: str = "best"):
    paths = {
        "best": "models/best_model.joblib",
        "rf":   "models/randomforest.joblib",
        "gb":   "models/gradientboosting.joblib",
        "lr":   "models/logisticregression.joblib",
        "xgb":  "models/xgboost.joblib",
    }
    path = paths.get(name, "models/best_model.joblib")
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found: {path}. Run train.py first.")
    return joblib.load(path)


def predict_survival(features: dict, model_name: str = "best") -> dict:
    model = load_model(model_name)
    df = pd.DataFrame([features])
    prob = float(model.predict_proba(df)[0][1])
    pred = int(model.predict(df)[0])

    if prob >= 0.70:
        verdict = "🟢 LIKELY SURVIVED"
        advice  = "High survival probability. Strong positive indicators present."
    elif prob >= 0.40:
        verdict = "🟡 UNCERTAIN"
        advice  = "Moderate survival probability. Mixed indicators."
    else:
        verdict = "🔴 LIKELY DID NOT SURVIVE"
        advice  = "Low survival probability. Unfavourable conditions."

    return {
        "survived": pred,
        "probability": round(prob, 4),
        "probability_pct": f"{prob*100:.1f}%",
        "verdict": verdict,
        "advice": advice,
    }


if __name__ == "__main__":
    sample = {
        "Pclass": 1, "Sex": "female", "Age": 28.0,
        "SibSp": 0, "Parch": 0, "Fare": 100.0,
        "Embarked": "S", "FamilySize": 1, "IsAlone": 1,
        "Title": "Miss", "FarePerPerson": 100.0,
        "AgeGroup": "Adult", "FareBin": "VeryHigh",
        "CabinKnown": 1, "Deck": "C", "FamilyType": "Solo",
        "PclassSex": "1_female", "AgePclass": 28.0,
        "IsChild": 0, "IsFemale": 1,
    }
    result = predict_survival(sample)
    print(f"Verdict: {result['verdict']}")
    print(f"Probability: {result['probability_pct']}")
