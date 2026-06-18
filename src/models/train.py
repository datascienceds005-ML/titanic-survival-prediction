"""
Titanic model training.
Baseline + Ensemble models with sklearn pipelines.
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

logger = logging.getLogger(__name__)


def get_feature_types(df: pd.DataFrame, target: str = "Survived"):
    X = df.drop(columns=[target, "Name", "Cabin"], errors="ignore")
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def build_all_pipelines(num_cols, cat_cols) -> Dict[str, Pipeline]:
    def pre():
        return build_preprocessor(num_cols, cat_cols)

    pipelines = {
        # Baseline
        "LogisticRegression": Pipeline([
            ("pre", pre()),
            ("model", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ]),
        "DecisionTree": Pipeline([
            ("pre", pre()),
            ("model", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ]),
        "KNN": Pipeline([
            ("pre", pre()),
            ("model", KNeighborsClassifier(n_neighbors=7)),
        ]),
        # Ensemble
        "RandomForest": Pipeline([
            ("pre", pre()),
            ("model", RandomForestClassifier(
                n_estimators=300, max_depth=8,
                min_samples_split=4, random_state=42, n_jobs=-1
            )),
        ]),
        "ExtraTrees": Pipeline([
            ("pre", pre()),
            ("model", ExtraTreesClassifier(
                n_estimators=300, max_depth=8, random_state=42, n_jobs=-1
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("pre", pre()),
            ("model", GradientBoostingClassifier(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, random_state=42
            )),
        ]),
        "AdaBoost": Pipeline([
            ("pre", pre()),
            ("model", AdaBoostClassifier(
                n_estimators=200, learning_rate=0.05,
                random_state=42, algorithm="SAMME"
            )),
        ]),
    }

    if HAS_XGB:
        pipelines["XGBoost"] = Pipeline([
            ("pre", pre()),
            ("model", XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                random_state=42, eval_metric="logloss", verbosity=0,
            )),
        ])

    return pipelines


def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC_AUC":   round(roc_auc_score(y_true, y_prob), 4),
    }


def train_and_evaluate(df: pd.DataFrame) -> Dict[str, Any]:
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/metrics", exist_ok=True)

    target = "Survived"
    X = df.drop(columns=[target, "Name", "Cabin"], errors="ignore")
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols, cat_cols = get_feature_types(
        pd.concat([X_train, y_train.rename(target)], axis=1)
    )
    pipelines = build_all_pipelines(num_cols, cat_cols)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    best_acc = 0
    best_name = None

    for name, pipeline in pipelines.items():
        logger.info(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        metrics["CV_Acc_Mean"] = round(cv_scores.mean(), 4)
        metrics["CV_Acc_Std"]  = round(cv_scores.std(), 4)
        results[name] = metrics
        joblib.dump(pipeline, f"models/{name.lower()}.joblib")
        logger.info(f"{name}: Acc={metrics['Accuracy']:.4f}, AUC={metrics['ROC_AUC']:.4f}")
        if metrics["Accuracy"] > best_acc:
            best_acc = metrics["Accuracy"]
            best_name = name

    if best_name:
        import shutil
        shutil.copy(f"models/{best_name.lower()}.joblib", "models/best_model.joblib")
        results["best_model"] = best_name

    with open("reports/metrics/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("  TITANIC MODEL RESULTS")
    print("="*60)
    for name, m in results.items():
        if name == "best_model":
            continue
        print(f"\n{name}:")
        for k, v in m.items():
            print(f"  {k:<15}: {v}")
    print(f"\n🏆 Best: {results.get('best_model','N/A')} (Acc={best_acc:.4f})")
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.load import load_data
    from src.data.clean import clean_data
    from src.features.engineer import engineer_features

    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    train_and_evaluate(df)
