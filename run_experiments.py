from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier, RandomForestClassifier,
    HistGradientBoostingClassifier, GradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from grader_contracts.ensembles import EnsembleInput
from ensemble_models import run_ensemble_experiment

SEED = 42
ROOT = Path(__file__).resolve().parent

def metric_row(y_true, probabilities):
    pred = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }

def main():
    frame = pd.read_csv(ROOT / "german.csv", sep=";")
    X = frame.drop(columns=["Creditability"])
    y = frame["Creditability"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    experiments = {
        "Extra Trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=500, min_samples_leaf=2, max_features=0.8,
                class_weight="balanced", random_state=SEED, n_jobs=-1,
            ),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=500, min_samples_leaf=3, max_features="sqrt",
                class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
            ),
        ),
        "HistGradientBoosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.06, max_leaf_nodes=15,
                l2_regularization=2.0, random_state=SEED,
            ),
        ),
        "Gradient Boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.04, max_depth=2,
                min_samples_leaf=8, random_state=SEED,
            ),
        ),
        "MLP": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16), alpha=0.01,
                learning_rate_init=0.003, max_iter=500,
                early_stopping=True, random_state=SEED,
            ),
        ),
    }

    rows = {}
    for name, model in experiments.items():
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        rows[name] = {
            "cv_roc_auc_mean": float(cv_scores.mean()),
            "cv_roc_auc_std": float(cv_scores.std()),
            "test_metrics": metric_row(y_test, proba),
        }

    rf_search = {}
    for leaf in (1, 2, 3, 5):
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=400, min_samples_leaf=leaf, max_features="sqrt",
                class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
            ),
        )
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1
        )
        rf_search[str(leaf)] = {
            "cv_roc_auc_mean": float(scores.mean()),
            "cv_roc_auc_std": float(scores.std()),
        }

    final = run_ensemble_experiment(EnsembleInput(
        train_features=X_train, train_target=y_train,
        test_features=X_test, test_target=y_test, random_seed=SEED,
    ))
    results = {
        "dataset": {"rows": int(len(frame)), "features": int(X.shape[1])},
        "split": {"test_size": 0.2, "random_seed": SEED, "stratified": True},
        "cross_validation": {"folds": 5, "random_seed": SEED},
        "model_experiments": rows,
        "random_forest_min_samples_leaf_search": rf_search,
        "selected_model": {
            "name": final.model_name,
            "test_metrics": {
                "roc_auc": final.roc_auc, "accuracy": final.accuracy,
                "precision": final.precision, "recall": final.recall,
                "f1": final.f1,
            },
            "threshold": 0.5,
        },
    }
    (ROOT / "experiment_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
