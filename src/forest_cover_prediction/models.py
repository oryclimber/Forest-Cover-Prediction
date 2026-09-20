import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
N_CLASSES = 7

CLASS_NAMES = [
    "Spruce/Fir",
    "Lodgepole Pine",
    "Ponderosa Pine",
    "Cottonwood/Willow",
    "Aspen",
    "Douglas-fir",
    "Krummholz",
]


def get_classifiers() -> dict:
    """Returns the models used for the initial benchmark"""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=50)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
        ),
    }


def benchmark_models(X_train: pd.DataFrame, y: np.array) -> tuple[dict, dict]:
    """Benchmark the candidate models using stratified cross-validation.

    Returns
    -------
    classifiers : dict
        Dictionary containing the unfitted classifiers.

    results : dict
        Mean cross-validation scores for each classifier.
    """

    classifiers = get_classifiers()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = [
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]

    results = {}

    for name, clf in classifiers.items():
        cv_scores = cross_validate(
            clf,
            X_train,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        results[name] = {
            metric: cv_scores[f"test_{metric}"].mean() for metric in scoring
        }

    return classifiers, results


def get_best_model_name(results: dict) -> str:
    """Return the model with the highest weighted F1 score."""

    return max(
        results,
        key=lambda name: results[name]["f1_weighted"],
    )


def train_best_model(
    classifiers: dict, results: dict, X_train: pd.DataFrame, y: np.array
):
    """Train the best benchmarked model on the full training set."""

    best_model_name = get_best_model_name(results)
    best_model = classifiers[best_model_name]

    best_model.fit(X_train, y)

    return best_model_name, best_model


def tune_hyperparameters(X_train: pd.DataFrame, y: np.array, n_trials=30) -> float:
    """Tune an XGBoost classifier using Optuna."""

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y,
        test_size=0.15,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "tree_method": "hist",
            "objective": "multi:softmax",
            "num_class": N_CLASSES,
            "eval_metric": "mlogloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }

        model = XGBClassifier(
            **params,
            early_stopping_rounds=50,
            verbosity=0,
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        predictions = model.predict(X_val)

        return accuracy_score(y_val, predictions)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    return study
