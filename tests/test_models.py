from forest_cover_prediction.models import (
    get_best_model_name,
    get_classifiers,
)


def test_get_classifiers_returns_expected_models():
    """Check that all expected classifiers are created."""

    classifiers = get_classifiers()

    assert len(classifiers) == 4

    assert "Logistic Regression" in classifiers
    assert "KNN" in classifiers
    assert "Random Forest" in classifiers
    assert "XGBoost" in classifiers


def test_get_best_model_name():
    """Check that the model with the highest F1 score is selected."""

    results = {
        "Logistic Regression": {
            "f1_weighted": 0.80,
        },
        "Random Forest": {
            "f1_weighted": 0.92,
        },
        "XGBoost": {
            "f1_weighted": 0.89,
        },
    }

    best_model = get_best_model_name(results)

    assert best_model == "Random Forest"


def test_get_best_model_name_handles_single_model():
    """Check that model selection works with only one model."""

    results = {
        "Random Forest": {
            "f1_weighted": 0.92,
        }
    }

    assert get_best_model_name(results) == "Random Forest"


def test_random_forest_can_fit_and_predict():
    """Check that the Random Forest model can fit a small dataset."""

    from sklearn.datasets import make_classification

    classifiers = get_classifiers()
    model = classifiers["Random Forest"]

    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_classes=3,
        random_state=42,
    )

    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == 100
    assert set(predictions).issubset({0, 1, 2})
