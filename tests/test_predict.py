import numpy as np

from forest_cover_prediction.predict import predict


class DummyModel:
    """Simple fake model used for testing."""

    def predict(self, X):
        return np.array([0, 1, 2, 3])


def test_predict_converts_zero_based_labels_to_one_based():
    """Check that predictions are converted from 0-6 to 1-7."""

    X_test = np.zeros((4, 2))

    predictions = predict(DummyModel(), X_test)

    expected = np.array([1, 2, 3, 4])

    np.testing.assert_array_equal(predictions, expected)


def test_predict_returns_numpy_array():
    """Check that predictions are returned as a NumPy array."""

    X_test = np.zeros((4, 2))

    predictions = predict(DummyModel(), X_test)

    assert isinstance(predictions, np.ndarray)
