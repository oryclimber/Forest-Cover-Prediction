def predict(best_model, X_test):
    """Generate predictions and convert them to the original label format."""

    y_test_pred = best_model.predict(X_test)

    # Convert model's 0-based class labels back to original 1-based labels
    y_test_pred += 1

    return y_test_pred
