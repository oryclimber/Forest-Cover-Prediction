import altair as alt
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from forest_cover_prediction.models import (
    CLASS_NAMES,
    RANDOM_STATE,
    benchmark_models,
    get_best_model_name,
    train_best_model,
)


@st.cache_data
def get_benchmark_results(X_train, y):
    return benchmark_models(X_train, y)


@st.cache_resource
def get_trained_best_model(_classifiers, results, X_train, y):
    return train_best_model(_classifiers, results, X_train, y)


def render_model_tab(X_train, y):
    st.header("Model Benchmarking")

    with st.spinner("Running cross-validation across candidate models..."):
        classifiers, results = get_benchmark_results(X_train, y)

    # --- Benchmark comparison chart ---
    results_df = (
        pd.DataFrame(results)
        .T.reset_index()
        .rename(columns={"index": "Model"})
        .melt(id_vars="Model", var_name="Metric", value_name="Score")
    )

    metric_labels = {
        "accuracy": "Accuracy",
        "precision_weighted": "Precision",
        "recall_weighted": "Recall",
        "f1_weighted": "F1 Score",
    }
    results_df["Metric"] = results_df["Metric"].map(metric_labels)

    n_models = results_df["Model"].nunique()
    bar_width = 300

    benchmark_chart = (
        alt.Chart(results_df)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", axis=alt.Axis(labelAngle=-40, labelOverlap=False)),
            xOffset="Metric:N",
            y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Metric:N", title="Metric"),
            tooltip=["Model:N", "Metric:N", alt.Tooltip("Score:Q", format=".3f")],
        )
        .properties(
            title="Cross-Validated Model Performance",
            width=bar_width * n_models,
            height=500,
        )
    )

    st.altair_chart(benchmark_chart, use_container_width=False)

    # --- Best model callout ---
    best_model_name = get_best_model_name(results)
    best_f1 = results[best_model_name]["f1_weighted"]

    st.subheader("Best Model")
    col1, col2 = st.columns(2)
    col1.metric("Selected Model", best_model_name)
    col2.metric("F1 Score (weighted)", f"{best_f1:.3f}")

    # --- Train best model and show confusion matrix ---
    st.subheader("Confusion Matrix")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    _, best_model = get_trained_best_model(classifiers, results, X_tr, y_tr)
    y_pred = best_model.predict(X_val)

    cm = confusion_matrix(y_val, y_pred)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)

    cm_long = (
        cm_df.reset_index()
        .melt(id_vars="index", var_name="Predicted", value_name="Count")
        .rename(columns={"index": "Actual"})
    )

    n_classes = len(CLASS_NAMES)
    cell_size = 180

    cm_heatmap = (
        alt.Chart(cm_long)
        .mark_rect()
        .encode(
            x=alt.X(
                "Predicted:N",
                sort=CLASS_NAMES,
                axis=alt.Axis(labelAngle=-40, labelOverlap=False, labelLimit=0),
            ),
            y=alt.Y(
                "Actual:N",
                sort=CLASS_NAMES,
                axis=alt.Axis(labelOverlap=False, labelLimit=0),
            ),
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), title="Count"),
            tooltip=["Actual:N", "Predicted:N", "Count:Q"],
        )
        .properties(
            title=f"Confusion Matrix — {best_model_name}",
            width=cell_size * n_classes,
            height=cell_size * n_classes,
        )
    )

    cm_text = cm_heatmap.mark_text(baseline="middle", fontSize=13).encode(
        text="Count:Q",
        color=alt.condition(
            alt.datum.Count > cm.max() / 2, alt.value("white"), alt.value("black")
        ),
    )

    st.altair_chart(cm_heatmap + cm_text, use_container_width=False)

    return best_model
