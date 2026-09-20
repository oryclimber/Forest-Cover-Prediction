import altair as alt
import pandas as pd
import streamlit as st

from forest_cover_prediction.data import (
    feature_engineering,  # whatever it's actually called
)
from forest_cover_prediction.models import CLASS_NAMES


def render_prediction_tab(best_model, X_train_columns):
    st.header("Predict Forest Cover Type")
    st.write("Adjust the feature values below to get a live prediction.")

    col1, col2 = st.columns(2)

    with col1:
        elevation = st.slider("Elevation (m)", 1800, 3900, 2800)
        aspect = st.slider("Aspect (degrees)", 0, 360, 180)
        slope = st.slider("Slope (degrees)", 0, 66, 15)
        horiz_hydrology = st.slider(
            "Horizontal Distance to Hydrology (m)", 0, 1400, 200
        )
        vert_hydrology = st.slider("Vertical Distance to Hydrology (m)", -180, 600, 50)

    with col2:
        horiz_roadways = st.slider("Horizontal Distance to Roadways (m)", 0, 7000, 1500)
        hillshade_9am = st.slider("Hillshade 9am", 0, 255, 220)
        hillshade_noon = st.slider("Hillshade Noon", 0, 255, 225)
        hillshade_3pm = st.slider("Hillshade 3pm", 0, 255, 140)
        horiz_fire = st.slider("Horizontal Distance to Fire Points (m)", 0, 7000, 1500)

    wilderness_areas = {
        "Rawah": "Wilderness_Area1",
        "Neota": "Wilderness_Area2",
        "Comanche Peak": "Wilderness_Area3",
        "Cache la Poudre": "Wilderness_Area4",
    }

    wilderness_area = st.selectbox("Wilderness Area", list(wilderness_areas.keys()))

    wilderness_area_raw_label = wilderness_areas[wilderness_area]

    soil_type = st.selectbox("Soil Type", [f"Soil_Type{i}" for i in range(1, 41)])

    raw_input = pd.DataFrame(
        [
            {
                "Elevation": elevation,
                "Aspect": aspect,
                "Slope": slope,
                "Horizontal_Distance_To_Hydrology": horiz_hydrology,
                "Vertical_Distance_To_Hydrology": vert_hydrology,
                "Horizontal_Distance_To_Roadways": horiz_roadways,
                "Hillshade_9am": hillshade_9am,
                "Hillshade_Noon": hillshade_noon,
                "Hillshade_3pm": hillshade_3pm,
                "Horizontal_Distance_To_Fire_Points": horiz_fire,
                "Wilderness_Area": wilderness_area_raw_label,
                "Soil_Type": soil_type,
            }
        ]
    )

    raw_input = pd.get_dummies(
        raw_input, columns=["Wilderness_Area", "Soil_Type"], prefix="", prefix_sep=""
    )
    raw_input["Id"] = 0

    input_row = feature_engineering(raw_input)
    input_row = input_row.reindex(columns=X_train_columns, fill_value=0)

    prediction = best_model.predict(input_row)[0]
    predicted_class = CLASS_NAMES[prediction]

    st.subheader(f"Predicted Cover Type: 🌲 {predicted_class}")

    # --- Probability bar chart ---
    if hasattr(best_model, "predict_proba"):
        probabilities = best_model.predict_proba(input_row)[0]
        proba_df = pd.DataFrame(
            {
                "Cover Type": CLASS_NAMES,
                "Probability": probabilities,
            }
        )

        n_classes = len(CLASS_NAMES)
        bar_width = 250

        proba_chart = (
            alt.Chart(proba_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Cover Type:N",
                    sort=alt.EncodingSortField(field="Probability", order="descending"),
                    axis=alt.Axis(labelAngle=-40, labelOverlap=False, labelLimit=0),
                ),
                y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.condition(
                    alt.datum["Cover Type"] == predicted_class,
                    alt.value("#2E7D32"),
                    alt.value("#B0C4DE"),
                ),
                tooltip=["Cover Type:N", alt.Tooltip("Probability:Q", format=".1%")],
            )
            .properties(
                title="Prediction Confidence by Cover Type",
                width=bar_width * n_classes,
                height=500,
            )
        )

        st.altair_chart(proba_chart, use_container_width=False)
    else:
        st.info(f"{type(best_model).__name__} doesn't support probability estimates.")
