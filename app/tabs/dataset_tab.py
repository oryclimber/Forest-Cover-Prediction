import altair as alt
import plotly.express as px
import streamlit as st


def render_dataset_tab(raw_train, X_train):
    st.header("Raw Data Exploration")

    cover_types = {
        1: "Spruce/Fir",
        2: "Lodgepole Pine",
        3: "Ponderosa Pine",
        4: "Cottonwood/Willow",
        5: "Aspen",
        6: "Douglas-fir",
        7: "Krummholz",
    }

    wilderness_areas = {
        "Wilderness_Area1": "Rawah Wilderness Area",
        "Wilderness_Area2": "Neota Wilderness Area",
        "Wilderness_Area3": "Comanche Peak Wilderness Area",
        "Wilderness_Area4": "Cache la Poudre Wilderness Area",
    }

    cover_counts = (
        raw_train["Cover_Type"].value_counts().rename(index=cover_types).reset_index()
    )

    cover_counts.columns = ["Cover Type", "Count"]

    wilderness_columns = [
        col for col in raw_train.columns if col.startswith("Wilderness_Area")
    ]

    # Create long-format dataframe
    wilderness_cover_counts = raw_train.melt(
        id_vars="Cover_Type",
        value_vars=wilderness_columns,
        var_name="Wilderness_Area",
        value_name="Presence",
    )

    # Keep only observations where the wilderness area is present
    wilderness_cover_counts = wilderness_cover_counts[
        wilderness_cover_counts["Presence"] == 1
    ]

    # Create contingency table
    contingency_table = (
        wilderness_cover_counts.groupby(["Wilderness_Area", "Cover_Type"])
        .size()
        .unstack(fill_value=0)
    )

    # Convert back to long format for Altair
    heatmap_data = contingency_table.reset_index().melt(
        id_vars="Wilderness_Area", var_name="Cover_Type", value_name="Count"
    )

    heatmap_data["Cover_Type"] = heatmap_data["Cover_Type"].map(cover_types)
    heatmap_data["Wilderness_Area"] = heatmap_data["Wilderness_Area"].map(
        wilderness_areas
    )

    n_categories = len(cover_types)
    bar_width = 180

    chart = (
        alt.Chart(cover_counts)
        .mark_bar(color="green")
        .encode(
            x=alt.X(
                "Cover Type:N",
                sort=list(cover_types.values()),
                axis=alt.Axis(labelAngle=-40, labelOverlap=False),
            ),
            y="Count:Q",
        )
        .properties(
            title="Forest Cover Type Distribution",
            width=bar_width * n_categories,
            height=400,
        )
    )

    st.altair_chart(chart, use_container_width=False)

    n_cover_types = len(cover_types)
    n_wilderness_areas = contingency_table.shape[0]
    cell_size = 180

    heatmap = (
        alt.Chart(heatmap_data)
        .mark_rect()
        .encode(
            x=alt.X(
                "Cover_Type:N",
                title="Cover Type",
                sort=list(cover_types.values()),
                axis=alt.Axis(labelAngle=-40, labelOverlap=False),
            ),
            y=alt.Y(
                "Wilderness_Area:N",
                title="Wilderness Area",
                axis=alt.Axis(labelOverlap=False, labelLimit=0),
            ),
            color=alt.Color(
                "Count:Q",
                title="Number of observations",
                scale=alt.Scale(scheme="blues", reverse=False),
            ),
            tooltip=[
                alt.Tooltip("Wilderness_Area:N", title="Wilderness Area"),
                alt.Tooltip("Cover_Type:N", title="Cover Type"),
                alt.Tooltip("Count:Q", title="Count"),
            ],
        )
        .properties(
            title="Wilderness Area by Cover Type",
            width=cell_size * n_cover_types,
            height=cell_size * n_wilderness_areas,
        )
    )

    text = heatmap.mark_text(baseline="middle", fontSize=11).encode(
        text=alt.Text("Count:Q"),
        color=alt.condition(
            alt.datum.Count > 1000, alt.value("white"), alt.value("black")
        ),
    )

    st.altair_chart(heatmap + text, use_container_width=False)

    # violin plot

    violin_data = raw_train.copy()
    violin_data["Cover_Type"] = violin_data["Cover_Type"].map(cover_types)

    n_cover_types = len(cover_types)
    box_width_per_category = 180

    fig = px.violin(
        violin_data,
        x="Cover_Type",
        y="Elevation",
        box=True,
        points=False,
        labels={"Cover_Type": "Forest Cover Type", "Elevation": "Elevation (m)"},
        title="Elevation Distribution by Forest Cover Type",
    )

    fig.update_layout(
        width=box_width_per_category * n_cover_types,
        height=600,
        xaxis={"tickangle": -40},
    )

    st.plotly_chart(fig, use_container_width=False)

    ### examine an engineered feature
    st.header("Explore Transformed Data (Raw + Engineered Features)")

    options_features = X_train.columns
    feature = st.selectbox("Select feature", options_features)

    fig = px.histogram(X_train, x=feature, nbins=40, title=f"Distribution of {feature}")

    st.plotly_chart(fig, use_container_width=True)
