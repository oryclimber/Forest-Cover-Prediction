import streamlit as st
from tabs.dataset_tab import render_dataset_tab
from tabs.model_tab import render_model_tab
from tabs.prediction_tab import render_prediction_tab

from forest_cover_prediction.data import data_pipeline

st.set_page_config(layout="wide")

st.title("Forest Cover Type Prediction")


@st.cache_data
def get_processed_data():
    return data_pipeline()


raw_train, X_train, X_test, y = get_processed_data()

tab1, tab2, tab3 = st.tabs(["📊 Dataset", "🤖 Model", "🌲 Prediction"])

with tab1:
    render_dataset_tab(raw_train, X_train)

with tab2:
    best_model = render_model_tab(X_train, y)

with tab3:
    render_prediction_tab(best_model, list(X_train.columns))
