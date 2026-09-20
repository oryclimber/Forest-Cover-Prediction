# Forest Cover Type Prediction

A Streamlit app that explores the Forest Cover Type dataset, benchmarks several
classification models, and lets users generate live predictions from manual feature input.

## Setup & Running

### Run with Docker (recommended)
```bash
docker build -t forest-cover-prediction .
docker run --rm -p 8501:8501 forest-cover-prediction
```
Then open **http://localhost:8505** in your browser.

### Run locally without Docker
```bash
pip install -r requirements.txt
streamlit run app/streamlit.py
```

## Project Structure
├── app/
│ ├── streamlit_app.py # Entrypoint
│ └── tabs/
│ ├── dataset_tab.py # Data exploration
│ ├── model_tab.py # Model benchmarking
│ └── prediction_tab.py # Live prediction interface
├── src/forest_cover_prediction/
│ ├── data.py # Data loading & feature engineering
│ └── models.py # Model definitions, benchmarking, tuning
├── data/raw/ # Training data
└── Dockerfile

## App Overview
- **Dataset tab** — cover type distribution, wilderness area breakdown, elevation by cover type
- **Model tab** — cross-validated comparison of Logistic Regression, KNN, Random Forest, and
  XGBoost, plus a confusion matrix for the best-performing model
- **Prediction tab** — manually adjust feature values and get a live predicted cover type with
  per-class confidence scores

## Data
This project uses the [Forest Cover Type dataset](<https://www.kaggle.com/competitions/dsaib-2025-2026-forest-cover-type-2/data>). The training data is bundled directly in this repo (`data/raw/`) so the app runs out of the box with no Kaggle account or API token required.
