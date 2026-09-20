import os

import numpy as np
import pandas as pd

ELU_CODES = {
    1: 2702,
    2: 2703,
    3: 2704,
    4: 2705,
    5: 2706,
    6: 2717,
    7: 3501,
    8: 3502,
    9: 4201,
    10: 4703,
    11: 4704,
    12: 4744,
    13: 4758,
    14: 5101,
    15: 5151,
    16: 6101,
    17: 6102,
    18: 6731,
    19: 7101,
    20: 7102,
    21: 7103,
    22: 7201,
    23: 7202,
    24: 7700,
    25: 7701,
    26: 7702,
    27: 7709,
    28: 7710,
    29: 7745,
    30: 7746,
    31: 7755,
    32: 7756,
    33: 7757,
    34: 7790,
    35: 8703,
    36: 8707,
    37: 8708,
    38: 8771,
    39: 8772,
    40: 8776,
}
ELU_DIGITS = {k: [int(d) for d in str(v)] for k, v in ELU_CODES.items()}

DATA_DIR = "data/raw"


def load_data():
    """Return path to local data directory."""
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test-full.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Expected data files not found in {DATA_DIR}. "
            "Make sure train.csv and test-full.csv are present."
        )

    return DATA_DIR


def inspect_data(path):
    """Load and inspect train and test data"""
    train_df = pd.read_csv(os.path.join(path, "train.csv"))
    test_df = pd.read_csv(os.path.join(path, "test-full.csv"))

    print("Training data shape:", train_df.shape)
    print("\nFirst few rows of training data:")
    print(train_df.head())
    print("\nTarget variable distribution:")
    print(train_df["Cover_Type"].value_counts().sort_index())

    return train_df, test_df


def prep_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, np.array]:
    """Separate features and target"""
    X_train = train_df.drop(columns=["Cover_Type"]).copy()
    # adjust from 1-7 labels to 0-6 for XGBoost Classifier
    y = train_df["Cover_Type"].copy() - 1
    X_test = test_df.copy()

    return X_train, X_test, y


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Creates engineered features"""
    original_features = df.shape[1]
    d = df.copy()

    # Hydrology
    hdist = d["Horizontal_Distance_To_Hydrology"]
    vdist = d["Vertical_Distance_To_Hydrology"]

    d["Hydro_Euclidean"] = np.sqrt(hdist**2 + vdist**2)
    d["Hydro_Manhattan"] = np.abs(hdist) + np.abs(vdist)  # often better for trees
    d["Hydro_Above_Water"] = (vdist > 0).astype(int)  # drainage vs moisture zone
    d["Hydro_Vertical_Sign"] = np.sign(vdist)  # -1/0/+1 categorical-ish

    # Distance
    road, fire = (
        d["Horizontal_Distance_To_Roadways"],
        d["Horizontal_Distance_To_Fire_Points"],
    )

    d["Dist_Road_minus_Hydro"] = road - hdist
    d["Dist_Fire_minus_Hydro"] = fire - hdist
    d["Dist_Road_minus_Fire"] = road - fire
    d["Dist_Road_plus_Hydro"] = road + hdist
    d["Dist_Road_plus_Fire"] = road + fire
    d["Dist_Mean_All"] = (road + hdist + fire) / 3
    d["Dist_Min_All"] = pd.concat([road, hdist, fire], axis=1).min(axis=1)
    d["Dist_Max_All"] = pd.concat([road, hdist, fire], axis=1).max(axis=1)

    # Elevation
    elev = d["Elevation"]

    # Binned elevation captures nonlinear species thresholds (montane/subalpine/alpine)
    d["Elevation_Bin"] = pd.cut(
        elev, bins=[0, 2000, 2500, 2750, 3000, 3250, 3500, 9999], labels=False
    ).astype(float)
    d["Elevation_x_Slope"] = elev * d["Slope"]
    d["Elev_minus_VHydro"] = elev - vdist  # elevation relative to local water table

    # Aspect
    aspect_rad = d["Aspect"] * np.pi / 180
    d["Aspect_sin"] = np.sin(aspect_rad)
    d["Aspect_cos"] = np.cos(aspect_rad)
    d["Aspect_Northness"] = np.cos(aspect_rad)  # +1 = north-facing (cool/moist)
    d["Aspect_Eastness"] = np.sin(aspect_rad)  # +1 = east-facing

    # Hillshade
    h9, h12, h3 = d["Hillshade_9am"], d["Hillshade_Noon"], d["Hillshade_3pm"]

    d["Hillshade_Mean"] = (h9 + h12 + h3) / 3
    d["Hillshade_Min"] = pd.concat([h9, h12, h3], axis=1).min(axis=1)
    d["Hillshade_Max"] = pd.concat([h9, h12, h3], axis=1).max(axis=1)
    d["Hillshade_Range"] = d["Hillshade_Max"] - d["Hillshade_Min"]
    d["Hillshade_AM_change"] = h12 - h9  # morning ramp (driven by aspect)
    d["Hillshade_PM_change"] = h3 - h12  # afternoon ramp (slope + aspect)
    d["Hillshade_Day_swing"] = h3 - h9  # full-day contrast

    # Wilderness
    wild_cols = [c for c in d.columns if c.startswith("Wilderness_Area")]
    d["Wilderness_Idx"] = d[wild_cols].idxmax(axis=1).str.extract(r"(\d+)").astype(int)

    # Soil / EU
    soil_cols = [c for c in d.columns if c.startswith("Soil_Type")]
    d["Soil_Idx"] = d[soil_cols].idxmax(axis=1).str.extract(r"(\d+)").astype(int)

    # All four ELU digits — your original only used 2 of 4
    d["Soil_ClimaticZone"] = d["Soil_Idx"].map(
        lambda x: ELU_DIGITS.get(x, [0, 0, 0, 0])[0]
    )
    d["Soil_GeologicZone"] = d["Soil_Idx"].map(
        lambda x: ELU_DIGITS.get(x, [0, 0, 0, 0])[1]
    )
    d["Soil_CapabilityClass"] = d["Soil_Idx"].map(
        lambda x: ELU_DIGITS.get(x, [0, 0, 0, 0])[2]
    )
    d["Soil_Subtype"] = d["Soil_Idx"].map(lambda x: ELU_DIGITS.get(x, [0, 0, 0, 0])[3])
    d["Soil_IsRocky"] = (d["Soil_Idx"] >= 35).astype(int)  # rocky soils 35–40

    # Composite Terrain
    # Heat load: south-facing steep = high, north-facing gentle = low
    d["Heat_Load"] = d["Slope"] * (-d["Aspect_Northness"])

    # Topographic wetness proxy: wet = near water + gentle slope
    slope_safe = d["Slope"].replace(0, 0.1)
    d["Topo_Wetness"] = d["Hydro_Manhattan"] / slope_safe

    print("Original features:", original_features)
    print("Features after engineering:", d.shape[1])
    print("New features created:", d.shape[1] - original_features)

    d = d.drop("Id", axis=1)

    return d.fillna(0)


def data_pipeline():
    path = load_data()

    train_df, test_df = inspect_data(path)

    X_train, X_test, y = prep_data(train_df, test_df)

    X_train_fe = feature_engineering(X_train)
    X_test_fe = feature_engineering(X_test)

    return train_df, X_train_fe, X_test_fe, y


if __name__ == "__main__":
    train_df, X_train, X_test, y = data_pipeline()
