
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


# ============================================================
# 1. Paths and configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

N_COMPONENTS = 8
RANDOM_STATE = 42

TRAIN_PATH = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = RAW_DIR / "UNSW_NB15_testing-set.csv"


# ============================================================
# 2. Load the official datasets
# ============================================================

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Training dataset not found: {TRAIN_PATH}")

if not TEST_PATH.exists():
    raise FileNotFoundError(f"Testing dataset not found: {TEST_PATH}")

train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
test_df = pd.read_csv(TEST_PATH, low_memory=False)

print("Training dataset:", train_df.shape)
print("Testing dataset:", test_df.shape)


# ============================================================
# 3. Identify labels and feature columns
# ============================================================

required_columns = {"id", "label", "attack_cat"}

for dataset_name, df in [
    ("training", train_df),
    ("testing", test_df),
]:
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_name} dataset is missing: {missing_columns}"
        )

    if df["label"].isna().any():
        raise ValueError(f"Missing labels in {dataset_name} dataset.")

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(
            f"Unexpected label values in {dataset_name} dataset."
        )

# Remove identifiers and target information from model inputs.
excluded_columns = ["id", "label", "attack_cat"]

feature_columns = [
    column
    for column in train_df.columns
    if column not in excluded_columns
]

if set(feature_columns) != (
    set(test_df.columns) - set(excluded_columns)
):
    raise ValueError("Training and testing feature columns do not match.")

X_train_df = train_df[feature_columns].copy()
X_test_df = test_df[feature_columns].copy()

# Save ground-truth labels separately for evaluation.
y_train = train_df["label"].to_numpy(dtype=np.int8)
y_test = test_df["label"].to_numpy(dtype=np.int8)

# Keep attack-family labels separate from the model inputs.
families_train = (
    train_df["attack_cat"].fillna("Unknown").astype(str).to_numpy(dtype=str)
)
families_test = (
    test_df["attack_cat"].fillna("Unknown").astype(str).to_numpy(dtype=str)
)

# Only normal training records may fit preprocessing.
normal_mask = y_train == 0
X_normal_train_df = X_train_df.loc[normal_mask]

print("\nFeature count before encoding:", len(feature_columns))
print("Normal training records:", len(X_normal_train_df))
print("Attack training records:", int((y_train == 1).sum()))


# ============================================================
# 4. Identify numerical and categorical features
# ============================================================

numeric_columns = X_train_df.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = [
    column
    for column in feature_columns
    if column not in numeric_columns
]

print("\nNumerical columns:", len(numeric_columns))
print("Categorical columns:", categorical_columns)


# ============================================================
# 5. Build the feature encoder
# ============================================================

# Numerical features are standardized.
# Categorical features are one-hot encoded.
#
# Dense one-hot output is used here so that PCA can operate
# directly on the resulting matrix.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_columns,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_columns,
        ),
    ],
    remainder="drop",
    sparse_threshold=0.0,
)

# CRITICAL: fit on normal training data ONLY.
preprocessor.fit(X_normal_train_df)

print("\nFeature encoder fitted on normal training data.")


# ============================================================
# 6. Transform training and testing data
# ============================================================

# Transforming other records does not fit or update the encoder.
X_normal_encoded = preprocessor.transform(X_normal_train_df)
X_train_encoded = preprocessor.transform(X_train_df)
X_test_encoded = preprocessor.transform(X_test_df)

print("\nEncoded normal training shape:", X_normal_encoded.shape)
print("Encoded full training shape:", X_train_encoded.shape)
print("Encoded testing shape:", X_test_encoded.shape)


# ============================================================
# 7. Fit PCA on normal training data only
# ============================================================

if min(X_normal_encoded.shape) < N_COMPONENTS:
    raise ValueError(
        f"Not enough data/features for {N_COMPONENTS} PCA components."
    )

pca = PCA(
    n_components=N_COMPONENTS,
    svd_solver="randomized",
    random_state=RANDOM_STATE,
)

pca.fit(X_normal_encoded)

# Transform all rows using the already-fitted PCA.
X_train_pca = pca.transform(X_train_encoded)
X_test_pca = pca.transform(X_test_encoded)

print("\nPCA explained variance ratio:")
for index, ratio in enumerate(pca.explained_variance_ratio_, start=1):
    print(f"Component {index}: {ratio:.6f}")

print(
    "Total variance explained by 8 components:",
    round(float(pca.explained_variance_ratio_.sum()), 6),
)


# ============================================================
# 8. Prepare classical and quantum representations
# ============================================================

# Classical models use PCA scores standardized using NORMAL
# training records only.
classical_scaler = StandardScaler()
classical_scaler.fit(X_train_pca[normal_mask])

X_train_classical = classical_scaler.transform(X_train_pca).astype(
    np.float32
)
X_test_classical = classical_scaler.transform(X_test_pca).astype(
    np.float32
)

# Quantum feature maps require bounded angular inputs.
# Standardize using normal training data, then apply a smooth,
# bounded transformation. Unlike clipping, this preserves a
# gradual change for extreme observations.
#
# theta = pi/2 + arctan(z)
# This maps every finite standardized value into (0, pi).

X_train_quantum = (
    np.pi / 2 + np.arctan(X_train_classical)
).astype(np.float32)

X_test_quantum = (
    np.pi / 2 + np.arctan(X_test_classical)
).astype(np.float32)

print("\nClassical feature matrix shapes:")
print("Training:", X_train_classical.shape)
print("Testing: ", X_test_classical.shape)

print("\nQuantum feature matrix shapes:")
print("Training:", X_train_quantum.shape)
print("Testing: ", X_test_quantum.shape)

print("\nQuantum angle ranges:")
print(
    "Training:",
    float(X_train_quantum.min()),
    "to",
    float(X_train_quantum.max()),
)
print(
    "Testing:",
    float(X_test_quantum.min()),
    "to",
    float(X_test_quantum.max()),
)

print(
    "\nVariance explained by PCA:",
    round(float(pca.explained_variance_ratio_.sum()), 6),
)

# ============================================================
# 9. Save transformed datasets and preprocessing objects
# ============================================================

np.savez_compressed(
    PROCESSED_DIR / "unsw_nb15_quantum_features.npz",
    X_train_classical=X_train_classical,
    X_test_classical=X_test_classical,
    X_train_quantum=X_train_quantum,
    X_test_quantum=X_test_quantum,
    X_train_pca=X_train_pca.astype(np.float32),
    X_test_pca=X_test_pca.astype(np.float32),
    y_train=y_train,
    y_test=y_test,
    families_train=families_train,
    families_test=families_test,
    normal_train_mask=normal_mask,
)

joblib.dump(
    {
        "preprocessor": preprocessor,
        "pca": pca,
        "classical_scaler": classical_scaler,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "n_components": N_COMPONENTS,
        "random_state": RANDOM_STATE,
    },
    PROCESSED_DIR / "preprocessing_objects.joblib",
)

print("\nPREPROCESSING COMPLETE")
print("Saved features:", PROCESSED_DIR / "unsw_nb15_quantum_features.npz")
print("Saved preprocessing objects:",
      PROCESSED_DIR / "preprocessing_objects.joblib")
