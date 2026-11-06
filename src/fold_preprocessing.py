
"""
Preprocess all UNSW-NB15 experimental folds.

- Discovers every family/seed manifest automatically.
- Reuses valid existing fold files.
- Regenerates missing or invalid fold files.
- Fits all preprocessing steps on the selected normal training rows only.
- Never fits preprocessing on validation or final test data.

Run:
    python src/fold_preprocessing.py

Optional:
    python src/fold_preprocessing.py --force
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "experimental_splits"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "folds"

CATEGORICAL_COLUMNS = ["proto", "service", "state"]
N_PCA_COMPONENTS = 8

REQUIRED_KEYS = {
    "X_train_classical",
    "X_validation_classical",
    "X_test_classical",
    "X_train_quantum",
    "X_validation_quantum",
    "X_test_quantum",
    "y_train",
    "y_validation",
    "y_test",
    "families_validation",
    "families_test",
    "train_indices",
    "validation_indices",
    "test_indices",
    "explained_variance_ratio",
}


# ============================================================
# LOAD DATA ONCE
# ============================================================

def load_source_data():
    """Load the original train/test CSV files once for the entire run."""

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training CSV not found: {TRAIN_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Testing CSV not found: {TEST_PATH}")

    if not SPLIT_DIR.exists():
        raise FileNotFoundError(
            f"Experimental splits not found: {SPLIT_DIR}\n"
            "Run src/experimental_protocol.py first."
        )

    train_df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)
    test_df = pd.read_csv(TEST_PATH).reset_index(drop=True)

    print(f"Training CSV: {train_df.shape}")
    print(f"Testing CSV:  {test_df.shape}")

    return train_df, test_df


# ============================================================
# CHECK WHETHER AN EXISTING FOLD IS USABLE
# ============================================================

def is_valid_fold(output_path, expected_train_size):
    """Check the required arrays and basic dimensions of a saved fold."""

    if not output_path.exists():
        return False

    try:
        with np.load(output_path, allow_pickle=False) as data:
            if not REQUIRED_KEYS.issubset(data.files):
                return False

            if len(data["X_train_classical"]) != expected_train_size:
                return False

            if len(data["X_train_quantum"]) != expected_train_size:
                return False

            if len(data["y_train"]) != expected_train_size:
                return False

            for key in (
                "X_train_classical",
                "X_validation_classical",
                "X_test_classical",
                "X_train_quantum",
                "X_validation_quantum",
                "X_test_quantum",
            ):
                array = data[key]

                if array.ndim != 2 or not np.isfinite(array).all():
                    return False

            if data["X_train_quantum"].shape[1] != N_PCA_COMPONENTS:
                return False

            if data["X_train_classical"].shape[1] != N_PCA_COMPONENTS:
                return False

            if len(data["y_validation"]) != len(
                data["X_validation_classical"]
            ):
                return False

            if len(data["y_test"]) != len(data["X_test_classical"]):
                return False

            if len(data["families_validation"]) != len(
                data["y_validation"]
            ):
                return False

            if len(data["families_test"]) != len(data["y_test"]):
                return False

            if len(data["train_indices"]) != expected_train_size:
                return False

        return True

    except (OSError, ValueError, KeyError, EOFError):
        return False


# ============================================================
# VALIDATE A MANIFEST
# ============================================================

def load_and_validate_manifest(manifest_path, train_df, test_df):
    """Load a fold manifest and verify its indices and labels."""

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    family = manifest["held_out_family"]
    seed = int(manifest["seed"])

    train_indices = np.asarray(
        manifest["train_normal_indices"], dtype=int
    )
    validation_indices = np.asarray(
        manifest["validation_indices"], dtype=int
    )
    test_indices = np.asarray(
        manifest["final_test_indices"], dtype=int
    )

    if len(train_indices) == 0:
        raise ValueError("Training sample is empty.")

    if len(validation_indices) == 0:
        raise ValueError("Validation sample is empty.")

    if len(test_indices) == 0:
        raise ValueError("Final test sample is empty.")

    for name, indices, source_length in (
        ("training", train_indices, len(train_df)),
        ("validation", validation_indices, len(train_df)),
        ("test", test_indices, len(test_df)),
    ):
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"Duplicate {name} indices detected.")

        if indices.min() < 0 or indices.max() >= source_length:
            raise IndexError(f"{name.capitalize()} index out of bounds.")

    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("Training and validation indices overlap.")

    if not np.all(
        train_df.iloc[train_indices]["label"].to_numpy() == 0
    ):
        raise ValueError("Training data contains attack rows.")

    actual_validation = train_df.iloc[validation_indices]

    actual_labels = actual_validation["label"].to_numpy(dtype=int)
    actual_families = (
        actual_validation["attack_cat"]
        .fillna("Unknown")
        .astype(str)
        .to_numpy()
    )

    if not np.array_equal(
        actual_labels,
        np.asarray(manifest["validation_labels"], dtype=int),
    ):
        raise ValueError("Validation labels do not match the source CSV.")

    if not np.array_equal(
        actual_families,
        np.asarray(manifest["validation_families"], dtype=str),
    ):
        raise ValueError("Validation families do not match the source CSV.")

    if np.any(actual_families == family):
        raise ValueError(
            f"Held-out family {family} appears in validation."
        )

    actual_test = test_df.iloc[test_indices]
    test_labels = actual_test["label"].to_numpy(dtype=int)
    test_families = (
        actual_test["attack_cat"]
        .fillna("Unknown")
        .astype(str)
        .to_numpy()
    )

    allowed = (test_labels == 0) | (
        (test_labels == 1) & (test_families == family)
    )

    if not np.all(allowed):
        raise ValueError(
            "Final test contains an unexpected attack family."
        )

    if not np.any(test_labels == 0):
        raise ValueError("Final test contains no normal rows.")

    if not np.any(
        (test_labels == 1) & (test_families == family)
    ):
        raise ValueError(
            f"Final test contains no {family} attack rows."
        )

    return (
        family,
        seed,
        train_indices,
        validation_indices,
        test_indices,
    )


# ============================================================
# PREPROCESS ONE FOLD
# ============================================================

def preprocess_one_fold(
    family,
    seed,
    train_indices,
    validation_indices,
    test_indices,
    train_df,
    test_df,
):
    """Fit preprocessing on training rows and transform the other splits."""

    excluded_columns = {"id", "label", "attack_cat"}
    feature_columns = [
        col for col in train_df.columns if col not in excluded_columns
    ]

    missing_columns = set(feature_columns) - set(test_df.columns)

    if missing_columns:
        raise ValueError(
            f"Test CSV is missing features: {sorted(missing_columns)}"
        )

    categorical_columns = [
        col for col in CATEGORICAL_COLUMNS if col in feature_columns
    ]
    numerical_columns = [
        col for col in feature_columns if col not in categorical_columns
    ]

    X_train_df = train_df.iloc[train_indices][feature_columns]
    X_validation_df = train_df.iloc[validation_indices][feature_columns]
    X_test_df = test_df.iloc[test_indices][feature_columns]

    print(f"  Training normal rows: {len(X_train_df)}")
    print(f"  Validation rows:      {len(X_validation_df)}")
    print(f"  Final test rows:      {len(X_test_df)}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), numerical_columns),
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
    )

    # Fit only on the selected normal training samples.
    X_train_encoded = preprocessor.fit_transform(X_train_df)

    # Transform validation and test without fitting.
    X_validation_encoded = preprocessor.transform(X_validation_df)
    X_test_encoded = preprocessor.transform(X_test_df)

    if min(X_train_encoded.shape) < N_PCA_COMPONENTS:
        raise ValueError(
            f"Cannot fit {N_PCA_COMPONENTS} PCA components on "
            f"training shape {X_train_encoded.shape}."
        )

    pca = PCA(
        n_components=N_PCA_COMPONENTS,
        random_state=seed,
    )

    X_train_pca = pca.fit_transform(X_train_encoded)
    X_validation_pca = pca.transform(X_validation_encoded)
    X_test_pca = pca.transform(X_test_encoded)

    # Standardization also fits only on the training data.
    scaler = StandardScaler()

    X_train_classical = scaler.fit_transform(X_train_pca)
    X_validation_classical = scaler.transform(X_validation_pca)
    X_test_classical = scaler.transform(X_test_pca)

    # Map classical features to quantum rotation angles in (0, pi).
    X_train_quantum = np.pi / 2 + np.arctan(X_train_classical)
    X_validation_quantum = np.pi / 2 + np.arctan(
        X_validation_classical
    )
    X_test_quantum = np.pi / 2 + np.arctan(X_test_classical)

    y_train = train_df.iloc[train_indices]["label"].to_numpy(dtype=int)
    y_validation = train_df.iloc[validation_indices]["label"].to_numpy(
        dtype=int
    )
    y_test = test_df.iloc[test_indices]["label"].to_numpy(dtype=int)

    families_validation = (
        train_df.iloc[validation_indices]["attack_cat"]
        .fillna("Unknown")
        .astype(str)
        .to_numpy(dtype=str)
    )
    families_test = (
        test_df.iloc[test_indices]["attack_cat"]
        .fillna("Unknown")
        .astype(str)
        .to_numpy(dtype=str)
    )

    output_dir = OUTPUT_DIR / family
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"seed_{seed}.npz"

    # Write to a temporary file first to avoid leaving a partial result.
    temporary_path = output_dir / f"seed_{seed}.tmp.npz"

    np.savez_compressed(
        temporary_path,
        X_train_classical=X_train_classical,
        X_validation_classical=X_validation_classical,
        X_test_classical=X_test_classical,
        X_train_quantum=X_train_quantum,
        X_validation_quantum=X_validation_quantum,
        X_test_quantum=X_test_quantum,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        families_validation=families_validation,
        families_test=families_test,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )

    temporary_path.replace(output_path)

    if not is_valid_fold(output_path, len(train_indices)):
        raise ValueError(f"Saved fold failed verification: {output_path}")

    print(f"  Encoded feature count: {X_train_encoded.shape[1]}")
    print(
        "  PCA explained variance: "
        f"{pca.explained_variance_ratio_.sum():.4f}"
    )
    print(f"  Saved: {output_path}")

    return output_path


# ============================================================
# RUN ALL FOLDS
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess all UNSW-NB15 experiment folds."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate every fold, including existing valid folds.",
    )
    args = parser.parse_args()

    train_df, test_df = load_source_data()

    manifests = sorted(SPLIT_DIR.glob("*/seed_*.json"))

    if not manifests:
        raise FileNotFoundError(
            f"No manifests found in {SPLIT_DIR}. "
            "Run src/experimental_protocol.py first."
        )

    print("\n" + "=" * 70)
    print("ALL-FOLD PREPROCESSING")
    print("=" * 70)
    print(f"Manifests discovered: {len(manifests)}")
    print(f"Force regeneration:   {args.force}")

    reused = 0
    generated = 0
    failed = []

    for number, manifest_path in enumerate(manifests, start=1):
        family = manifest_path.parent.name

        try:
            with manifest_path.open("r", encoding="utf-8") as file:
                manifest_header = json.load(file)

            seed = int(manifest_header["seed"])
            expected_train_size = len(manifest_header["train_normal_indices"])
            output_path = OUTPUT_DIR / family / f"seed_{seed}.npz"

            print(
                f"\n[{number}/{len(manifests)}] "
                f"{family} / seed {seed}"
            )

            if (
                not args.force
                and is_valid_fold(output_path, expected_train_size)
            ):
                print("  REUSED: existing fold is valid.")
                reused += 1
                continue

            (
                validated_family,
                validated_seed,
                train_indices,
                validation_indices,
                test_indices,
            ) = load_and_validate_manifest(
                manifest_path, train_df, test_df
            )

            preprocess_one_fold(
                validated_family,
                validated_seed,
                train_indices,
                validation_indices,
                test_indices,
                train_df,
                test_df,
            )

            generated += 1

        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            failed.append(
                (manifest_path.parent.name, manifest_path.name, str(exc))
            )

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Manifests discovered: {len(manifests)}")
    print(f"Valid folds reused:   {reused}")
    print(f"Folds generated:      {generated}")
    print(f"Failures:             {len(failed)}")

    if failed:
        print("\nFailed folds:")
        for family, manifest_name, error in failed:
            print(f"- {family}/{manifest_name}: {error}")

        print("\nRerun the same command after fixing the reported issue.")
        print("Already valid folds will be reused.")
        sys.exit(1)

    print("\nAll folds are ready for the experiment runner.")


if __name__ == "__main__":
    main()
