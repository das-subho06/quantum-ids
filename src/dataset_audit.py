
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Find the official training and testing CSVs
# --------------------------------------------------

csv_files = list(RAW_DIR.glob("*.csv"))

training_files = [
    file for file in csv_files
    if "training" in file.name.lower()
    or "train" in file.name.lower()
]

testing_files = [
    file for file in csv_files
    if "testing" in file.name.lower()
    or "test" in file.name.lower()
]

if len(training_files) != 1 or len(testing_files) != 1:
    raise FileNotFoundError(
        "\nExpected exactly one training CSV and one testing CSV "
        f"inside:\n{RAW_DIR}\n\n"
        f"Found CSV files: {[file.name for file in csv_files]}\n"
        "Keep the original filenames, or update the file-selection "
        "logic to match your filenames."
    )

TRAIN_PATH = training_files[0]
TEST_PATH = testing_files[0]

print("Training file:", TRAIN_PATH.name)
print("Testing file:", TEST_PATH.name)


# --------------------------------------------------
# 3. Load datasets
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
test_df = pd.read_csv(TEST_PATH, low_memory=False)

print("\n========== DATASET SHAPES ==========")
print("Training:", train_df.shape)
print("Testing: ", test_df.shape)

print("\n========== COLUMN NAMES ==========")
print("Training columns:")
print(train_df.columns.tolist())

print("\nTesting columns:")
print(test_df.columns.tolist())


# --------------------------------------------------
# 4. Check target columns
# --------------------------------------------------

def find_column(df, candidates):
    """Find a column without depending on capitalization or spaces."""
    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    return None


label_candidates = ["label", "class"]
attack_candidates = ["attack_cat", "attack category"]

train_label_col = find_column(train_df, label_candidates)
test_label_col = find_column(test_df, label_candidates)

train_attack_col = find_column(train_df, attack_candidates)
test_attack_col = find_column(test_df, attack_candidates)

if train_label_col is None or test_label_col is None:
    raise ValueError(
        "Could not find the binary label column. "
        "Inspect the printed column names and update label_candidates."
    )

print("\n========== BINARY LABEL DISTRIBUTION ==========")

print("\nTraining:")
print(train_df[train_label_col].value_counts(dropna=False))

print("\nTesting:")
print(test_df[test_label_col].value_counts(dropna=False))


# --------------------------------------------------
# 5. Count normal traffic and attack families
# --------------------------------------------------

if train_attack_col is not None and test_attack_col is not None:
    print("\n========== TRAINING ATTACK FAMILIES ==========")
    print(train_df[train_attack_col].value_counts(dropna=False))

    print("\n========== TESTING ATTACK FAMILIES ==========")
    print(test_df[test_attack_col].value_counts(dropna=False))

    train_counts = (
        train_df[train_attack_col]
        .fillna("Missing")
        .astype(str)
        .value_counts()
    )

    test_counts = (
        test_df[test_attack_col]
        .fillna("Missing")
        .astype(str)
        .value_counts()
    )

    train_counts.to_csv(
        METRICS_DIR / "training_attack_family_counts.csv",
        header=["count"]
    )

    test_counts.to_csv(
        METRICS_DIR / "testing_attack_family_counts.csv",
        header=["count"]
    )

    # Training distribution
    ax = train_counts.sort_values().plot(
        kind="barh",
        figsize=(9, 6),
        title="UNSW-NB15 Training Attack-Family Distribution"
    )
    ax.set_xlabel("Number of records")
    ax.set_ylabel("Attack family")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "training_attack_families.png",
        dpi=200
    )
    plt.close()

    # Testing distribution
    ax = test_counts.sort_values().plot(
        kind="barh",
        figsize=(9, 6),
        title="UNSW-NB15 Testing Attack-Family Distribution"
    )
    ax.set_xlabel("Number of records")
    ax.set_ylabel("Attack family")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "testing_attack_families.png",
        dpi=200
    )
    plt.close()

else:
    print(
        "\nWARNING: Attack-family column not found in both datasets. "
        "Check the actual column names."
    )


# --------------------------------------------------
# 6. Missing values and duplicate records
# --------------------------------------------------

print("\n========== DATA QUALITY ==========")

quality_report = {}

for name, df in [
    ("training", train_df),
    ("testing", test_df),
]:
    missing = df.isna().sum()
    duplicate_count = int(df.duplicated().sum())

    quality_report[name] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": duplicate_count,
        "total_missing_cells": int(missing.sum()),
        "columns_with_missing_values": {
            str(column): int(count)
            for column, count in missing.items()
            if count > 0
        },
    }

    print(f"\n{name.upper()}")
    print("Duplicate rows:", duplicate_count)
    print("Total missing cells:", int(missing.sum()))

    if missing.any():
        print("Columns with missing values:")
        print(missing[missing > 0])
    else:
        print("No missing values found.")

with open(
    METRICS_DIR / "dataset_quality_report.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(quality_report, file, indent=4)


# --------------------------------------------------
# 7. Column types and categorical cardinality
# --------------------------------------------------

print("\n========== COLUMN TYPES ==========")
print(train_df.dtypes.to_string())

cardinality_report = []

for column in train_df.columns:
    cardinality_report.append({
        "column": str(column),
        "dtype": str(train_df[column].dtype),
        "unique_values": int(train_df[column].nunique(dropna=False)),
        "missing_values": int(train_df[column].isna().sum()),
    })

pd.DataFrame(cardinality_report).to_csv(
    METRICS_DIR / "column_cardinality.csv",
    index=False
)


# --------------------------------------------------
# 8. Verify train/test feature compatibility
# --------------------------------------------------

train_columns = set(train_df.columns)
test_columns = set(test_df.columns)

print("\n========== TRAIN/TEST COMPATIBILITY ==========")
print("Columns only in training:", sorted(train_columns - test_columns))
print("Columns only in testing:", sorted(test_columns - train_columns))

if train_columns == test_columns:
    print("Both datasets contain the same columns.")
else:
    print("WARNING: Column mismatch detected. Investigate before modeling.")


# --------------------------------------------------
# 9. Summary
# --------------------------------------------------

print("\n========== AUDIT COMPLETE ==========")
print("Reports saved to:", METRICS_DIR)
print("Figures saved to:", FIGURES_DIR)
