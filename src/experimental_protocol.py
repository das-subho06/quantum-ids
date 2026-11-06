
"""
Experimental protocol for the UNSW-NB15 quantum IDS study.

Creates reproducible leave-one-attack-family-out split manifests.
No model training is performed in this file.

Important:
- Only normal training rows are used to fit one-class models.
- Preprocessing must later be fitted only on the selected normal
  training subset for each seed.
- A held-out attack family is excluded from validation/tuning.
- Official test rows are reserved for final evaluation.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "experimental_splits"

RANDOM_SEEDS = [42, 123, 2025, 2026, 3407]

NORMAL_TRAIN_SIZE = 300
NORMAL_VALIDATION_SIZE = 2000
MAX_VALIDATION_ATTACKS_PER_FAMILY = 300

# Attack families with fewer official test examples than this
# are retained, but flagged as statistically uncertain.
MIN_TEST_FAMILY_COUNT = 100


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_datasets():
    """Load the official training and testing CSV files."""

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training CSV not found:\n{TRAIN_PATH}\n"
            "Check that the official training file is inside data/raw/."
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Testing CSV not found:\n{TEST_PATH}\n"
            "Check that the official testing file is inside data/raw/."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    required_columns = {"label", "attack_cat"}

    for dataset_name, df in [
        ("training", train_df),
        ("testing", test_df),
    ]:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{dataset_name} dataset is missing columns: {missing}"
            )

    if list(train_df.columns) != list(test_df.columns):
        raise ValueError(
            "Training and testing columns do not match. "
            "Check the official CSV files before proceeding."
        )

    return train_df, test_df


def sample_indices(indices, maximum_size, seed):
    """Return a reproducible sample of row indices."""

    indices = np.asarray(indices, dtype=int)

    if len(indices) <= maximum_size:
        return indices

    rng = np.random.default_rng(seed)

    return np.sort(
        rng.choice(indices, size=maximum_size, replace=False)
    )


def save_manifest(manifest, output_path):
    """Write a split manifest as readable JSON."""

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


# ---------------------------------------------------------
# Main experimental protocol
# ---------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_datasets()

    # Reset indices so every manifest index maps directly
    # to the corresponding row position in the loaded CSV.
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_labels = train_df["label"].to_numpy()
    test_labels = test_df["label"].to_numpy()

    train_families = (
        train_df["attack_cat"].fillna("Unknown").astype(str).to_numpy()
    )
    test_families = (
        test_df["attack_cat"].fillna("Unknown").astype(str).to_numpy()
    )

    # In UNSW-NB15, label 0 denotes normal traffic.
    normal_train_indices = np.flatnonzero(train_labels == 0)
    attack_train_indices = np.flatnonzero(train_labels == 1)

    normal_test_indices = np.flatnonzero(test_labels == 0)
    attack_test_indices = np.flatnonzero(test_labels == 1)

    # Use actual attack-family labels, excluding the normal class.
    attack_families = sorted(
        family
        for family in np.unique(train_families[attack_train_indices])
        if family.lower() != "normal"
    )

    if not attack_families:
        raise ValueError("No attack families were found in the training CSV.")

    print("=" * 70)
    print("UNSW-NB15 EXPERIMENTAL PROTOCOL")
    print("=" * 70)

    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows:  {len(test_df):,}")
    print(f"Normal training rows: {len(normal_train_indices):,}")
    print(f"Normal testing rows:  {len(normal_test_indices):,}")
    print(f"Attack families: {len(attack_families)}")
    print(f"Random seeds: {RANDOM_SEEDS}")

    # Record family counts before generating folds.
    family_counts = []

    for family in attack_families:
        train_count = int(
            np.sum(
                (train_labels == 1) & (train_families == family)
            )
        )
        test_count = int(
            np.sum(
                (test_labels == 1) & (test_families == family)
            )
        )

        family_counts.append({
            "family": family,
            "training_count": train_count,
            "testing_count": test_count,
            "low_test_support": test_count < MIN_TEST_FAMILY_COUNT,
        })

    counts_path = OUTPUT_DIR / "attack_family_counts.csv"
    pd.DataFrame(family_counts).to_csv(counts_path, index=False)

    manifest_summary = []

    # Each family gets its own held-out fold.
    for held_out_family in attack_families:

        print(f"\nHeld-out family: {held_out_family}")

        held_out_test_indices = np.flatnonzero(
            (test_labels == 1) & (test_families == held_out_family)
        )

        if len(held_out_test_indices) == 0:
            print("  WARNING: No examples in official test set; skipping.")
            continue

        # All normal test rows plus only the current held-out
        # family form the primary family-specific test set.
        final_test_indices = np.sort(
            np.concatenate([
                normal_test_indices,
                held_out_test_indices,
            ])
        )

        for seed in RANDOM_SEEDS:

            # Split NORMAL training rows only.
            # The held-out normal portion is used for validation
            # and never used to fit the one-class detector.
            normal_fit_pool, normal_validation_pool = train_test_split(
                normal_train_indices,
                test_size=0.20,
                random_state=seed,
                shuffle=True,
            )

            if len(normal_fit_pool) < NORMAL_TRAIN_SIZE:
                raise ValueError(
                    "Not enough normal training rows for the requested "
                    f"sample size of {NORMAL_TRAIN_SIZE}."
                )

            # The one-class model receives about 300 normal flows.
            rng = np.random.default_rng(seed)
            normal_train_sample = np.sort(
                rng.choice(
                    normal_fit_pool,
                    size=NORMAL_TRAIN_SIZE,
                    replace=False,
                )
            )

            # Use a capped, reproducible normal validation set.
            normal_validation_sample = sample_indices(
                normal_validation_pool,
                NORMAL_VALIDATION_SIZE,
                seed + 10,
            )

            # Validation attacks come from the official training CSV,
            # but never from the family currently held out.
            validation_attack_indices = attack_train_indices[
                train_families[attack_train_indices] != held_out_family
            ]

            # Cap each family separately so large attack families do
            # not dominate validation hyperparameter selection.
            validation_attack_sample_parts = []

            eligible_validation_families = sorted(
                family
                for family in np.unique(
                    train_families[validation_attack_indices]
                )
                if family.lower() != "normal"
            )

            for family_number, family in enumerate(
                eligible_validation_families
            ):
                family_indices = validation_attack_indices[
                    train_families[validation_attack_indices] == family
                ]

                sampled_family_indices = sample_indices(
                    family_indices,
                    MAX_VALIDATION_ATTACKS_PER_FAMILY,
                    seed + 100 + family_number,
                )

                validation_attack_sample_parts.append(
                    sampled_family_indices
                )

            if validation_attack_sample_parts:
                validation_attack_sample = np.sort(
                    np.concatenate(validation_attack_sample_parts)
                )
            else:
                validation_attack_sample = np.array([], dtype=int)

            # Validation set: normal validation flows plus attacks
            # from non-held-out families only.
            validation_indices = np.sort(
                np.concatenate([
                    normal_validation_sample,
                    validation_attack_sample,
                ])
            )

            
# Keep labels and family names aligned with validation_indices.
            validation_labels = train_labels[validation_indices]

            validation_families = train_families[validation_indices]

            # Save the exact row positions and labels needed to
            # reconstruct this experiment later.
            manifest = {
                "dataset": "UNSW-NB15",
                "seed": int(seed),
                "held_out_family": held_out_family,
                "normal_train_size": int(len(normal_train_sample)),
                "normal_validation_size": int(
                    len(normal_validation_sample)
                ),
                "validation_attack_size": int(
                    len(validation_attack_sample)
                ),
                "validation_attack_families": eligible_validation_families,
                "train_csv": str(TRAIN_PATH.relative_to(PROJECT_ROOT)),
                "test_csv": str(TEST_PATH.relative_to(PROJECT_ROOT)),
                "train_normal_indices": normal_train_sample.tolist(),
                "validation_indices": validation_indices.tolist(),
                "validation_labels": validation_labels.tolist(),
                "validation_families": validation_families.tolist(),
                "validation_normal_indices": (
                    normal_validation_sample.tolist()
                ),
                "validation_attack_indices": (
                    validation_attack_sample.tolist()
                ),
                "final_test_indices": final_test_indices.tolist(),
                "final_test_normal_count": int(len(normal_test_indices)),
                "final_test_held_out_attack_count": int(
                    len(held_out_test_indices)
                ),
                "preprocessing_rule": (
                    "Fit encoder, scaler, and PCA only on "
                    "train_normal_indices."
                ),
                "threshold_rule": (
                    "Choose the 1% FPR threshold using validation "
                    "normal scores only; do not use test labels."
                ),
            }

            family_dir = OUTPUT_DIR / held_out_family.replace("/", "_")
            family_dir.mkdir(parents=True, exist_ok=True)

            manifest_path = family_dir / f"seed_{seed}.json"
            save_manifest(manifest, manifest_path)

            manifest_summary.append({
                "held_out_family": held_out_family,
                "seed": seed,
                "normal_train_count": len(normal_train_sample),
                "normal_validation_count": len(normal_validation_sample),
                "validation_attack_count": len(validation_attack_sample),
                "test_normal_count": len(normal_test_indices),
                "test_held_out_attack_count": len(held_out_test_indices),
                "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            })

            print(
                f"  Seed {seed}: train normal={len(normal_train_sample)}, "
                f"validation normal={len(normal_validation_sample)}, "
                f"validation attacks={len(validation_attack_sample)}, "
                f"test held-out attacks={len(held_out_test_indices)}"
            )

    summary_path = OUTPUT_DIR / "protocol_summary.csv"
    pd.DataFrame(manifest_summary).to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("PROTOCOL SPLITS CREATED")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Family counts:    {counts_path}")
    print(f"Protocol summary: {summary_path}")
    print(f"Manifest files:   {len(manifest_summary)}")
    print("\nNext: implement fold-specific preprocessing using these manifests.")


if __name__ == "__main__":
    main()
