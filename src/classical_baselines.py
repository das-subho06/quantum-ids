
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
)


# ============================================================
# 1. Paths and configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unsw_nb15_quantum_features.npz"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Begin with a manageable sample for the first experiment.
# We will benchmark and tune the models after verifying this run.
NORMAL_TRAIN_LIMIT = 10000


# ============================================================
# 2. Load preprocessed data
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Preprocessed dataset not found: {DATA_PATH}\n"
        "Run python src\\preprocessing.py first."
    )

data = np.load(DATA_PATH, allow_pickle=False)

X_train = data["X_train_classical"]
X_test = data["X_test_classical"]

y_train = data["y_train"]
y_test = data["y_test"]

normal_mask = data["normal_train_mask"].astype(bool)

families_test = data["families_test"].astype(str)

X_normal_train = X_train[normal_mask]

print("Normal training data:", X_normal_train.shape)
print("Test data:", X_test.shape)
print("Test normal records:", int(np.sum(y_test == 0)))
print("Test attack records:", int(np.sum(y_test == 1)))


# ============================================================
# 3. Sample normal training records
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)

if len(X_normal_train) > NORMAL_TRAIN_LIMIT:
    selected_indices = rng.choice(
        len(X_normal_train),
        size=NORMAL_TRAIN_LIMIT,
        replace=False,
    )
    X_normal_fit = X_normal_train[selected_indices]
else:
    X_normal_fit = X_normal_train

print("Records used to fit each model:", len(X_normal_fit))


# ============================================================
# 4. Define evaluation function
# ============================================================

def evaluate_detector(
    model_name,
    y_true,
    anomaly_scores,
    attack_families=None,
):
    """
    Larger anomaly scores must indicate a greater likelihood
    of an attack.
    """

    auroc = roc_auc_score(y_true, anomaly_scores)
    auprc = average_precision_score(y_true, anomaly_scores)

    fpr, tpr, thresholds = roc_curve(
        y_true,
        anomaly_scores,
        drop_intermediate=False,
    )

    # Highest observed TPR whose FPR does not exceed 1%.
    valid_indices = np.flatnonzero(fpr <= 0.01)
    tpr_at_1pct_fpr = float(tpr[valid_indices[-1]])

    result = {
        "model": model_name,
        "test_samples": int(len(y_true)),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "tpr_at_1pct_fpr": tpr_at_1pct_fpr,
    }

    print(f"\n========== {model_name} ==========")
    print(f"AUROC:       {auroc:.4f}")
    print(f"AUPRC:       {auprc:.4f}")
    print(f"TPR at 1% FPR: {tpr_at_1pct_fpr:.4f}")

    if attack_families is not None:
        print("\nAUROC by test attack family:")

        family_results = {}

        for family in sorted(set(attack_families)):
            family_mask = (
                (attack_families == family)
                & (y_true == 1)
            )

            if not np.any(family_mask):
                continue

            # Compare this attack family with all normal test flows.
            normal_scores = anomaly_scores[y_true == 0]
            family_scores = anomaly_scores[family_mask]

            family_labels = np.concatenate([
                np.zeros(len(normal_scores), dtype=np.int8),
                np.ones(len(family_scores), dtype=np.int8),
            ])

            combined_scores = np.concatenate([
                normal_scores,
                family_scores,
            ])

            family_auroc = roc_auc_score(
                family_labels,
                combined_scores,
            )

            family_results[str(family)] = {
                "attack_samples": int(family_mask.sum()),
                "auroc": float(family_auroc),
            }

            print(
                f"{family:18s} "
                f"n={int(family_mask.sum()):5d} "
                f"AUROC={family_auroc:.4f}"
            )

        result["attack_family_results"] = family_results

    return result


# ============================================================
# 5. Train and evaluate One-Class SVM
# ============================================================

# One-Class SVM decision_function is higher for more normal
# samples, so negate it to obtain an anomaly score.

ocsvm = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.05,
)

print("\nFitting RBF One-Class SVM...")
ocsvm.fit(X_normal_fit)

ocsvm_anomaly_scores = -ocsvm.decision_function(X_test).ravel()

ocsvm_result = evaluate_detector(
    model_name="RBF One-Class SVM",
    y_true=y_test,
    anomaly_scores=ocsvm_anomaly_scores,
    attack_families=families_test,
)


# ============================================================
# 6. Train and evaluate Isolation Forest
# ============================================================

# Isolation Forest score_samples is higher for more normal
# samples, so negate it for anomaly ranking.

isolation_forest = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("\nFitting Isolation Forest...")
isolation_forest.fit(X_normal_fit)

if_anomaly_scores = -isolation_forest.score_samples(X_test)

if_result = evaluate_detector(
    model_name="Isolation Forest",
    y_true=y_test,
    anomaly_scores=if_anomaly_scores,
    attack_families=families_test,
)


# ============================================================
# 7. Save results
# ============================================================

results = [ocsvm_result, if_result]

results_path = OUTPUT_DIR / "classical_baseline_results.json"

with open(results_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

summary = pd.DataFrame([
    {
        "model": result["model"],
        "auroc": result["auroc"],
        "auprc": result["auprc"],
        "tpr_at_1pct_fpr": result["tpr_at_1pct_fpr"],
    }
    for result in results
])

summary_path = OUTPUT_DIR / "classical_baseline_summary.csv"
summary.to_csv(summary_path, index=False)

print("\n========== BASELINES COMPLETE ==========")
print(summary.to_string(index=False))
print("\nSaved:", results_path)
print("Saved:", summary_path)
