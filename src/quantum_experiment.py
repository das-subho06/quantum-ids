
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.svm import OneClassSVM

from qiskit.circuit.library import (
    ZFeatureMap,
    ZZFeatureMap,
    PauliFeatureMap,
)
from qiskit.quantum_info import Statevector


# ============================================================
# PATHS AND EXPERIMENT SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
FOLDS_DIR = ROOT / "data" / "processed" / "folds"
CACHE_DIR = ROOT / "data" / "processed" / "quantum_cache"
RESULTS_DIR = ROOT / "results"

RESULTS_CSV = RESULTS_DIR / "experiment_results.csv"
ERRORS_CSV = RESULTS_DIR / "experiment_errors.csv"

# One fixed schema for every model row. Never append rows with different columns.
RESULT_COLUMNS = [
    "fold", "heldout_family", "seed", "n_train", "n_validation", "n_test",
    "test_sampling", "model", "feature_map", "n_features", "reps", "bandwidth",
    "validation_auroc", "validation_threshold", "best_nu", "tuning_details",
    "runtime_seconds", "auroc", "auprc",
    "tpr_at_validation_1pct_threshold", "observed_fpr",
    "kernel_offdiag_mean", "kernel_offdiag_std", "kernel_effective_rank",
    "kernel_target_alignment",
]

FEATURE_DIMENSIONS = [4, 6, 8]
BANDWIDTHS = [0.5, 1.0, 2.0]
REPETITIONS = [1]

NU_VALUES = [0.01, 0.05, 0.10]
RBF_GAMMAS = ["scale", 0.01, 0.1, 1.0, 10.0]

RANDOM_SEED = 2026

# Keep memory and runtime manageable on a laptop.
MAX_VALIDATION_NORMALS = 1000
MAX_VALIDATION_ATTACKS_PER_FAMILY = 100
MAX_TEST_NORMALS = 1000
MAX_TEST_ATTACKS = 1000

# Set --full-test to use every available test row.
# Set --all-validation to use the entire validation set.
# These options can substantially increase runtime and memory use.


# ============================================================
# REPRODUCIBLE SAMPLING
# ============================================================

def stratified_indices(
    labels,
    families,
    max_normals,
    max_attacks_per_family,
    seed,
):
    """Sample normal and attack rows reproducibly."""

    labels = np.asarray(labels).astype(int)
    families = np.asarray(families).astype(str)

    rng = np.random.default_rng(seed)
    selected = []

    normal_idx = np.flatnonzero(labels == 0)

    if len(normal_idx) > max_normals:
        normal_idx = rng.choice(
            normal_idx, size=max_normals, replace=False
        )

    selected.extend(normal_idx.tolist())

    attack_families = sorted(
        set(families[labels == 1].tolist())
    )

    for family in attack_families:
        family_idx = np.flatnonzero(
            (labels == 1) & (families == family)
        )

        if len(family_idx) > max_attacks_per_family:
            family_idx = rng.choice(
                family_idx,
                size=max_attacks_per_family,
                replace=False,
            )

        selected.extend(family_idx.tolist())

    return np.asarray(sorted(selected), dtype=int)


def select_test_indices(labels, families, seed, full_test):
    """Keep the designated held-out attack family intact when small."""

    labels = np.asarray(labels).astype(int)
    families = np.asarray(families).astype(str)

    if full_test:
        return np.arange(len(labels))

    rng = np.random.default_rng(seed)
    selected = []

    normal_idx = np.flatnonzero(labels == 0)

    if len(normal_idx) > MAX_TEST_NORMALS:
        normal_idx = rng.choice(
            normal_idx,
            size=MAX_TEST_NORMALS,
            replace=False,
        )

    selected.extend(normal_idx.tolist())

    # Test manifests contain normal traffic plus the held-out family.
    for family in sorted(set(families[labels == 1].tolist())):
        idx = np.flatnonzero(
            (labels == 1) & (families == family)
        )

        if len(idx) > MAX_TEST_ATTACKS:
            idx = rng.choice(
                idx, size=MAX_TEST_ATTACKS, replace=False
            )

        selected.extend(idx.tolist())

    return np.asarray(sorted(selected), dtype=int)


# ============================================================
# QUANTUM FEATURE MAPS AND STATEVECTORS
# ============================================================

def make_feature_map(name, n_features, reps):
    if name == "Z":
        return ZFeatureMap(
            feature_dimension=n_features,
            reps=reps,
        )

    if name == "ZZ":
        return ZZFeatureMap(
            feature_dimension=n_features,
            reps=reps,
        )

    if name == "Pauli":
        return PauliFeatureMap(
            feature_dimension=n_features,
            reps=reps,
            paulis=["Z", "YY"],
        )

    raise ValueError(f"Unknown feature map: {name}")


def data_fingerprint(X):
    """Stable key for the numerical input array."""

    arr = np.ascontiguousarray(X, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def state_cache_path(fold_name, split_name, map_name, dimension, reps, bw, X):
    fingerprint = data_fingerprint(X)

    folder = CACHE_DIR / fold_name
    folder.mkdir(parents=True, exist_ok=True)

    name = (
        f"{split_name}_{map_name}_d{dimension}_r{reps}"
        f"_bw{bw}_{fingerprint}.npy"
    )

    return folder / name


def compute_states(X, map_name, reps, bandwidth):
    """Compute exact statevectors for a batch of encoded observations."""

    feature_map = make_feature_map(
        map_name,
        X.shape[1],
        reps,
    )

    parameters = list(feature_map.parameters)
    states = np.empty(
        (len(X), 2 ** X.shape[1]),
        dtype=np.complex128,
    )

    for i, row in enumerate(X):
        values = np.asarray(row, dtype=float) * bandwidth

        bound = feature_map.assign_parameters(
            dict(zip(parameters, values)),
            inplace=False,
        )

        states[i] = Statevector.from_instruction(bound).data

    return states


def load_or_compute_states(
    X,
    fold_name,
    split_name,
    map_name,
    reps,
    bandwidth,
):
    path = state_cache_path(
        fold_name,
        split_name,
        map_name,
        X.shape[1],
        reps,
        bandwidth,
        X,
    )

    if path.exists():
        try:
            cached = np.load(path, allow_pickle=False)
            if cached.shape == (len(X), 2 ** X.shape[1]):
                return cached
        except (OSError, ValueError):
            pass

    states = compute_states(
        X, map_name, reps, bandwidth
    )

    temp = path.with_suffix(".tmp.npy")
    np.save(temp, states)
    temp.replace(path)

    return states


def fidelity_kernel(states_a, states_b):
    """K(x,z) = |<phi(x)|phi(z)>|^2."""

    overlaps = states_a.conj() @ states_b.T
    kernel = np.abs(overlaps) ** 2

    return np.clip(kernel.real, 0.0, 1.0)


# ============================================================
# METRICS AND DIAGNOSTICS
# ============================================================

def calculate_metrics(y_true, anomaly_scores, threshold):
    y_true = np.asarray(y_true).astype(int)
    anomaly_scores = np.asarray(anomaly_scores, dtype=float)

    if len(np.unique(y_true)) < 2:
        return {
            "auroc": np.nan,
            "auprc": np.nan,
            "tpr_at_validation_1pct_threshold": np.nan,
            "observed_fpr": np.nan,
        }

    normal = y_true == 0
    attack = y_true == 1

    return {
        "auroc": float(
            roc_auc_score(y_true, anomaly_scores)
        ),
        "auprc": float(
            average_precision_score(y_true, anomaly_scores)
        ),
        "tpr_at_validation_1pct_threshold": float(
            np.mean(anomaly_scores[attack] >= threshold)
        ),
        "observed_fpr": float(
            np.mean(anomaly_scores[normal] >= threshold)
        ),
    }


def threshold_from_validation(y_validation, validation_scores):
    """Choose a threshold using normal validation observations only."""

    y_validation = np.asarray(y_validation).astype(int)
    validation_scores = np.asarray(validation_scores, dtype=float)

    normal_scores = validation_scores[y_validation == 0]

    if len(normal_scores) == 0:
        raise ValueError("Validation set has no normal observations.")

    return float(np.quantile(normal_scores, 0.99))


def kernel_diagnostics(K, y):
    """Diagnostics computed on the sampled validation kernel."""

    K = np.asarray(K, dtype=float)
    y = np.asarray(y).astype(int)

    n = len(K)
    if n < 3:
        return {
            "kernel_offdiag_mean": np.nan,
            "kernel_offdiag_std": np.nan,
            "kernel_effective_rank": np.nan,
            "kernel_target_alignment": np.nan,
        }

    offdiag = K[~np.eye(n, dtype=bool)]

    # Effective rank of the centered kernel spectrum.
    H = np.eye(n) - np.ones((n, n)) / n
    K_centered = H @ K @ H
    eigenvalues = np.linalg.eigvalsh(K_centered)
    eigenvalues = np.clip(eigenvalues, 0, None)

    total = eigenvalues.sum()
    if total > 0:
        probabilities = eigenvalues / total
        nonzero = probabilities > 0
        effective_rank = float(
            np.exp(
                -np.sum(
                    probabilities[nonzero]
                    * np.log(probabilities[nonzero])
                )
            )
        )
    else:
        effective_rank = 0.0

    # Binary target kernel; center both kernels before alignment.
    target = np.where(y == 1, 1.0, -1.0)
    target_kernel = np.outer(target, target)

    Kc = K_centered
    Tc = H @ target_kernel @ H

    denom = np.linalg.norm(Kc, "fro") * np.linalg.norm(Tc, "fro")
    alignment = (
        float(np.sum(Kc * Tc) / denom)
        if denom > 0 else np.nan
    )

    return {
        "kernel_offdiag_mean": float(offdiag.mean()),
        "kernel_offdiag_std": float(offdiag.std()),
        "kernel_effective_rank": effective_rank,
        "kernel_target_alignment": alignment,
    }


# ============================================================
# ONE-CLASS SVM SCORING AND VALIDATION TUNING
# ============================================================

def tune_precomputed_ocsvm(K_train, K_validation, y_validation):
    best = None

    for nu in NU_VALUES:
        model = OneClassSVM(
            kernel="precomputed",
            nu=nu,
        )
        model.fit(K_train)

        # Larger values mean more normal; negate for anomaly scores.
        scores = -model.decision_function(K_validation).ravel()

        if len(np.unique(y_validation)) < 2:
            continue

        auc = roc_auc_score(y_validation, scores)

        if best is None or auc > best["validation_auroc"]:
            best = {
                "model": model,
                "nu": nu,
                "validation_scores": scores,
                "validation_auroc": float(auc),
            }

    if best is None:
        raise ValueError("Validation labels need normal and attack samples.")

    return best


def tune_rbf_ocsvm(X_train, X_validation, y_validation):
    best = None

    for gamma in RBF_GAMMAS:
        for nu in NU_VALUES:
            model = OneClassSVM(
                kernel="rbf",
                gamma=gamma,
                nu=nu,
            )
            model.fit(X_train)

            scores = -model.decision_function(X_validation).ravel()

            auc = roc_auc_score(y_validation, scores)

            if best is None or auc > best["validation_auroc"]:
                best = {
                    "model": model,
                    "gamma": gamma,
                    "nu": nu,
                    "validation_scores": scores,
                    "validation_auroc": float(auc),
                }

    return best


def tune_isolation_forest(X_train, X_validation, y_validation):
    best = None

    # IsolationForest's contamination changes the decision threshold,
    # not the ranking of anomaly scores. Tune model size/randomness instead.
    for n_estimators in (100, 200):
        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=min(256, len(X_train)),
            contamination="auto",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(X_train)

        scores = -model.score_samples(X_validation)
        auc = roc_auc_score(y_validation, scores)

        if best is None or auc > best["validation_auroc"]:
            best = {
                "model": model,
                "n_estimators": n_estimators,
                "validation_scores": scores,
                "validation_auroc": float(auc),
            }

    return best


# ============================================================
# RESULT CHECKPOINTS
# ============================================================

def result_key(row):
    return (
        row["fold"],
        row["model"],
        str(row.get("feature_map", "")),
        int(row.get("n_features", 0)),
        int(row.get("reps", 0)),
        str(row.get("bandwidth", "")),
    )


def load_completed_keys():
    if not RESULTS_CSV.exists():
        return set()

    try:
        frame = pd.read_csv(RESULTS_CSV)
        return {result_key(row) for row in frame.to_dict("records")}
    except (OSError, ValueError, KeyError):
        return set()


def append_result(row):
    """Append one result using the same ordered schema for every model."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    normalized = {column: row.get(column, np.nan) for column in RESULT_COLUMNS}
    pd.DataFrame([normalized], columns=RESULT_COLUMNS).to_csv(
        RESULTS_CSV,
        mode="a",
        header=not RESULTS_CSV.exists(),
        index=False,
    )


def append_error(fold_name, stage, error):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{
        "fold": fold_name,
        "stage": stage,
        "error": str(error),
    }]).to_csv(
        ERRORS_CSV,
        mode="a",
        header=not ERRORS_CSV.exists(),
        index=False,
    )


# ============================================================
# FOLD LOADING
# ============================================================

def load_fold(path, all_validation, full_test):
    fold_name = f"{path.parent.name}/seed_{path.stem.split('_')[-1]}"

    with np.load(path, allow_pickle=False) as data:
        X_train = data["X_train_classical"].astype(float)
        Xv = data["X_validation_classical"].astype(float)
        Xt = data["X_test_classical"].astype(float)

        Xq_train = data["X_train_quantum"].astype(float)
        Xqv = data["X_validation_quantum"].astype(float)
        Xqt = data["X_test_quantum"].astype(float)

        yv = data["y_validation"].astype(int)
        yt = data["y_test"].astype(int)

        fv = data["families_validation"].astype(str)
        ft = data["families_test"].astype(str)

    heldout_family = path.parent.name
    seed_text = path.stem.split("_")[-1]
    seed = int(seed_text)

    # Validation data contains normal traffic and non-held-out attacks.
    if not all_validation:
        v_idx = stratified_indices(
            yv,
            fv,
            MAX_VALIDATION_NORMALS,
            MAX_VALIDATION_ATTACKS_PER_FAMILY,
            seed + 10_000,
        )
    else:
        v_idx = np.arange(len(yv))

    # Guard against any manifest row that accidentally contains the
    # held-out family in validation.
    v_idx = v_idx[
        ~((yv[v_idx] == 1) & (fv[v_idx] == heldout_family))
    ]

    t_idx = select_test_indices(
        yt,
        ft,
        seed + 20_000,
        full_test,
    )

    return {
        "fold": fold_name,
        "family": heldout_family,
        "seed": seed,
        "X_train": X_train,
        "X_validation": Xv[v_idx],
        "X_test": Xt[t_idx],
        "Xq_train": Xq_train,
        "Xq_validation": Xqv[v_idx],
        "Xq_test": Xqt[t_idx],
        "y_validation": yv[v_idx],
        "y_test": yt[t_idx],
        "families_validation": fv[v_idx],
        "families_test": ft[t_idx],
        "validation_indices": v_idx,
        "test_indices": t_idx,
        "test_sampling": "full" if full_test else "sampled",
    }


# ============================================================
# CLASSICAL MODELS
# ============================================================

def run_classical_models(fold, completed):
    """Run classical baselines on the same PCA dimensions as quantum models."""
    yv = fold["y_validation"]
    yt = fold["y_test"]

    for n_features in FEATURE_DIMENSIONS:
        if n_features > fold["X_train"].shape[1]:
            continue

        X_train = fold["X_train"][:, :n_features]
        X_validation = fold["X_validation"][:, :n_features]
        X_test = fold["X_test"][:, :n_features]

        common = {
            "fold": fold["fold"],
            "heldout_family": fold["family"],
            "seed": fold["seed"],
            "n_train": len(X_train),
            "n_validation": len(yv),
            "n_test": len(yt),
            "test_sampling": fold["test_sampling"],
            "feature_map": "",
            "n_features": n_features,
            "reps": 0,
            "bandwidth": 0.0,
        }

        for model_name in ("RBF_OCSVM", "IsolationForest"):
            row = {**common, "model": model_name}
            if result_key(row) in completed:
                continue

            try:
                start = time.time()
                if model_name == "RBF_OCSVM":
                    best = tune_rbf_ocsvm(X_train, X_validation, yv)
                else:
                    best = tune_isolation_forest(X_train, X_validation, yv)

                validation_scores = best["validation_scores"]
                threshold = threshold_from_validation(yv, validation_scores)
                if model_name == "RBF_OCSVM":
                    test_scores = -best["model"].decision_function(X_test).ravel()
                else:
                    test_scores = -best["model"].score_samples(X_test)

                metrics = calculate_metrics(yt, test_scores, threshold)
                tuning_details = {
                    key: value for key, value in best.items()
                    if key not in ("model", "validation_scores")
                }
                row.update({
                    "validation_auroc": best["validation_auroc"],
                    "validation_threshold": threshold,
                    "best_nu": best.get("nu", np.nan),
                    "tuning_details": json.dumps(tuning_details),
                    "runtime_seconds": time.time() - start,
                    **metrics,
                })
                append_result(row)
                completed.add(result_key(row))

            except Exception as exc:
                stage = f"{model_name}_d{n_features}"
                append_error(fold["fold"], stage, exc)
                warnings.warn(f"{fold['fold']} {stage}: {exc}")


# ============================================================
# QUANTUM MODELS
# ============================================================

def run_quantum_models(fold, completed):
    yv = fold["y_validation"]
    yt = fold["y_test"]

    for n_features in FEATURE_DIMENSIONS:
        if n_features > fold["Xq_train"].shape[1]:
            continue

        X_train = fold["Xq_train"][:, :n_features]
        X_validation = fold["Xq_validation"][:, :n_features]
        X_test = fold["Xq_test"][:, :n_features]

        for map_name in ("Z", "ZZ", "Pauli"):
            for reps in REPETITIONS:
                for bandwidth in BANDWIDTHS:
                    row = {
                        "fold": fold["fold"],
                        "heldout_family": fold["family"],
                        "seed": fold["seed"],
                        "n_train": len(X_train),
                        "n_validation": len(yv),
                        "n_test": len(yt),
                        "test_sampling": fold["test_sampling"],
                        "model": "Quantum_OCSVM",
                        "feature_map": map_name,
                        "n_features": n_features,
                        "reps": reps,
                        "bandwidth": bandwidth,
                    }

                    key = result_key(row)
                    if key in completed:
                        continue

                    try:
                        start = time.time()

                        S_train = load_or_compute_states(
                            X_train, fold["fold"], "train",
                            map_name, reps, bandwidth,
                        )
                        S_val = load_or_compute_states(
                            X_validation, fold["fold"], "validation",
                            map_name, reps, bandwidth,
                        )
                        S_test = load_or_compute_states(
                            X_test, fold["fold"], "test",
                            map_name, reps, bandwidth,
                        )

                        K_train = fidelity_kernel(S_train, S_train)
                        K_train = (K_train + K_train.T) / 2.0
                        np.fill_diagonal(K_train, 1.0)

                        K_val_train = fidelity_kernel(S_val, S_train)
                        K_test_train = fidelity_kernel(S_test, S_train)

                        # Tune nu using validation labels only.
                        best = tune_precomputed_ocsvm(
                            K_train,
                            K_val_train,
                            yv,
                        )

                        threshold = threshold_from_validation(
                            yv,
                            best["validation_scores"],
                        )

                        test_scores = -best["model"].decision_function(
                            K_test_train
                        ).ravel()

                        metrics = calculate_metrics(
                            yt,
                            test_scores,
                            threshold,
                        )

                        # Kernel-target alignment on a reproducible
                        # subset of validation states.
                        diag_rng = np.random.default_rng(
                            fold["seed"] + n_features
                        )
                        diag_n = min(500, len(S_val))
                        diag_idx = np.sort(
                            diag_rng.choice(
                                len(S_val),
                                size=diag_n,
                                replace=False,
                            )
                        )
                        K_diag = fidelity_kernel(
                            S_val[diag_idx],
                            S_val[diag_idx],
                        )
                        diagnostics = kernel_diagnostics(
                            K_diag, yv[diag_idx]
                        )

                        row.update({
                            "validation_auroc": best["validation_auroc"],
                            "validation_threshold": threshold,
                            "best_nu": best["nu"],
                            "tuning_details": json.dumps({"nu_candidates": NU_VALUES}),
                            "runtime_seconds": time.time() - start,
                            **metrics,
                            **diagnostics,
                        })

                        append_result(row)
                        completed.add(key)

                        print(
                            f"  {map_name:5s} d={n_features} "
                            f"r={reps} bw={bandwidth}: "
                            f"test AUROC={metrics['auroc']:.4f}"
                        )

                    except Exception as exc:
                        append_error(
                            fold["fold"],
                            f"{map_name}_d{n_features}_r{reps}_bw{bandwidth}",
                            exc,
                        )
                        warnings.warn(
                            f"{fold['fold']} {map_name} "
                            f"d={n_features}, r={reps}, "
                            f"bw={bandwidth}: {exc}"
                        )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Quantum-kernel one-class anomaly detection experiment"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Archive prior result/error CSVs and recompute all result rows (cached states may be reused).",
    )
    parser.add_argument(
        "--full-test",
        action="store_true",
        help="Use every held-out test row; substantially slower.",
    )
    parser.add_argument(
        "--all-validation",
        action="store_true",
        help="Use the full validation set; substantially slower.",
    )
    parser.add_argument(
        "--classical-only",
        action="store_true",
        help="Run only classical baselines.",
    )
    parser.add_argument(
        "--quantum-only",
        action="store_true",
        help="Run only quantum-kernel models.",
    )
    args = parser.parse_args()

    # Keep sampled-test and full-test results separate.
    global RESULTS_CSV, ERRORS_CSV

    if args.full_test:
        RESULTS_CSV = RESULTS_DIR / "experiment_results_full_test.csv"
        ERRORS_CSV = RESULTS_DIR / "experiment_errors_full_test.csv"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Results file: {RESULTS_CSV}")
    print(f"Errors file: {ERRORS_CSV}")

    if args.force:
        # Preserve the previous run, then start fresh so --force never duplicates rows.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        for old_path in (RESULTS_CSV, ERRORS_CSV):
            if old_path.exists():
                backup_path = old_path.with_name(
                    f"{old_path.stem}_backup_{stamp}{old_path.suffix}"
                )
                old_path.replace(backup_path)
                print(f"Archived previous output: {backup_path.name}")

    paths = sorted(FOLDS_DIR.glob("*/*.npz"))

    if not paths:
        raise FileNotFoundError(
            f"No fold files found under {FOLDS_DIR}"
        )

    completed = set() if args.force else load_completed_keys()

    print("=" * 72)
    print("QUANTUM-KERNEL INTRUSION DETECTION EXPERIMENT")
    print("=" * 72)
    print(f"Folds found: {len(paths)}")
    print(f"Completed result keys: {len(completed)}")
    print(f"Full test: {args.full_test}")
    print(f"All validation: {args.all_validation}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Results file: {RESULTS_CSV}")
    print()

    failures = 0

    for i, path in enumerate(paths, start=1):
        fold_name = f"{path.parent.name}/{path.stem}"
        print(f"[{i}/{len(paths)}] {fold_name}")

        try:
            fold = load_fold(
                path,
                all_validation=args.all_validation,
                full_test=args.full_test,
            )

            if not args.quantum_only:
                run_classical_models(fold, completed)

            if not args.classical_only:
                run_quantum_models(fold, completed)

            print(
                f"  Validation rows: {len(fold['y_validation'])}; "
                f"test rows: {len(fold['y_test'])}"
            )

        except Exception as exc:
            failures += 1
            append_error(fold_name, "fold_setup", exc)
            print(f"  FOLD FAILED: {exc}")

    print()
    print("=" * 72)
    print("EXPERIMENT RUN FINISHED")
    print("=" * 72)
    print(f"Folds discovered: {len(paths)}")
    print(f"Fold setup failures: {failures}")
    print(f"Results: {RESULTS_CSV}")
    print(f"Errors: {ERRORS_CSV}")
    print()
    print("The run can be resumed by running the same command again.")


if __name__ == "__main__":
    main()
