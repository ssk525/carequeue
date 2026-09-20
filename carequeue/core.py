from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42

# Input bounds for the API / demos (not clinical thresholds).
NUMERIC_BOUNDS = {
    "time_in_hospital": (1, 60),
    "num_lab_procedures": (0, 500),
    "num_procedures": (0, 100),
    "num_medications": (0, 300),
    "number_outpatient": (0, 100),
    "number_emergency": (0, 100),
    "number_inpatient": (0, 100),
    "number_diagnoses": (1, 100),
}

CATEGORY_VALUES = {
    "age": [f"[{n}-{n + 10})" for n in range(0, 100, 10)],
    "A1Cresult": ["None", "Norm", ">7", ">8"],
    "max_glu_serum": ["None", "Norm", ">200", ">300"],
    "insulin": ["No", "Down", "Steady", "Up"],
    "change": ["No", "Ch"],
    "diabetesMed": ["No", "Yes"],
}

ID_CATEGORIES = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

NUMERIC = list(NUMERIC_BOUNDS)
CATEGORICAL = ["age", *ID_CATEGORIES, *list(CATEGORY_VALUES)[1:]]
FEATURES = NUMERIC + CATEGORICAL

# Death / hospice dispositions (UCI discharge_disposition_id).
EXCLUDED_DISPOSITIONS = {"11", "13", "14", "19", "20", "21"}


def read_csv(path) -> pd.DataFrame:
    # Keep literal "None" lab categories; treat "?" / blanks as missing.
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_values=["?", ""],
    )


def prepare_cohort(raw: pd.DataFrame) -> pd.DataFrame:
    required = set(FEATURES + ["patient_nbr", "encounter_id", "readmitted"])
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    if raw["encounter_id"].duplicated().any():
        raise ValueError("Duplicate encounter identifiers found.")

    if raw["patient_nbr"].isna().any():
        raise ValueError("Patient identifiers are required for grouped splits.")

    if not raw["readmitted"].isin(["NO", ">30", "<30"]).all():
        raise ValueError("Unexpected or missing readmission labels.")

    disposition = raw["discharge_disposition_id"].astype(str)
    eligible = disposition.notna() & ~disposition.isin(EXCLUDED_DISPOSITIONS)
    eligible &= raw["discharge_disposition_id"].notna()
    frame = raw.loc[eligible].copy()
    frame["target"] = (frame["readmitted"] == "<30").astype(int)

    if frame.empty:
        raise ValueError("The eligible cohort is empty.")

    clean_features(frame)
    return frame.reset_index(drop=True)


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    x = frame[FEATURES].copy()

    for name, (low, high) in NUMERIC_BOUNDS.items():
        original = x[name]
        numeric = pd.to_numeric(original, errors="coerce")
        if (original.notna() & numeric.isna()).any():
            raise ValueError(f"{name} must be numeric.")

        if numeric.notna().any():
            if not numeric.dropna().between(low, high).all():
                raise ValueError(f"{name} must be between {low} and {high}.")
            if (numeric.dropna() % 1 != 0).any():
                raise ValueError(f"{name} must contain whole numbers.")

        x[name] = numeric

    for name in CATEGORICAL:
        x[name] = x[name].map(
            lambda value: str(value) if pd.notna(value) else np.nan
        )

    for name, allowed in CATEGORY_VALUES.items():
        values = x[name].dropna()
        if not values.isin(allowed).all():
            raise ValueError(f"Unexpected category in {name}.")

    for name in ID_CATEGORIES:
        values = x[name].dropna()
        if not values.str.fullmatch(r"\d{1,2}").all():
            raise ValueError(f"{name} must be a one- or two-digit code.")

    return x


def split_patients(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def split(part, test_size, seed):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed,
        )
        left, right = next(
            splitter.split(part, groups=part["patient_nbr"])
        )
        return part.iloc[left].copy(), part.iloc[right].copy()

    development, test = split(frame, 0.20, SEED)
    train, holdout = split(development, 0.25, SEED + 1)
    selection, calibration = split(holdout, 0.50, SEED + 2)

    parts = {
        "train": train,
        "selection": selection,
        "calibration": calibration,
        "test": test,
    }

    seen = set()
    for name, part in parts.items():
        groups = set(part["patient_nbr"])
        if seen.intersection(groups):
            raise AssertionError("Patient leakage detected.")
        seen.update(groups)
        if part["target"].nunique() != 2:
            raise ValueError(f"{name} does not contain both target classes.")

    return parts


def make_model(kind: str) -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encode",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float32,
            ),
        ),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])

    if kind == "logistic":
        estimator = LogisticRegression(
            max_iter=2000,
            random_state=SEED,
        )
    elif kind == "gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=150,
            max_leaf_nodes=15,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=SEED,
        )
    else:
        raise ValueError(f"Unknown model: {kind}")

    return Pipeline([
        ("preprocess", preprocess),
        ("model", estimator),
    ])


def logit_scores(probabilities) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_calibrator(model, x, y):
    calibrator = LogisticRegression(C=1e6, max_iter=1000)
    scores = logit_scores(model.predict_proba(x)[:, 1])
    calibrator.fit(scores, y)
    return calibrator


def predict_risk(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    x = clean_features(frame)
    raw = bundle["model"].predict_proba(x)[:, 1]
    return bundle["calibrator"].predict_proba(logit_scores(raw))[:, 1]


def top_k_indices(probabilities, capacity: int) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError("Probabilities must be a finite one-dimensional array.")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be between zero and one.")
    if not 0 <= capacity <= len(p):
        raise ValueError("Capacity must be between zero and batch size.")

    # Stable sort so tied scores keep input order.
    return np.argsort(-p, kind="stable")[:capacity]


def evaluate(y, p, fraction=0.10) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if not len(y):
        raise ValueError("Cannot evaluate an empty sample.")

    k = max(1, int(np.ceil(len(y) * fraction)))
    selected = top_k_indices(p, k)
    prevalence = float(y.mean())
    positives = int(y.sum())
    captured = int(y[selected].sum())
    two_classes = len(np.unique(y)) == 2

    return {
        "encounters": int(len(y)),
        "positive_rate": prevalence,
        "average_precision": (
            float(average_precision_score(y, p)) if two_classes else None
        ),
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else None,
        "brier_score": float(brier_score_loss(y, p)),
        "queue_fraction": fraction,
        "queue_size": k,
        "precision_at_capacity": float(y[selected].mean()),
        "recall_at_capacity": captured / positives if positives else None,
        "lift_at_capacity": (
            float(y[selected].mean() / prevalence) if prevalence else None
        ),
    }


def group_bootstrap_ap(y, p, groups, repeats=200) -> dict:
    y = np.asarray(y)
    p = np.asarray(p)
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    blocks = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(SEED)
    scores = []

    for _ in range(repeats):
        sampled = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[i] for i in sampled])
        if len(np.unique(y[idx])) == 2:
            scores.append(average_precision_score(y[idx], p[idx]))

    if not scores:
        return {"lower": None, "upper": None, "valid_resamples": 0}

    low, high = np.quantile(scores, [0.025, 0.975])
    return {
        "lower": float(low),
        "upper": float(high),
        "valid_resamples": len(scores),
    }


def numeric_reference(x: pd.DataFrame) -> dict:
    result = {}
    for name in NUMERIC:
        values = x[name].dropna().to_numpy(dtype=float)
        if not len(values):
            raise ValueError(f"No observed training values for {name}.")

        cuts = np.unique(
            np.quantile(values, np.arange(0.1, 1.0, 0.1))
        )
        counts, _ = np.histogram(values, bins=np.r_[-np.inf, cuts, np.inf])

        result[name] = {
            "cuts": cuts.tolist(),
            "proportions": (counts / counts.sum()).tolist(),
            "missing_rate": float(x[name].isna().mean()),
        }
    return result


def drift_report(reference: dict, x: pd.DataFrame) -> dict:
    report = {}
    for name, profile in reference.items():
        values = x[name].dropna().to_numpy(dtype=float)
        psi = None
        if len(values):
            bins = np.r_[-np.inf, profile["cuts"], np.inf]
            counts, _ = np.histogram(values, bins=bins)
            observed = counts / counts.sum()
            expected = np.asarray(profile["proportions"], dtype=float)

            observed = np.clip(observed, 1e-6, None)
            expected = np.clip(expected, 1e-6, None)
            observed /= observed.sum()
            expected /= expected.sum()
            psi = float(
                np.sum((observed - expected) * np.log(observed / expected))
            )

        missing = float(x[name].isna().mean())
        report[name] = {
            "psi": psi,
            "missing_rate": missing,
            "missing_rate_change": missing - profile["missing_rate"],
        }
    return report
