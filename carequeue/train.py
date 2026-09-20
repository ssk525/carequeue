from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
import requests
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from carequeue.core import (
    FEATURES,
    SEED,
    clean_features,
    evaluate,
    fit_calibrator,
    group_bootstrap_ap,
    make_model,
    numeric_reference,
    predict_risk,
    prepare_cohort,
    read_csv,
    split_patients,
    top_k_indices,
)

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)


def download_dataset(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    total = 0
    limit = 100 * 1024 * 1024

    with requests.get(
        DATASET_URL,
        timeout=(10, 120),
        stream=True,
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            total += len(chunk)
            if total > limit:
                raise ValueError("Archive exceeds the download size limit.")
            chunks.append(chunk)

    with ZipFile(io.BytesIO(b"".join(chunks))) as archive:
        matches = [
            name
            for name in archive.namelist()
            if Path(name).name == "diabetic_data.csv"
        ]
        if len(matches) != 1:
            raise ValueError("Expected exactly one diabetic_data.csv.")

        member = archive.getinfo(matches[0])
        if member.file_size > limit:
            raise ValueError("Dataset exceeds the extracted size limit.")

        # Read only the expected member; do not extract arbitrary paths.
        data = archive.read(member)

    destination.write_bytes(data)


def subgroup_report(test: pd.DataFrame, probabilities) -> list[dict]:
    y = test["target"].to_numpy()
    p = np.asarray(probabilities)
    selected = np.zeros(len(test), dtype=bool)
    selected[top_k_indices(p, max(1, int(np.ceil(0.10 * len(test)))))] = True

    rows = []
    for attribute in ["race", "gender", "age"]:
        if attribute not in test:
            continue

        values = test[attribute].fillna("Missing").astype(str).to_numpy()
        for group in sorted(np.unique(values)):
            mask = values == group
            n = int(mask.sum())
            positives = int(y[mask].sum())
            negatives = n - positives
            enough = n >= 100 and positives >= 10 and negatives >= 10
            captured = int(y[mask & selected].sum())

            rows.append({
                "attribute": attribute,
                "group": group,
                "encounters": n,
                "patients": int(test.loc[mask, "patient_nbr"].nunique()),
                "positive_count": positives,
                "positive_rate": float(y[mask].mean()),
                "selection_rate_global_queue": float(selected[mask].mean()),
                "recall_global_queue": (
                    captured / positives if enough else None
                ),
                "average_precision": (
                    float(average_precision_score(y[mask], p[mask]))
                    if enough else None
                ),
                "performance_suppressed_small_sample": not enough,
            })
    return rows


def reliability_bins(y, probabilities) -> list[dict]:
    y = np.asarray(y)
    p = np.asarray(probabilities)
    bin_ids = np.minimum((p * 10).astype(int), 9)
    rows = []

    for bin_id in range(10):
        mask = bin_ids == bin_id
        if mask.any():
            rows.append({
                "bin": bin_id,
                "count": int(mask.sum()),
                "mean_prediction": float(p[mask].mean()),
                "observed_rate": float(y[mask].mean()),
            })
    return rows


def train(data_path: Path, output: Path) -> dict:
    raw = read_csv(data_path)
    cohort = prepare_cohort(raw)
    parts = split_patients(cohort)
    x = {name: clean_features(part) for name, part in parts.items()}
    y = {name: part["target"].to_numpy() for name, part in parts.items()}

    models = {}
    selection_scores = {}
    for name in ["logistic", "gradient_boosting"]:
        model = make_model(name)
        model.fit(x["train"], y["train"])
        predictions = model.predict_proba(x["selection"])[:, 1]
        selection_scores[name] = float(
            average_precision_score(y["selection"], predictions)
        )
        models[name] = model

    winner = max(selection_scores, key=selection_scores.get)
    model = models[winner]

    # Do not retrain on calibration or test data.
    calibrator = fit_calibrator(
        model,
        x["calibration"],
        y["calibration"],
    )

    checksum = hashlib.sha256(data_path.read_bytes()).hexdigest()
    created = datetime.now(timezone.utc).isoformat()
    version = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + checksum[:8]
    )

    bundle = {
        "model": model,
        "calibrator": calibrator,
        "feature_names": FEATURES,
        "numeric_reference": numeric_reference(x["train"]),
        "version": version,
    }

    predictions = predict_risk(bundle, parts["test"])
    raw_predictions = model.predict_proba(x["test"])[:, 1]
    baseline = np.full(len(parts["test"]), y["train"].mean())

    sample = parts["selection"].sample(
        n=min(4000, len(parts["selection"])),
        random_state=SEED,
    )
    importance = permutation_importance(
        model,
        clean_features(sample),
        sample["target"],
        scoring="average_precision",
        n_repeats=3,
        random_state=SEED,
        n_jobs=1,
    )

    ranking = sorted(
        [
            {
                "feature": name,
                "mean_ap_decrease": float(mean),
                "std_ap_decrease": float(std),
            }
            for name, mean, std in zip(
                FEATURES,
                importance.importances_mean,
                importance.importances_std,
            )
        ],
        key=lambda row: row["mean_ap_decrease"],
        reverse=True,
    )

    baseline_metrics = evaluate(y["test"], baseline)
    metrics = {
        "model_version": version,
        "created_utc": created,
        "dataset_sha256": checksum,
        "seed": SEED,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "raw_encounters": int(len(raw)),
        "eligible_encounters": int(len(cohort)),
        "excluded_encounters": int(len(raw) - len(cohort)),
        "split_counts": {
            name: {
                "encounters": int(len(part)),
                "patients": int(part["patient_nbr"].nunique()),
                "positive_rate": float(part["target"].mean()),
            }
            for name, part in parts.items()
        },
        "selected_model": winner,
        "selection_average_precision": selection_scores,
        "test": evaluate(y["test"], predictions),
        "uncalibrated_test": evaluate(y["test"], raw_predictions),
        # Queue metrics are omitted for this constant-score baseline:
        # arbitrary tie ordering would make them misleading.
        "prevalence_baseline_test": {
            key: baseline_metrics[key]
            for key in ["average_precision", "roc_auc", "brier_score"]
        },
        "test_ap_patient_bootstrap_95_interval": group_bootstrap_ap(
            y["test"],
            predictions,
            parts["test"]["patient_nbr"],
        ),
        "reliability_bins": reliability_bins(y["test"], predictions),
        "subgroups": subgroup_report(parts["test"], predictions),
        "validation_permutation_importance": ranking,
    }

    bundle["metadata"] = {
        "model_version": version,
        "created_utc": created,
        "dataset_sha256": checksum,
        "selected_model": winner,
        "prediction_time": "discharge",
        "intended_use": "Retrospective research demonstration only",
    }

    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output / "model.joblib")
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/diabetic_data.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts"),
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    if not args.data.exists():
        if not args.download:
            parser.error("Dataset not found. Supply --download or --data.")
        download_dataset(args.data)

    metrics = train(args.data, args.output)
    print(json.dumps({
        "selected_model": metrics["selected_model"],
        "test": metrics["test"],
        "artifacts": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
