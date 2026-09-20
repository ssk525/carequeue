import argparse
import json
from pathlib import Path

import joblib

from carequeue.core import clean_features, drift_report, read_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/model.joblib"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/drift.json"),
    )
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    batch = read_csv(args.input)
    if batch.empty:
        parser.error("Monitoring batch is empty.")

    x = clean_features(batch)
    report = {
        "model_version": bundle["version"],
        "batch_rows": len(batch),
        "scope": "numeric features + missingness",
        "features": drift_report(bundle["numeric_reference"], x),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
