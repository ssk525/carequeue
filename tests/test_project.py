import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from carequeue.api import create_app
from carequeue.core import (
    FEATURES,
    clean_features,
    drift_report,
    numeric_reference,
    prepare_cohort,
    read_csv,
    split_patients,
    top_k_indices,
)
from carequeue.train import train


def example_record():
    return {
        "time_in_hospital": 4,
        "num_lab_procedures": 40,
        "num_procedures": 1,
        "num_medications": 15,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 1,
        "number_diagnoses": 7,
        "age": "[60-70)",
        "admission_type_id": "1",
        "discharge_disposition_id": "1",
        "admission_source_id": "1",
        "A1Cresult": "None",
        "max_glu_serum": "None",
        "insulin": "Steady",
        "change": "No",
        "diabetesMed": "Yes",
    }


def synthetic_dataset(patients=80):
    records = []
    for patient in range(patients):
        # Both outcomes per synthetic patient ensure both classes in
        # every group partition. This is a software fixture, not biology.
        for visit in range(2):
            record = example_record()
            record.update({
                "patient_nbr": str(patient),
                "encounter_id": str(patient * 2 + visit),
                "readmitted": "<30" if visit else "NO",
                "number_inpatient": visit,
                "num_lab_procedures": 20 + patient % 30,
                "race": "SyntheticGroup",
                "gender": "Female" if patient % 2 else "Male",
            })
            records.append(record)
    return pd.DataFrame(records)


def test_literal_none_is_preserved(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame([example_record()]).to_csv(path, index=False)
    assert read_csv(path).loc[0, "A1Cresult"] == "None"


def test_feature_contract_excludes_identifiers_and_target():
    prohibited = {"patient_nbr", "encounter_id", "readmitted", "target"}
    assert not prohibited.intersection(FEATURES)
    assert "race" not in FEATURES
    assert "gender" not in FEATURES


def test_patient_splits_do_not_overlap():
    cohort = prepare_cohort(synthetic_dataset())
    parts = split_patients(cohort)
    groups = [set(part["patient_nbr"]) for part in parts.values()]

    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            assert groups[left].isdisjoint(groups[right])


def test_invalid_numeric_value_is_rejected():
    record = example_record()
    record["num_medications"] = -1
    with pytest.raises(ValueError):
        clean_features(pd.DataFrame([record]))


def test_missing_feature_is_rejected():
    record = example_record()
    del record["age"]
    with pytest.raises(ValueError):
        clean_features(pd.DataFrame([record]))


def test_queue_is_capacity_limited_and_stable():
    assert top_k_indices([0.2, 0.9, 0.9], 2).tolist() == [1, 2]
    assert top_k_indices([0.2, 0.9], 0).tolist() == []
    with pytest.raises(ValueError):
        top_k_indices([0.2], 2)


def test_identical_numeric_distribution_has_zero_drift():
    x = clean_features(synthetic_dataset())
    report = drift_report(numeric_reference(x), x)
    assert all(abs(row["psi"]) < 1e-10 for row in report.values())


@pytest.fixture(scope="module")
def trained_artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("training")
    data = root / "synthetic.csv"
    output = root / "artifacts"
    synthetic_dataset().to_csv(data, index=False)

    metrics = train(data, output)
    assert np.isfinite(metrics["test"]["brier_score"])
    assert (output / "metrics.json").exists()
    return output / "model.joblib"


def test_train_save_load_and_api(trained_artifact):
    with TestClient(create_app(str(trained_artifact))) as client:
        assert client.get("/health").status_code == 200

        response = client.post("/predict", json=example_record())
        assert response.status_code == 200
        probability = response.json()["readmission_probability"]
        assert 0 <= probability <= 1

        batch = {
            "records": [example_record(), example_record()],
            "capacity": 1,
        }
        queue = client.post("/triage", json=batch)
        assert queue.status_code == 200
        assert len(queue.json()["selected"]) == 1

        invalid = example_record()
        invalid["readmitted"] = "<30"
        assert client.post("/predict", json=invalid).status_code == 422

        excluded = example_record()
        excluded["discharge_disposition_id"] = "11"
        assert client.post("/predict", json=excluded).status_code == 422

        batch["capacity"] = 3
        assert client.post("/triage", json=batch).status_code == 422
