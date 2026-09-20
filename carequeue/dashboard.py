import json
import os
from pathlib import Path

import requests
import streamlit as st

from carequeue.core import (
    CATEGORY_VALUES,
    ID_CATEGORIES,
    NUMERIC_BOUNDS,
)

st.set_page_config(page_title="CareQueue", layout="wide")
st.title("CareQueue")
st.caption("Discharge readmission risk scoring and follow-up queue demo")
st.info("Synthetic inputs only — not for clinical decisions.")

api_url = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

defaults = {
    "time_in_hospital": 4,
    "num_lab_procedures": 40,
    "num_procedures": 1,
    "num_medications": 15,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 1,
    "number_diagnoses": 7,
}

with st.form("encounter"):
    left, right = st.columns(2)
    record = {}

    with left:
        st.subheader("Encounter and utilization")
        for name, (low, high) in NUMERIC_BOUNDS.items():
            record[name] = int(st.number_input(
                name,
                min_value=low,
                max_value=high,
                value=defaults[name],
                step=1,
            ))

    with right:
        st.subheader("Discharge-time categories")
        for name, choices in CATEGORY_VALUES.items():
            default_index = 6 if name == "age" else 0
            record[name] = st.selectbox(
                name,
                choices,
                index=default_index,
            )

        for name in ID_CATEGORIES:
            record[name] = st.text_input(name, value="1")

        st.caption("Admission / discharge IDs use the UCI code mapping.")

    submitted = st.form_submit_button("Score encounter")

if submitted:
    try:
        response = requests.post(
            f"{api_url}/predict",
            json=record,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        st.metric(
            "30-day readmission probability",
            f"{result['readmission_probability']:.1%}",
        )
        st.caption(f"Model version: {result['model_version']}")
    except requests.HTTPError as exc:
        st.error(f"API rejected the request: {exc.response.text}")
    except requests.RequestException:
        st.error("API unreachable — start uvicorn first.")

metrics_path = Path("artifacts/metrics.json")
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.subheader("Held-out test results")

    test = metrics["test"]
    columns = st.columns(3)
    columns[0].metric("Average precision", f"{test['average_precision']:.3f}")
    columns[1].metric("Brier score", f"{test['brier_score']:.3f}")
    columns[2].metric("Lift at 10% capacity", f"{test['lift_at_capacity']:.2f}×")

    with st.expander("Evaluation details and subgroup audit"):
        st.json(metrics)
