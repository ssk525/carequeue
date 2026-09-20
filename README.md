# CareQueue

A research-only, end-to-end machine learning project for hospital
readmission risk estimation and capacity-constrained discharge follow-up.

## Problem

A care coordinator has limited follow-up capacity. CareQueue estimates
readmission risk at discharge and ranks eligible encounters for a
fixed-size follow-up queue.

Predicted risk does not measure the causal benefit of follow-up.

## Dataset

UCI Diabetes 130-US Hospitals for Years 1999–2008.

- Dataset: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- Associated study: https://doi.org/10.1155/2014/781670

Review the dataset's current reuse terms and citation requirements
before redistributing it. Raw data and model artifacts are not committed.

## Quick start

Use Python 3.11+ (validated locally on 3.12).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest -q
python -m carequeue.train --download
uvicorn carequeue.api:app --host 127.0.0.1 --port 8000
```

In another activated terminal:

```bash
python -m streamlit run carequeue/dashboard.py
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

For a pinned environment after validation:

```bash
python -m pip install -r requirements.lock.txt
```

## Method

Patients are disjoint across training, model selection, probability
calibration, and testing (~60% / 10% / 10% / 20% by patient groups).

Candidate models are logistic regression and histogram gradient boosting.
Selection uses average precision. Calibration is fitted on a separate
partition. Test results include calibration, capacity-based evaluation,
patient-cluster bootstrap uncertainty, and subgroup summaries.

Race and gender are excluded from predictors but audited on the test set.
Death/hospice discharge dispositions are excluded from the eligible cohort.

The split is not chronological — the released data do not support a clean
deployment-time simulation.

## Outputs

Training writes `artifacts/model.joblib` and `artifacts/metrics.json`.

The API exposes `/health`, `/model-info`, `/predict`, `/triage`, and `/docs`.

Offline monitoring checks numeric input drift and missingness.
It does not establish model performance or clinical safety.

## Results

Verified after local training on the UCI archive
(dataset SHA-256 prefix `0689e7ec…`, seed `42`).

| Metric | Prevalence baseline | Selected model (calibrated) |
|---|---:|---:|
| Eligible encounters | — | 99,343 (2,423 excluded) |
| Selected model | — | histogram gradient boosting |
| Test encounters | 19,821 | 19,821 |
| Positive rate | 11.1% | 11.1% |
| Average precision | 0.111 | **0.218** |
| ROC-AUC | 0.500 | 0.662 |
| Brier score | 0.099 | 0.095 |
| Precision @ 10% capacity | — | 0.261 |
| Recall @ 10% capacity | — | 0.235 |
| Lift @ 10% capacity | — | **2.35×** |

Patient-cluster bootstrap 95% interval for test average precision:
**0.198 – 0.239** (200 valid resamples).

Selection-set average precision: logistic `0.214`, gradient boosting `0.228`.

Calibration changed Brier score only marginally on this run
(uncalibrated AP matched calibrated AP at `0.218`). Both are reported in
`artifacts/metrics.json` so calibration can be discussed honestly.

These numbers are retrospective research results on historical data.
They are not evidence of clinical effectiveness or transportability to a
present-day hospital.

## Limitations and intended use

This project uses historical observational data.

- Readmissions outside the source data may not be captured.
- Administrative codes and subgroup labels have limitations.
- Removing protected attributes does not establish fairness.
- A high-risk prediction does not prove that an intervention will help.
- The software is not validated for clinical use.

Use synthetic inputs in public demonstrations.
Do not upload identifiable patient data.

## Reproducibility

Training records the dataset checksum, random seed, split counts,
Python version, and scikit-learn version.

`requirements.lock.txt` captures a validated local environment.
CI installs from `requirements.txt` ranges for compatibility.

Only load model artifacts that you trust. Joblib artifacts can execute code.

## Project layout

```
carequeue/
├── carequeue/          # core, train, api, dashboard, monitor
├── tests/              # synthetic contract tests
├── .github/workflows/  # CI
├── Dockerfile
├── requirements.txt
└── requirements.lock.txt
```
