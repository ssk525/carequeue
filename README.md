# CareQueue

Predict 30-day hospital readmission risk at discharge, then rank patients
for a fixed follow-up capacity.

Built on the UCI Diabetes 130-US Hospitals dataset (~100K encounters).
Not a clinical product — portfolio / research code only.

## Why this exists

Hospitals cannot call every discharged patient the same day. CareQueue
scores eligible encounters and returns a top-k follow-up queue given a
capacity budget.

Risk ≠ benefit of calling. This project ranks who to contact first; it
does not estimate whether follow-up prevents readmission.

## Setup

Python 3.11+ (tested on 3.12).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m carequeue.train --download
```

Pinned deps (optional): `pip install -r requirements.lock.txt`

### API

```bash
uvicorn carequeue.api:app --host 127.0.0.1 --port 8000
# docs: http://127.0.0.1:8000/docs
```

### Dashboard

```bash
streamlit run carequeue/dashboard.py
```

### Drift check (after training)

```bash
python -c "from carequeue.core import read_csv, prepare_cohort, FEATURES; \
prepare_cohort(read_csv('data/diabetic_data.csv')).sample(1000, random_state=42)[FEATURES] \
.to_csv('data/demo_batch.csv', index=False)"
python -m carequeue.monitor --input data/demo_batch.csv
```

### Docker

```bash
docker build -t carequeue .
docker run --rm -p 127.0.0.1:8000:8000 -v "$(pwd)/artifacts:/app/artifacts:ro" carequeue
```

Or use `make install test train api`.

## Approach

- Eligible cohort excludes death/hospice discharge dispositions
- Patient-level splits (no shared `patient_nbr` across train / selection /
  calibration / test)
- Models: logistic regression vs histogram gradient boosting
- Selection metric: average precision
- Separate calibration fold; held-out test never used for fitting
- Queue metrics at 10% capacity (precision, recall, lift)
- Subgroup audit on race / gender / age (attributes not used as features)
- Numeric PSI drift report against the training reference

Splits are random by patient, not chronological — the public file does
not support a clean time-based deployment simulation.

## Results

Local run on the UCI zip (seed 42, SHA-256 prefix `0689e7ec`):

| | Baseline | Model |
|---|---:|---:|
| Eligible encounters | — | 99,343 |
| Model | — | hist. gradient boosting |
| Test n | 19,821 | 19,821 |
| Positive rate | 11.1% | 11.1% |
| Average precision | 0.111 | 0.218 |
| ROC-AUC | 0.500 | 0.662 |
| Brier | 0.099 | 0.095 |
| Precision @ 10% | — | 0.261 |
| Recall @ 10% | — | 0.235 |
| Lift @ 10% | — | 2.35× |

Bootstrap 95% CI for test AP (patient blocks): 0.198 – 0.239.

Selection AP: logistic 0.214, boosting 0.228. Calibration barely moved
Brier on this run; both calibrated and raw scores are in
`artifacts/metrics.json`.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | readiness + model version |
| `GET /model-info` | metadata |
| `POST /predict` | single encounter score |
| `POST /triage` | capacity-limited ranked queue |

## Layout

```
carequeue/     core, train, api, dashboard, monitor
tests/         synthetic contract tests
artifacts/     local only (gitignored)
data/          local only (gitignored)
```

## Data

- https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- Paper: https://doi.org/10.1155/2014/781670

Raw CSV and trained artifacts are not in git. Check UCI terms before
redistributing the data.

## Limits

Old observational data, incomplete readmission capture possible, no claim
of fairness from dropping race/gender, no causal estimate of follow-up
benefit, not validated for clinical use. Demo with synthetic inputs only.
