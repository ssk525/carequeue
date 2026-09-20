# CareQueue — project story

Where this lives on your machine:

`/Users/saswatsuvamkumar/Desktop/MY GITS /carequeue`

GitHub: https://github.com/ssk525/carequeue

---

## The story (use this tone on LinkedIn / resume / interviews)

Hospitals discharge more patients than a care team can call back the same day. If you treat every discharge the same, high-risk patients get lost in the queue. I built CareQueue around that constraint: at discharge, score the chance a patient comes back within 30 days, then return only the top-k people the team can actually contact given today’s capacity.

I used the public UCI Diabetes 130-US Hospitals dataset (~100K encounters). The hard part was not picking a fancy model — it was making the evaluation honest. Patients can have multiple visits, so I split by `patient_nbr` so the same person never leaks from train into test. I kept race and gender out of the features but still checked performance by subgroup. I compared logistic regression to histogram gradient boosting on average precision, calibrated probabilities on a separate fold, and judged the final model the way ops would use it: precision, recall, and lift inside a 10% follow-up budget.

On the held-out test set the selected model reached about 0.22 average precision (vs ~0.11 prevalence baseline) and about 2.3× lift at 10% capacity. Then I wrapped it in a FastAPI service (`/predict` and `/triage`), a small Streamlit demo, numeric drift checks, tests, and Docker so the project is not just a notebook — it is a runnable pipeline from data to ranked queue.

This is a portfolio / research demo on historical data. It is not a clinical product, and a high risk score does not prove that calling the patient prevents readmission.

---

## What actually happens in the project (step by step)

1. **Download / load data** — `python -m carequeue.train --download` pulls the UCI zip and extracts `diabetic_data.csv` into `data/` (gitignored).
2. **Build the cohort** — drop death/hospice discharge dispositions; label `<30` readmission as the positive class.
3. **Split by patient** — train / selection / calibration / test with no shared patients.
4. **Train candidates** — logistic regression and hist. gradient boosting; pick the better average precision on the selection fold.
5. **Calibrate** — fit a calibrator on the calibration fold only.
6. **Evaluate** — untouched test set: AP, ROC-AUC, Brier, queue metrics @10%, bootstrap CI, subgroup table, permutation importance → `artifacts/metrics.json` + `artifacts/model.joblib`.
7. **Serve** — FastAPI loads the artifact; `/predict` scores one encounter; `/triage` returns a capacity-limited ranked list.
8. **Demo UI** — Streamlit form talks to the API with synthetic inputs.
9. **Monitor** — `carequeue.monitor` compares a new batch’s numeric distributions to the training reference (PSI).

---

## What each file is for

| Path | Role |
|---|---|
| `carequeue/core.py` | Features, cleaning, patient splits, models, queue selection, metrics, drift |
| `carequeue/train.py` | Download, train, select, calibrate, write artifacts |
| `carequeue/api.py` | FastAPI `/health`, `/model-info`, `/predict`, `/triage` |
| `carequeue/dashboard.py` | Streamlit demo |
| `carequeue/monitor.py` | Offline numeric drift report |
| `tests/test_project.py` | Synthetic tests (splits, API contract, no leakage) |
| `artifacts/` | Local model + metrics (not committed) |
| `data/` | Local CSV (not committed) |
| `README.md` | Short setup + verified numbers |
| `Makefile` | `make install test train api dashboard` |

---

## Tools and why they are there

- **pandas / scikit-learn** — cohort prep, pipelines, models, calibration, metrics
- **FastAPI + pydantic** — typed inference API, reject unknown fields (no accidental target leakage in requests)
- **Streamlit** — quick interactive demo for interviews
- **pytest** — prove splits, queue logic, and API behavior without the full hospital CSV
- **Docker + GitHub Actions (local CI file)** — show you can package and check the project
- **joblib** — save/load the trained bundle

---

## Honest note on originality

This was **not cloned from an existing CareQueue GitHub repo**.

What *is* public / shared knowledge:

- The **UCI Diabetes 130-US Hospitals** dataset and the Strack et al. (2014) readmission study are well known.
- Predicting 30-day readmission is a **common** ML teaching problem (many blogs, Kaggle-style notebooks, course projects).

What we built here specifically:

- The **CareQueue** framing (capacity-limited follow-up queue, not “train a classifier and stop”)
- The **repo layout**: patient-disjoint train/selection/calibration/test, FastAPI triage, Streamlit, drift script, tests, Docker
- Code written and validated in this project folder, trained locally, then pushed to **your** `ssk525/carequeue`

So: classic public dataset + standard ML methods, custom end-to-end product framing and implementation — not a copy-paste of someone else’s full repository.
