from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from carequeue.core import (
    EXCLUDED_DISPOSITIONS,
    FEATURES,
    predict_risk,
    top_k_indices,
)


class Encounter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_in_hospital: int = Field(ge=1, le=60, strict=True)
    num_lab_procedures: int = Field(ge=0, le=500, strict=True)
    num_procedures: int = Field(ge=0, le=100, strict=True)
    num_medications: int = Field(ge=0, le=300, strict=True)
    number_outpatient: int = Field(ge=0, le=100, strict=True)
    number_emergency: int = Field(ge=0, le=100, strict=True)
    number_inpatient: int = Field(ge=0, le=100, strict=True)
    number_diagnoses: int = Field(ge=1, le=100, strict=True)

    age: str = Field(max_length=32)
    admission_type_id: str = Field(max_length=2)
    discharge_disposition_id: str = Field(max_length=2)
    admission_source_id: str = Field(max_length=2)
    A1Cresult: str = Field(max_length=16)
    max_glu_serum: str = Field(max_length=16)
    insulin: str = Field(max_length=16)
    change: str = Field(max_length=16)
    diabetesMed: str = Field(max_length=16)


class QueueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[Encounter] = Field(min_length=1, max_length=1000)
    capacity: int = Field(ge=0, le=1000, strict=True)


def validate_eligibility(records: list[dict]) -> None:
    for record in records:
        if record["discharge_disposition_id"] in EXCLUDED_DISPOSITIONS:
            raise HTTPException(
                status_code=422,
                detail="A record is outside the model's eligible cohort.",
            )


def create_app(model_path: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = Path(
            model_path
            or os.getenv("MODEL_PATH", "artifacts/model.joblib")
        )
        if not path.is_file():
            raise RuntimeError("Model artifact missing. Run training first.")

        # Joblib/pickle can execute code. Load only your own trusted artifacts.
        bundle = joblib.load(path)
        if bundle.get("feature_names") != FEATURES:
            raise RuntimeError("Model and API feature contracts do not match.")

        app.state.bundle = bundle
        yield

    app = FastAPI(
        title="CareQueue",
        version="0.1.0",
        description="Research-only discharge follow-up prioritization.",
        lifespan=lifespan,
    )

    def score(request: Request, records: list[dict]):
        validate_eligibility(records)
        try:
            return predict_risk(
                request.app.state.bundle,
                pd.DataFrame(records),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/health")
    def health(request: Request):
        return {
            "status": "ready",
            "model_version": request.app.state.bundle["version"],
        }

    @app.get("/model-info")
    def model_info(request: Request):
        return request.app.state.bundle["metadata"]

    @app.post("/predict")
    def predict(record: Encounter, request: Request):
        probabilities = score(request, [record.model_dump()])
        return {
            "readmission_probability": float(probabilities[0]),
            "model_version": request.app.state.bundle["version"],
            "clinical_use": False,
        }

    @app.post("/triage")
    def triage(payload: QueueRequest, request: Request):
        if payload.capacity > len(payload.records):
            raise HTTPException(
                status_code=422,
                detail="Capacity cannot exceed batch size.",
            )

        records = [record.model_dump() for record in payload.records]
        probabilities = score(request, records)
        selected = top_k_indices(probabilities, payload.capacity)

        return {
            "batch_size": len(records),
            "capacity": payload.capacity,
            "selected": [
                {
                    "rank": rank,
                    "input_index": int(index),
                    "readmission_probability": float(probabilities[index]),
                }
                for rank, index in enumerate(selected, start=1)
            ],
            "model_version": request.app.state.bundle["version"],
            "clinical_use": False,
        }

    return app


app = create_app()
