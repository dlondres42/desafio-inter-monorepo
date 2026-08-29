"""Scoring and model-description endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from dolores.inference import FeatureValidationError, LoadedModel
from fastapi import APIRouter, Depends, HTTPException, Request

from app.schemas.predict import PredictRequest, PredictResponse

router = APIRouter()


def get_model(request: Request) -> LoadedModel:
    """Reach the model the lifespan loaded."""

    return request.app.state.model  # type: ignore[no-any-return]


ModelDep = Annotated[LoadedModel, Depends(get_model)]


@router.post("/predict")
async def predict(body: PredictRequest, model: ModelDep) -> PredictResponse:
    """Score one payload against the loaded model.

    Raises:
        HTTPException: 422 when the payload does not match the model's schema.
            The library's message names every offending field, so a caller needs
            one round trip to fix a bad request rather than one per field.
    """
    try:
        prediction = model.predict(body.instance)
    except FeatureValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictResponse(
        label=prediction.label,
        confidence=prediction.confidence,
        model_id=model.manifest.model_id,
    )


@router.get("/model")
async def describe_model(model: ModelDep) -> dict[str, Any]:
    """The loaded bundle's manifest: features, dtypes, classes, runtime."""
    return model.manifest.to_dict()
