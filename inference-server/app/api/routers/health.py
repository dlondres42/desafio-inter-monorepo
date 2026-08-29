"""Probe endpoints."""

from fastapi import APIRouter, HTTPException, Request

from app.schemas.health import HealthResponse, ReadyResponse

router = APIRouter()


@router.get("/healthz")
async def check_liveness() -> HealthResponse:
    """Liveness: the process is up. Deliberately says nothing about the model."""
    return HealthResponse()


@router.get("/readyz")
async def check_readiness(request: Request) -> ReadyResponse:
    """Readiness: a model is loaded and this pod can answer.

    Raises:
        HTTPException: 503 while no model is loaded, so the Service withholds
            traffic instead of routing it to a pod that would fail.
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="no model loaded")
    return ReadyResponse(model_id=model.manifest.model_id)
