from fastapi import APIRouter

from app.schemas.health import HealthResponse

router = APIRouter()


@router.get("/healthz")
async def check_liveness() -> HealthResponse:
    return HealthResponse()


@router.get("/readyz")
async def check_readiness() -> None:
    pass
