from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Liveness payload. Intentionally carries no model information."""

    status: str = "alive"


class ReadyResponse(BaseModel):
    """Readiness payload, naming the bundle this pod is serving."""

    status: str = "ready"
    model_id: str
