"""Request and response shapes for the prediction endpoint."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """One payload to score.

    ``instance`` is deliberately an open mapping rather than a typed model with
    named fields. A typed request would hardcode one dataset's schema into the
    server and defeat the point of serving any bundle; the loaded model's
    manifest validates the contents instead.

    The division of labour: pydantic validates the HTTP envelope, the manifest
    validates the feature schema.
    """

    instance: dict[str, Any] = Field(
        ..., description="Feature name to value. Order is irrelevant."
    )


class PredictResponse(BaseModel):
    """The scored result."""

    label: str
    confidence: float | None = Field(
        None, description="Highest class probability, absent without predict_proba."
    )
    model_id: str = Field(..., description="Identifies the bundle that answered.")
