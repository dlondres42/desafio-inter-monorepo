from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_MODEL_PATH = "/model/iris-classifier.zip"


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything this service needs to know in order to start."""

    model_path: Path


@lru_cache
def get_settings() -> Settings:
    """Read settings once per process."""
    return Settings(model_path=Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)))
