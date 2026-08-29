from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").is_dir())
BUNDLE = REPO_ROOT / "data" / "iris-classifier.zip"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A client whose app has really loaded a bundle.

    Anchored at the repository root rather than the working directory, so the
    suite passes from anywhere. The bundle is the one the consumer notebook
    writes, which makes these tests an assertion that the two halves of the
    project still agree on the format.
    """
    if not BUNDLE.exists():
        pytest.skip(f"no bundle at {BUNDLE}; run the consumer notebook first")

    monkeypatch.setenv("MODEL_PATH", str(BUNDLE))

    from app.config import get_settings

    get_settings.cache_clear()
    from app.main import app

    # `with` is what triggers the lifespan, and therefore the model load.
    with TestClient(app) as running:
        yield running
