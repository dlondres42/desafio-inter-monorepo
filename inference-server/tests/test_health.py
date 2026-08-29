from fastapi.testclient import TestClient


def test_liveness_probe(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_liveness_does_not_depend_on_the_model(client: TestClient) -> None:
    """Liveness that checks the model turns a bad artifact into a restart loop.

    A stalled rollout is recoverable; CrashLoopBackOff across every replica is
    not. Readiness is what gates traffic.
    """
    import app.api.routers.health as module

    assert "model" not in module.check_liveness.__code__.co_names


def test_readiness_reports_the_loaded_bundle(client: TestClient) -> None:
    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["model_id"]


def test_root_still_responds(client: TestClient) -> None:
    assert client.get("/").json() == {"Hello": "World"}
