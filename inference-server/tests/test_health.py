from fastapi.testclient import TestClient


def test_liveness_probe(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"
