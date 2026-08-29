from __future__ import annotations

from fastapi.testclient import TestClient

PAYLOAD = {
    "SepalLengthCm": 6.3,
    "SepalWidthCm": 2.8,
    "PetalLengthCm": 5.1,
    "PetalWidthCm": 1.5,
}


def test_predict_returns_a_label(client: TestClient) -> None:
    response = client.post("/predict", json={"instance": PAYLOAD})

    assert response.status_code == 200
    body = response.json()
    assert body["label"].startswith("Iris-")
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_reports_which_bundle_answered(client: TestClient) -> None:
    """Two pods may run different models; the response says which one this was."""
    body = client.post("/predict", json={"instance": PAYLOAD}).json()

    assert body["model_id"]


def test_predict_is_field_order_independent(client: TestClient) -> None:
    reordered = dict(reversed(list(PAYLOAD.items())))

    first = client.post("/predict", json={"instance": PAYLOAD}).json()
    second = client.post("/predict", json={"instance": reordered}).json()

    assert first == second


def test_predict_rejects_a_bad_schema_with_422(client: TestClient) -> None:
    """The library's FeatureValidationError is the 422 body, verbatim."""
    bad = {**PAYLOAD}
    bad["PetalLength"] = bad.pop("PetalLengthCm")

    response = client.post("/predict", json={"instance": bad})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "PetalLengthCm" in detail and "PetalLength" in detail


def test_predict_rejects_an_uncoercible_value_with_422(client: TestClient) -> None:
    response = client.post(
        "/predict", json={"instance": {**PAYLOAD, "SepalLengthCm": "wide"}}
    )

    assert response.status_code == 422
    assert "SepalLengthCm" in response.json()["detail"]


def test_predict_rejects_a_malformed_envelope_with_422(client: TestClient) -> None:
    """Pydantic guards the envelope; the manifest guards the contents."""
    assert client.post("/predict", json={"nope": {}}).status_code == 422


def test_model_endpoint_exposes_the_manifest(client: TestClient) -> None:
    """Makes 'this image serves any model' demonstrable rather than asserted."""
    body = client.get("/model").json()

    assert body["target"]["name"] == "Species"
    assert body["features"][0]["name"] == "SepalLengthCm"
    assert "scikit-learn" in body["runtime"]


def test_predict_with_an_overridden_model() -> None:
    """The payoff of Depends: swap the model without a bundle, env var or lifespan.

    Nothing here touches the filesystem, so routing and error mapping stay
    testable even where no artifact exists — a CI runner, say.
    """
    from types import SimpleNamespace

    from dolores.inference import Prediction

    from app.api.routers.predict import get_model
    from app.main import app

    stub = SimpleNamespace(
        predict=lambda payload: Prediction(label="stub-label", confidence=0.5),
        manifest=SimpleNamespace(model_id="stub-model"),
    )
    app.dependency_overrides[get_model] = lambda: stub
    try:
        # No `with`: the lifespan never runs, so no bundle is ever loaded.
        response = TestClient(app).post("/predict", json={"instance": {"any": 1}})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "label": "stub-label",
        "confidence": 0.5,
        "model_id": "stub-model",
    }
