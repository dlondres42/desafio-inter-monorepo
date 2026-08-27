from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from dolores.inference import FeatureValidationError, load_model, save_bundle

FEATURES = ["f0", "f1", "f2"]


class BareEstimator:
    """Satisfies Predictor and nothing more. Must live at module scope to pickle."""

    classes_ = ("alpha", "beta", "gamma")
    feature_names_in_ = FEATURES

    def predict(self, X: pd.DataFrame) -> list[str]:
        return ["alpha"] * len(X)


def test_predict_returns_label_and_confidence(
    bundle_path: Path, payload: dict[str, float]
) -> None:
    prediction = load_model(bundle_path).predict(payload)

    assert prediction.label in ("alpha", "beta", "gamma")
    assert prediction.confidence is not None
    assert 0.0 <= prediction.confidence <= 1.0


def test_predict_label_is_a_builtin_str(
    bundle_path: Path, payload: dict[str, float]
) -> None:
    """Not numpy.str_ — FastAPI's JSON encoder depends on the builtin."""
    assert type(load_model(bundle_path).predict(payload).label) is str


def test_predict_confidence_is_a_builtin_float(
    bundle_path: Path, payload: dict[str, float]
) -> None:
    assert type(load_model(bundle_path).predict(payload).confidence) is float


def test_predict_is_order_independent(
    bundle_path: Path, payload: dict[str, float]
) -> None:
    model = load_model(bundle_path)
    reversed_payload = dict(reversed(list(payload.items())))

    assert model.predict(reversed_payload) == model.predict(payload)


def test_predict_matches_a_direct_estimator_call(
    bundle_path: Path,
    payload: dict[str, float],
    fitted_estimator: LogisticRegression,
) -> None:
    """Proves the reordering is correct, not merely stable."""
    ordered = pd.DataFrame([[payload[name] for name in FEATURES]], columns=FEATURES)
    expected = fitted_estimator.predict(ordered)[0]

    assert load_model(bundle_path).predict(payload).label == expected


def test_predict_rejects_missing_feature(
    bundle_path: Path, payload: dict[str, float]
) -> None:
    del payload["f1"]

    with pytest.raises(FeatureValidationError) as excinfo:
        load_model(bundle_path).predict(payload)

    message = str(excinfo.value)
    assert "f1" in message
    assert "f0" not in message, "only the offending field should be named"


def test_predict_rejects_extra_feature(
    bundle_path: Path, payload: dict[str, float]
) -> None:
    payload["surprise"] = 1.0

    with pytest.raises(FeatureValidationError, match="surprise"):
        load_model(bundle_path).predict(payload)


def test_predict_reports_missing_and_extra_together(
    bundle_path: Path, payload: dict[str, float]
) -> None:
    """One round trip per bad request, not whack-a-mole for the API caller."""
    del payload["f1"]
    payload["surprise"] = 1.0

    with pytest.raises(FeatureValidationError) as excinfo:
        load_model(bundle_path).predict(payload)

    message = str(excinfo.value)
    assert "f1" in message
    assert "surprise" in message


def test_predict_coerces_numeric_strings(
    bundle_path: Path, payload: dict[str, Any]
) -> None:
    model = load_model(bundle_path)
    expected = model.predict(payload)

    assert model.predict({k: str(v) for k, v in payload.items()}) == expected


@pytest.mark.parametrize("bad", ["abc", None, float("nan")])
def test_predict_rejects_unusable_value(
    bundle_path: Path, payload: dict[str, Any], bad: object
) -> None:
    payload["f1"] = bad

    with pytest.raises(FeatureValidationError, match="f1"):
        load_model(bundle_path).predict(payload)


def test_predict_does_not_mutate_the_payload(
    bundle_path: Path, payload: dict[str, float]
) -> None:
    before = dict(payload)

    load_model(bundle_path).predict(payload)

    assert payload == before


def test_predict_confidence_is_none_without_predict_proba(
    tmp_path: Path, frame: pd.DataFrame, payload: dict[str, float]
) -> None:
    """Confidence is optional; a Predictor only has to predict."""
    path = save_bundle(
        tmp_path / "bare.zip", BareEstimator(), frame[FEATURES], target="label"
    )

    prediction = load_model(path).predict(payload)

    assert prediction.label == "alpha"
    assert prediction.confidence is None
