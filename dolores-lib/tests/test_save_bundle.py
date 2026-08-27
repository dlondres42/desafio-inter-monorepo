from __future__ import annotations

import json
import platform
import re
import zipfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from dolores.inference import (
    ESTIMATOR_NAME,
    MANIFEST_NAME,
    SchemaMismatchError,
    read_manifest,
    save_bundle,
)

FEATURES = ["f0", "f1", "f2"]


class StubEstimator:
    """Satisfies Predictor with a schema chosen per test. Module-level so it pickles."""

    def __init__(self, feature_names: Any, classes: tuple[str, ...] = ("x", "y")):
        self.feature_names_in_ = list(feature_names)
        self.classes_ = classes

    def predict(self, X: pd.DataFrame) -> list[str]:
        return [self.classes_[0]] * len(X)


def test_save_bundle_writes_zip_with_exactly_two_members(bundle_path: Path) -> None:
    with zipfile.ZipFile(bundle_path) as bundle:
        assert sorted(bundle.namelist()) == [ESTIMATOR_NAME, MANIFEST_NAME]


def test_read_manifest_does_not_unpickle(
    bundle_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest is a sibling of the pickle precisely so this is possible.

    If reading metadata required unpickling, the runtime check could never
    precede it, and the whole ordering guarantee would be circular.
    """

    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("read_manifest must not unpickle the estimator")

    monkeypatch.setattr(joblib, "load", _explode)

    assert read_manifest(bundle_path).feature_names == FEATURES


def test_save_bundle_records_installed_runtime(bundle_path: Path) -> None:
    runtime = read_manifest(bundle_path).runtime

    assert runtime["python"] == platform.python_version()
    assert runtime["scikit-learn"] == version("scikit-learn")
    assert runtime["numpy"] == version("numpy")
    assert runtime["dolores-lib"] == version("dolores-lib")


def test_save_bundle_records_estimator_class_path(bundle_path: Path) -> None:
    assert read_manifest(bundle_path).estimator.endswith("LogisticRegression")


def test_save_bundle_records_classes_from_estimator(bundle_path: Path) -> None:
    classes = read_manifest(bundle_path).target.classes

    assert classes == ("alpha", "beta", "gamma")
    assert all(type(label) is str for label in classes)


def test_save_bundle_records_feature_dtypes(bundle_path: Path) -> None:
    features = read_manifest(bundle_path).features

    assert [(f.name, f.dtype) for f in features] == [(n, "float64") for n in FEATURES]


def test_save_bundle_normalises_dtypes_to_a_closed_set(tmp_path: Path) -> None:
    """pandas 2 says ``object`` for strings, pandas 3 says ``str``.

    The manifest must say neither: recording the raw dtype would make a bundle
    written under one pandas major unloadable under the other.
    """
    frame = pd.DataFrame(
        {
            "f": [1.0, 2.0],
            "i": [1, 2],
            "b": [True, False],
            "s": ["a", "b"],
        }
    )

    path = save_bundle(
        tmp_path / "m.zip", StubEstimator(frame.columns), frame, target="t"
    )

    assert {f.name: f.dtype for f in read_manifest(path).features} == {
        "f": "float64",
        "i": "int64",
        "b": "bool",
        "s": "string",
    }


def test_save_bundle_records_target_name(bundle_path: Path) -> None:
    assert read_manifest(bundle_path).target.name == "label"


def test_save_bundle_refuses_on_renamed_column(
    tmp_path: Path, fitted_estimator: LogisticRegression, frame: pd.DataFrame
) -> None:
    renamed = frame[FEATURES].rename(columns={"f2": "elsewhere"})

    with pytest.raises(SchemaMismatchError, match="elsewhere"):
        save_bundle(tmp_path / "m.zip", fitted_estimator, renamed, target="label")


def test_save_bundle_refuses_on_permuted_column_order(
    tmp_path: Path, fitted_estimator: LogisticRegression, frame: pd.DataFrame
) -> None:
    """Same names, different order. A set comparison passes here and is wrong."""
    permuted = frame[["f2", "f1", "f0"]]

    with pytest.raises(SchemaMismatchError):
        save_bundle(tmp_path / "m.zip", fitted_estimator, permuted, target="label")


def test_save_bundle_does_not_write_when_it_refuses(
    tmp_path: Path, fitted_estimator: LogisticRegression, frame: pd.DataFrame
) -> None:
    path = tmp_path / "m.zip"

    with pytest.raises(SchemaMismatchError):
        save_bundle(path, fitted_estimator, frame[["f2", "f1", "f0"]], target="label")

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_save_bundle_refuses_estimator_fitted_on_ndarray(
    tmp_path: Path, frame: pd.DataFrame
) -> None:
    estimator = LogisticRegression(max_iter=1000)
    estimator.fit(frame[FEATURES].to_numpy(), frame["label"])

    with pytest.raises(SchemaMismatchError, match="DataFrame"):
        save_bundle(tmp_path / "m.zip", estimator, frame[FEATURES], target="label")


def test_save_bundle_refuses_unfitted_estimator(
    tmp_path: Path, frame: pd.DataFrame
) -> None:
    with pytest.raises(SchemaMismatchError, match="fitted"):
        save_bundle(
            tmp_path / "m.zip", LogisticRegression(), frame[FEATURES], target="label"
        )


def test_save_bundle_rejects_non_predictor(tmp_path: Path, frame: pd.DataFrame) -> None:
    with pytest.raises(TypeError, match="Predictor"):
        save_bundle(
            tmp_path / "m.zip",
            object(),  # type: ignore[arg-type]
            frame[FEATURES],
            target="label",
        )


def test_save_bundle_coerces_numpy_metrics(
    tmp_path: Path, fitted_estimator: LogisticRegression, frame: pd.DataFrame
) -> None:
    path = save_bundle(
        tmp_path / "m.zip",
        fitted_estimator,
        frame[FEATURES],
        target="label",
        metrics={"f1": np.float64(0.9)},
    )

    with zipfile.ZipFile(path) as bundle:
        raw: dict[str, Any] = json.loads(bundle.read(MANIFEST_NAME))
    assert type(raw["metrics"]["f1"]) is float


def test_save_bundle_returns_the_written_path(
    tmp_path: Path, fitted_estimator: LogisticRegression, frame: pd.DataFrame
) -> None:
    path = tmp_path / "m.zip"

    assert save_bundle(path, fitted_estimator, frame[FEATURES], target="label") == path


def test_save_bundle_default_model_id_is_readable_and_unique(
    tmp_path: Path, fitted_estimator: LogisticRegression, frame: pd.DataFrame
) -> None:
    ids = [
        read_manifest(
            save_bundle(
                tmp_path / f"m{i}.zip",
                fitted_estimator,
                frame[FEATURES],
                target="label",
            )
        ).model_id
        for i in range(2)
    ]

    assert all(re.fullmatch(r"[a-z0-9]+-\d{8}T\d{6}Z-[0-9a-f]{6}", i) for i in ids)
    assert ids[0] != ids[1]


def test_save_bundle_honours_explicit_model_id(
    tmp_path: Path, fitted_estimator: LogisticRegression, frame: pd.DataFrame
) -> None:
    path = save_bundle(
        tmp_path / "m.zip",
        fitted_estimator,
        frame[FEATURES],
        target="label",
        model_id="hand-picked",
    )

    assert read_manifest(path).model_id == "hand-picked"
