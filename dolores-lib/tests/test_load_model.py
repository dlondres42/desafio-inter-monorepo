from __future__ import annotations

import pickle
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from dolores.inference import (
    ESTIMATOR_NAME,
    MANIFEST_NAME,
    BundleError,
    RuntimeMismatchError,
    load_model,
)


def test_load_model_exposes_the_manifest(bundle_path: Path) -> None:
    assert load_model(bundle_path).manifest.feature_names == ["f0", "f1", "f2"]


def test_load_model_exposes_features(bundle_path: Path) -> None:
    """The server's /model route reads this to advertise its schema."""
    assert load_model(bundle_path).features == ["f0", "f1", "f2"]


def test_load_model_raises_for_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "absent.zip")


def test_load_model_raises_for_non_zip(tmp_path: Path) -> None:
    path = tmp_path / "not-a-bundle.zip"
    path.write_text("this is not a zip archive")

    with pytest.raises(BundleError):
        load_model(path)


def test_load_model_raises_when_estimator_member_missing(
    tamper: Callable[..., Path],
) -> None:
    with pytest.raises(BundleError, match=ESTIMATOR_NAME):
        load_model(tamper(drop=[ESTIMATOR_NAME]))


def test_load_model_raises_when_manifest_member_missing(
    tamper: Callable[..., Path],
) -> None:
    with pytest.raises(BundleError, match=MANIFEST_NAME):
        load_model(tamper(drop=[MANIFEST_NAME]))


def test_load_model_rejects_runtime_mismatch(
    tamper: Callable[..., Path], manifest_dict: dict[str, Any]
) -> None:
    manifest_dict["runtime"]["scikit-learn"] = "0.1.2"

    with pytest.raises(RuntimeMismatchError) as excinfo:
        load_model(tamper(manifest=manifest_dict))

    message = str(excinfo.value)
    assert "scikit-learn" in message
    assert "0.1.2" in message


def test_load_model_checks_runtime_before_unpickling(
    tamper: Callable[..., Path], manifest_dict: dict[str, Any]
) -> None:
    """Corrupt both the runtime and the pickle; the runtime error must win.

    This is the ordering the sibling manifest exists to make possible. If the
    estimator were unpickled first, an incompatible bundle would surface as an
    UnpicklingError — or worse, load and mispredict.
    """
    manifest_dict["runtime"]["scikit-learn"] = "0.1.2"
    broken = tamper(manifest=manifest_dict, estimator_bytes=b"definitely not a pickle")

    with pytest.raises(RuntimeMismatchError):
        load_model(broken)


def test_load_model_skips_runtime_check_when_disabled(
    tamper: Callable[..., Path], manifest_dict: dict[str, Any]
) -> None:
    manifest_dict["runtime"]["scikit-learn"] = "0.1.2"

    model = load_model(tamper(manifest=manifest_dict), check_runtime=False)

    assert model.features == ["f0", "f1", "f2"]


def test_load_model_tolerates_patch_version_drift(
    tamper: Callable[..., Path], manifest_dict: dict[str, Any]
) -> None:
    """Only major.minor is compared. Patch bumps do not break pickles."""
    major, minor, _ = manifest_dict["runtime"]["scikit-learn"].split(".", 2)
    manifest_dict["runtime"]["scikit-learn"] = f"{major}.{minor}.999"

    assert load_model(tamper(manifest=manifest_dict)).features == ["f0", "f1", "f2"]


def test_load_model_rejects_non_predictor_payload(
    tamper: Callable[..., Path],
) -> None:
    """The structural guarantee: whatever is in there must actually predict."""
    with pytest.raises(BundleError, match="Predictor"):
        load_model(tamper(estimator_bytes=pickle.dumps({"not": "a model"})))


def test_inference_package_does_not_import_sklearn() -> None:
    """The server's import surface stays one package wide.

    sklearn still loads when a sklearn pickle is unpickled; the claim is about
    importing the package, and it is what lets a bundle carry a non-sklearn
    predictor.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import dolores.inference, sys; "
            "assert 'sklearn' not in sys.modules, sorted(sys.modules)",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
