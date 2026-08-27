"""Writing and reading model bundles, and scoring payloads against one.

A bundle is a zip holding exactly two members: the pickled estimator and the
manifest describing it. Keeping them siblings rather than nesting the metadata
inside the pickle is what lets :func:`load_model` reject an incompatible bundle
*before* unpickling it.
"""

from __future__ import annotations

import json
import math
import os
import platform
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

import joblib
import pandas as pd

from dolores.inference._errors import (
    BundleError,
    FeatureValidationError,
    IncompatibleBundleError,
    RuntimeMismatchError,
    SchemaMismatchError,
)
from dolores.inference._manifest import (
    CHECKED_PACKAGES,
    ESTIMATOR_NAME,
    FORMAT_VERSION,
    MANIFEST_NAME,
    Feature,
    Manifest,
    Target,
    normalise_dtype,
)


@runtime_checkable
class Predictor(Protocol):
    """The one thing a served model has to do.

    ``isinstance`` against a runtime-checkable Protocol verifies that ``predict``
    is *present*, not that its signature matches — a real limit, and the reason
    ``classes_`` and ``feature_names_in_`` get explicit checks in
    :func:`save_bundle` instead of being declared here as members.
    """

    def predict(self, X: pd.DataFrame) -> Any: ...


@dataclass(frozen=True, slots=True)
class Prediction:
    """One scored payload.

    Attributes:
        label: The predicted class, always a builtin ``str``.
        confidence: Highest class probability, or ``None`` when the estimator
            has no ``predict_proba``.
    """

    label: str
    confidence: float | None


def _installed_runtime() -> dict[str, str]:
    """Versions of everything a pickle's compatibility depends on.

    Read through ``importlib.metadata`` rather than by importing the packages,
    which is what keeps ``import dolores.inference`` from pulling in scikit-learn.
    """
    runtime = {"python": platform.python_version()}
    for package in ("scikit-learn", "numpy", "dolores-lib"):
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:  # pragma: no cover - all three are hard deps
            continue
    return runtime


def _major_minor(raw: str) -> str:
    return ".".join(raw.split(".")[:2])


def _default_model_id(estimator: Predictor) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = type(estimator).__name__.lower()
    # The timestamp alone collides within a second; the suffix makes the id
    # unique without making it unreadable.
    return f"{slug}-{stamp}-{uuid4().hex[:6]}"


def save_bundle(
    path: Path | str,
    estimator: Predictor,
    X: pd.DataFrame,
    *,
    target: str,
    metrics: Mapping[str, float] | None = None,
    model_id: str | None = None,
) -> Path:
    """Write a fitted estimator and its manifest to a bundle.

    Takes primitives rather than a training result so that this package never
    needs to import :mod:`dolores.experiment` — the serving path must not depend
    on training code.

    Every check runs before anything is written, so a refusal leaves no file
    behind, and the write itself is atomic.

    Args:
        path: Destination ``.zip``.
        estimator: A fitted estimator exposing ``classes_`` and
            ``feature_names_in_``.
        X: The frame the estimator was fit on. Its columns and dtypes become the
            manifest's feature schema.
        target: Name of the column the estimator predicts.
        metrics: Optional held-out scores to carry in the manifest.
        model_id: Identifier to record. Generated if omitted.

    Returns:
        The path written.

    Raises:
        TypeError: If ``estimator`` does not satisfy :class:`Predictor`.
        SchemaMismatchError: If the estimator is unfitted, was fit on an array
            rather than a DataFrame, or disagrees with ``X`` about the feature
            names or their order.
    """
    path = Path(path)

    if not isinstance(estimator, Predictor):
        raise TypeError(
            f"{type(estimator).__name__} does not satisfy Predictor: no predict method"
        )
    if not hasattr(estimator, "classes_"):
        raise SchemaMismatchError(
            f"{type(estimator).__name__} has no classes_; it does not look fitted"
        )
    if not hasattr(estimator, "feature_names_in_"):
        raise SchemaMismatchError(
            f"{type(estimator).__name__} has no feature_names_in_; fit it on a "
            "DataFrame so the manifest can record feature names"
        )

    # Order matters as much as membership: predict is positional, so a permuted
    # frame would silently feed the wrong column to the wrong coefficient.
    fitted_names = [str(name) for name in estimator.feature_names_in_]
    frame_names = [str(column) for column in X.columns]
    if fitted_names != frame_names:
        raise SchemaMismatchError(
            "estimator and frame disagree about features:\n"
            f"  estimator was fit on: {fitted_names}\n"
            f"  frame provides:       {frame_names}"
        )

    manifest = Manifest(
        format_version=FORMAT_VERSION,
        model_id=model_id or _default_model_id(estimator),
        created_at=datetime.now(UTC).isoformat(),
        estimator=f"{type(estimator).__module__}.{type(estimator).__name__}",
        features=tuple(
            Feature(name=name, dtype=normalise_dtype(X[name].dtype))
            for name in frame_names
        ),
        target=Target(
            name=target,
            classes=tuple(str(label) for label in estimator.classes_),
        ),
        runtime=_installed_runtime(),
        metrics={str(k): float(v) for k, v in (metrics or {}).items()},
    )

    staging = path.with_name(path.name + ".tmp")
    try:
        with zipfile.ZipFile(staging, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr(MANIFEST_NAME, json.dumps(manifest.to_dict(), indent=2))
            with bundle.open(ESTIMATOR_NAME, "w") as member:
                joblib.dump(estimator, member)
        os.replace(staging, path)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    return path


def _open_bundle(path: Path | str) -> zipfile.ZipFile:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such bundle: {path}")
    try:
        bundle = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise BundleError(f"{path} is not a zip archive: {exc}") from exc
    return bundle


def read_manifest(path: Path | str) -> Manifest:
    """Read a bundle's manifest without unpickling its estimator.

    Args:
        path: Path to a bundle.

    Returns:
        The validated manifest.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        BundleError: If the file is not a zip or has no manifest member.
        IncompatibleBundleError: If the manifest is malformed.
    """
    with _open_bundle(path) as bundle:
        if MANIFEST_NAME not in bundle.namelist():
            raise BundleError(f"bundle {path} has no {MANIFEST_NAME} member")
        raw = json.loads(bundle.read(MANIFEST_NAME))
    return Manifest.from_dict(raw)


def _check_runtime(manifest: Manifest) -> None:
    """Compare recorded against installed versions, on major.minor only.

    Patch releases do not change pickle compatibility; scikit-learn minor
    releases do. Exact-string comparison would fail on harmless upgrades and
    train people to pass ``check_runtime=False``, which is worse than a check
    calibrated to what actually breaks.
    """
    installed = _installed_runtime()
    for package in CHECKED_PACKAGES:
        recorded = manifest.runtime.get(package)
        if recorded is None or package not in installed:
            continue
        if _major_minor(recorded) != _major_minor(installed[package]):
            raise RuntimeMismatchError(
                f"bundle was written against {package} {recorded}, "
                f"this environment has {installed[package]}"
            )


class LoadedModel:
    """A bundle, loaded and ready to score payloads.

    Owns the whole gap between parsed JSON and a prediction: field validation,
    dtype coercion and column ordering all happen here, so the inference server
    never touches pandas or numpy.
    """

    def __init__(self, manifest: Manifest, estimator: Predictor) -> None:
        self.manifest = manifest
        self._estimator = estimator

    @property
    def features(self) -> list[str]:
        """Feature names in canonical order."""
        return self.manifest.feature_names

    def predict(self, payload: Mapping[str, Any]) -> Prediction:
        """Score one payload.

        The payload is keyed by feature name, so field order is irrelevant — the
        manifest defines the order the estimator sees.

        Args:
            payload: One row, keyed by feature name.

        Returns:
            The predicted label, with confidence when available.

        Raises:
            FeatureValidationError: If fields are missing or unexpected, or a
                value cannot be read as the dtype the model was fit on.
        """
        self._validate_fields(payload)
        row = {
            feature.name: _coerce(payload[feature.name], feature)
            for feature in self.manifest.features
        }
        frame = pd.DataFrame([row], columns=self.features)

        label = str(self._estimator.predict(frame)[0])

        confidence: float | None = None
        predict_proba = getattr(self._estimator, "predict_proba", None)
        if predict_proba is not None:
            confidence = float(predict_proba(frame).max())
        return Prediction(label=label, confidence=confidence)

    def _validate_fields(self, payload: Mapping[str, Any]) -> None:
        expected = set(self.features)
        provided = set(payload)
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        if not missing and not unexpected:
            return

        # Both halves in one message: an API caller should need one round trip
        # to fix a bad request, not one per wrong field.
        problems = []
        if missing:
            problems.append(f"missing {missing}")
        if unexpected:
            problems.append(f"unexpected {unexpected}")
        raise FeatureValidationError(
            f"payload does not match the model's schema: {'; '.join(problems)}"
        )


def _coerce(value: Any, feature: Feature) -> Any:
    """Read one payload value as the dtype the model was fit on."""
    if value is None:
        raise FeatureValidationError(
            f"feature {feature.name!r} is null; expected {feature.dtype}"
        )
    try:
        if feature.dtype == "float64":
            coerced = float(value)
        elif feature.dtype == "int64":
            coerced = int(value)
        elif feature.dtype == "bool":
            return bool(value)
        else:
            return str(value)
    except (TypeError, ValueError) as exc:
        raise FeatureValidationError(
            f"feature {feature.name!r} cannot be read as {feature.dtype}: {value!r}"
        ) from exc

    if isinstance(coerced, float) and math.isnan(coerced):
        raise FeatureValidationError(
            f"feature {feature.name!r} is NaN; expected {feature.dtype}"
        )
    return coerced


def load_model(path: Path | str, *, check_runtime: bool = True) -> LoadedModel:
    """Load a bundle for serving.

    Checks run in a deliberate order — manifest, then format version, then
    runtime, and only then unpickling. An artifact that cannot work here is
    rejected before any of its bytes are executed.

    Args:
        path: Path to a bundle.
        check_runtime: Compare recorded versions against installed ones. Only
            disable this when you know the pickle is loadable anyway.

    Returns:
        The loaded model.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        BundleError: If the file is not a zip, is missing a member, or holds
            something that is not a :class:`Predictor`.
        IncompatibleBundleError: If the manifest cannot be read.
        RuntimeMismatchError: If recorded and installed versions disagree.
    """
    with _open_bundle(path) as bundle:
        members = bundle.namelist()
        for required in (MANIFEST_NAME, ESTIMATOR_NAME):
            if required not in members:
                raise BundleError(f"bundle {path} has no {required} member")

        manifest = Manifest.from_dict(json.loads(bundle.read(MANIFEST_NAME)))
        if check_runtime:
            _check_runtime(manifest)

        with bundle.open(ESTIMATOR_NAME) as member:
            try:
                estimator = joblib.load(member)
            except Exception as exc:
                raise BundleError(
                    f"bundle {path} holds an unreadable estimator: {exc}"
                ) from exc

    if not isinstance(estimator, Predictor):
        raise BundleError(
            f"bundle {path} holds a {type(estimator).__name__}, which does not "
            "satisfy Predictor: no predict method"
        )
    return LoadedModel(manifest=manifest, estimator=estimator)


__all__ = [
    "IncompatibleBundleError",
    "LoadedModel",
    "Predictor",
    "Prediction",
    "load_model",
    "read_manifest",
    "save_bundle",
]
