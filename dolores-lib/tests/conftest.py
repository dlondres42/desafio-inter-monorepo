"""Shared fixtures.

Deliberately synthetic. The library is dataset-agnostic, so its unit tests must
not reach for ``data/Iris.csv`` at the repository root — doing so would couple the
package to the monorepo and break testing an installed wheel. The real dataset is
exercised end to end exactly once, by the notebook in ``client_code``.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from dolores.data import Dataset
from dolores.inference import ESTIMATOR_NAME, MANIFEST_NAME, save_bundle

CLASSES = ("alpha", "beta", "gamma")
FEATURES = ("f0", "f1", "f2")


@pytest.fixture(scope="session")
def frame() -> pd.DataFrame:
    """Thirty rows, ten per class, three well-separated float features.

    Both numbers are load-bearing. Ten rows per class is what makes a stratified
    ``test_size=0.2`` split come out at exactly two rows per class, so the
    proportions can be asserted as a literal. The centres are far enough apart
    that estimators converge without warnings, which matters once tests start
    running under ``filterwarnings("error")``.
    """
    rng = np.random.default_rng(0)
    parts = [
        pd.DataFrame(
            rng.normal(loc=centre, scale=0.15, size=(10, len(FEATURES))),
            columns=list(FEATURES),
        ).assign(label=name)
        for name, centre in zip(CLASSES, (0.0, 3.0, 6.0), strict=True)
    ]
    return pd.concat(parts, ignore_index=True)


@pytest.fixture
def csv_path(tmp_path: Path, frame: pd.DataFrame) -> Path:
    """``frame`` on disk, carrying a spurious ``Id`` column for ``drop`` to remove."""
    path = tmp_path / "toy.csv"
    ordered = frame.assign(Id=range(1, len(frame) + 1))[["Id", *FEATURES, "label"]]
    ordered.to_csv(path, index=False)
    return path


@pytest.fixture
def dataset(frame: pd.DataFrame) -> Dataset:
    return Dataset(frame=frame.copy(), target="label")


@pytest.fixture(scope="session")
def fitted_estimator(frame: pd.DataFrame) -> LogisticRegression:
    """The smallest thing satisfying ``Predictor`` with the attributes a bundle needs.

    Fit on a DataFrame rather than an array, so ``feature_names_in_`` is populated —
    which is what ``save_bundle`` cross-checks the manifest against.
    """
    estimator = LogisticRegression(max_iter=1000)
    estimator.fit(frame[list(FEATURES)], frame["label"])
    return estimator


@pytest.fixture
def bundle_path(
    tmp_path: Path, fitted_estimator: LogisticRegression, frame: pd.DataFrame
) -> Path:
    return save_bundle(
        tmp_path / "model.zip",
        fitted_estimator,
        frame[list(FEATURES)],
        target="label",
        metrics={"f1": 1.0},
    )


@pytest.fixture
def payload(frame: pd.DataFrame) -> dict[str, float]:
    """A well-formed request body: one row, keyed by feature name."""
    row = frame.iloc[0]
    return {name: float(row[name]) for name in FEATURES}


@pytest.fixture
def tamper(bundle_path: Path, tmp_path: Path) -> Callable[..., Path]:
    """Rewrite members of a valid bundle, to build a broken one.

    Every corruption test — runtime mismatch, missing member, garbage payload,
    check ordering — is one call to this.
    """

    def _tamper(
        *,
        manifest: Mapping[str, Any] | None = None,
        estimator_bytes: bytes | None = None,
        drop: Sequence[str] = (),
    ) -> Path:
        with zipfile.ZipFile(bundle_path) as source:
            members = {name: source.read(name) for name in source.namelist()}

        if manifest is not None:
            members[MANIFEST_NAME] = json.dumps(manifest).encode()
        if estimator_bytes is not None:
            members[ESTIMATOR_NAME] = estimator_bytes
        for name in drop:
            members.pop(name, None)

        broken = tmp_path / "tampered.zip"
        with zipfile.ZipFile(broken, "w") as target:
            for name, data in members.items():
                target.writestr(name, data)
        return broken

    return _tamper


@pytest.fixture
def manifest_dict(bundle_path: Path) -> dict[str, Any]:
    """The raw manifest of a valid bundle, as a mutable dict to corrupt."""
    with zipfile.ZipFile(bundle_path) as bundle:
        loaded: dict[str, Any] = json.loads(bundle.read(MANIFEST_NAME))
    return loaded
