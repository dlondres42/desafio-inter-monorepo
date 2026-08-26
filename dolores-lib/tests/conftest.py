"""Shared fixtures.

Deliberately synthetic. The library is dataset-agnostic, so its unit tests must
not reach for ``data/Iris.csv`` at the repository root — doing so would couple the
package to the monorepo and break testing an installed wheel. The real dataset is
exercised end to end exactly once, by the notebook in ``client_code``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dolores.data import Dataset

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
