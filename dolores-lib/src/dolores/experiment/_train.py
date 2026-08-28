"""Fit one candidate, score it on held-out data, and hand back everything about it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from dolores.data import Dataset
from dolores.experiment.metrics import ClassificationMetrics, evaluate
from dolores.inference import Predictor, save_bundle


@runtime_checkable
class Estimator(Predictor, Protocol):
    """A :class:`~dolores.inference.Predictor` that can also be fitted.

    Training needs strictly more than serving does, so it asks for strictly more.
    The serving path never sees this protocol.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any: ...


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """One candidate, fitted and scored — which is also exactly one MLflow run.

    Attributes:
        name: Human label for the candidate. Becomes the MLflow run name.
        estimator: The fitted estimator.
        params: ``get_params()`` at fit time, for logging.
        metrics: Scores on the held-out split.
        features: Feature names in the order the estimator was fitted on.
        target: Name of the predicted column.
        train_X: The training features. Kept because :meth:`save_bundle` needs the
            real frame to cross-check against ``feature_names_in_`` — a
            reconstruction would defeat the permuted-column guard — and because it
            doubles as MLflow's ``input_example``.
    """

    name: str
    estimator: Estimator
    params: Mapping[str, Any]
    metrics: ClassificationMetrics
    features: tuple[str, ...]
    target: str
    train_X: pd.DataFrame

    def save_bundle(self, path: Path | str) -> Path:
        """Write this result's estimator and schema to a servable bundle.

        A thin adapter over :func:`dolores.inference.save_bundle`, which takes
        primitives rather than a ``TrainingResult`` so that the serving package
        never has to import this one.

        Args:
            path: Destination ``.zip``.

        Returns:
            The path written.
        """
        return save_bundle(
            path,
            self.estimator,
            self.train_X,
            target=self.target,
            metrics=self.metrics.scalars(),
        )


def train(
    estimator: Estimator,
    train_ds: Dataset,
    test_ds: Dataset,
    *,
    name: str | None = None,
) -> TrainingResult:
    """Fit ``estimator`` on ``train_ds`` and score it on ``test_ds``.

    Args:
        estimator: Any object that can ``fit`` and ``predict``. Fitted in place.
        train_ds: The training split.
        test_ds: The held-out split. Scoring happens here and only here.
        name: Label for the candidate. Defaults to the estimator's class name.

    Returns:
        The fitted estimator with its scores, params and schema.

    Raises:
        TypeError: If ``estimator`` cannot both fit and predict.
        ValueError: If the two datasets disagree about features or target.
    """
    if not isinstance(estimator, Estimator):
        raise TypeError(
            f"{type(estimator).__name__} cannot be trained: "
            "it needs both fit and predict methods"
        )
    if train_ds.target != test_ds.target:
        raise ValueError(
            f"datasets disagree about the target: training on {train_ds.target!r}, "
            f"testing on {test_ds.target!r}"
        )
    if train_ds.features != test_ds.features:
        missing = sorted(set(train_ds.features) ^ set(test_ds.features))
        raise ValueError(
            "datasets disagree about features: "
            f"{', '.join(repr(name) for name in missing) or 'different order'}"
        )

    train_X = train_ds.X
    estimator.fit(train_X, train_ds.y)
    predictions = estimator.predict(test_ds.X)

    # Not every Predictor is a scikit-learn estimator; params are a nicety for
    # logging, not part of the contract.
    get_params = getattr(estimator, "get_params", None)
    params = dict(get_params()) if callable(get_params) else {}

    return TrainingResult(
        name=name or type(estimator).__name__,
        estimator=estimator,
        params=params,
        # tolist() at the boundary: metrics takes plain sequences, so it stays
        # usable without pandas in hand.
        metrics=evaluate(test_ds.y.tolist(), predictions),
        features=tuple(train_ds.features),
        target=train_ds.target,
        train_X=train_X,
    )
