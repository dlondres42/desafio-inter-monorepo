"""Classification metrics as one value object.

A public module inside an otherwise private package, deliberately: these are pure
functions over ``y_true``/``y_pred`` with no dependency on :mod:`dolores.data` or
:mod:`dolores.inference`, so scoring a model this library did not train stays
possible.

Everything here is rendered as builtin types. sklearn returns plain floats for the
scalar metrics but an ``ndarray`` for the confusion matrix, and a numpy scalar
survives every intermediate step only to fail at ``json.dumps`` — so the conversion
happens once, here, rather than at each call site.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """How a fitted model scored on a held-out split.

    Attributes:
        accuracy: Fraction of correct predictions.
        precision: Macro-averaged precision, unweighted across classes.
        recall: Macro-averaged recall.
        f1: Macro-averaged F1.
        labels: Every class seen in either input, sorted. Indexes both axes of
            ``confusion_matrix``.
        confusion_matrix: Rows are true classes, columns are predicted.
        report: Per-class precision/recall/f1/support, plus the ``macro avg`` and
            ``weighted avg`` rows. sklearn's bare ``accuracy`` float is dropped —
            it is :attr:`accuracy`, and leaving it in would make this a mapping of
            mixed types.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    labels: tuple[str, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    report: Mapping[str, Mapping[str, float]]

    def scalars(self) -> dict[str, float]:
        """The four headline numbers, flat.

        This is the MLflow ``log_metrics`` payload and the manifest's ``metrics``
        field: both reject nested structures, so this stays one level deep.
        """
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

    def to_dict(self) -> dict[str, Any]:
        """Render as JSON-serializable builtins, confusion matrix included."""
        return {
            **self.scalars(),
            "labels": list(self.labels),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "report": {k: dict(v) for k, v in self.report.items()},
        }

    def write_json(self, path: Path | str) -> Path:
        """Persist to ``path`` as JSON.

        Args:
            path: Destination file.

        Returns:
            The path written.
        """
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path


def evaluate(y_true: Sequence[Any], y_pred: Sequence[Any]) -> ClassificationMetrics:
    """Score predictions against ground truth.

    Args:
        y_true: Observed labels.
        y_pred: Predicted labels, aligned with ``y_true``.

    Returns:
        Every metric the case asks for, in one value object.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true has {len(y_true)} entries but y_pred has {len(y_pred)}"
        )

    # Fix the label set once and pass it everywhere, so the confusion matrix axes,
    # the macro averages and the report all agree on the same classes in the same
    # order — including a class that appears in only one of the two inputs.
    labels = sorted({str(label) for label in (*y_true, *y_pred)})
    truth = [str(label) for label in y_true]
    predicted = [str(label) for label in y_pred]

    # zero_division=0 rather than the default: a class that is never predicted is
    # 0/0 precision, which is a fact about this split, not a warning worth raising.
    averaged: dict[str, Any] = {
        "labels": labels,
        "average": "macro",
        "zero_division": 0,
    }
    report: dict[str, Any] = classification_report(
        truth, predicted, labels=labels, output_dict=True, zero_division=0
    )
    report.pop("accuracy", None)

    return ClassificationMetrics(
        accuracy=float(accuracy_score(truth, predicted)),
        precision=float(precision_score(truth, predicted, **averaged)),
        recall=float(recall_score(truth, predicted, **averaged)),
        f1=float(f1_score(truth, predicted, **averaged)),
        labels=tuple(labels),
        confusion_matrix=tuple(
            tuple(int(cell) for cell in row)
            for row in confusion_matrix(truth, predicted, labels=labels)
        ),
        report={
            name: {k: float(v) for k, v in scores.items()}
            for name, scores in report.items()
        },
    )
