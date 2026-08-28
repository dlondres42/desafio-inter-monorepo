from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from dolores.experiment.metrics import ClassificationMetrics, evaluate

# Deliberately asymmetric: every row of the confusion matrix has a different
# shape, so a transposed matrix cannot accidentally pass.
Y_TRUE = ["a", "a", "a", "b", "b", "c"]
Y_PRED = ["a", "b", "c", "b", "b", "a"]


@pytest.fixture
def metrics() -> ClassificationMetrics:
    return evaluate(Y_TRUE, Y_PRED)


def test_evaluate_perfect_predictions() -> None:
    m = evaluate(Y_TRUE, Y_TRUE)

    assert (m.accuracy, m.precision, m.recall, m.f1) == (1.0, 1.0, 1.0, 1.0)


def test_evaluate_known_accuracy(metrics: ClassificationMetrics) -> None:
    # a->a, b->b, b->b are the three correct calls out of six.
    assert metrics.accuracy == pytest.approx(0.5)


def test_evaluate_matches_sklearn_macro_averages(
    metrics: ClassificationMetrics,
) -> None:
    kwargs: dict[str, Any] = {"average": "macro", "zero_division": 0}

    assert metrics.precision == pytest.approx(precision_score(Y_TRUE, Y_PRED, **kwargs))
    assert metrics.recall == pytest.approx(recall_score(Y_TRUE, Y_PRED, **kwargs))
    assert metrics.f1 == pytest.approx(f1_score(Y_TRUE, Y_PRED, **kwargs))


def test_confusion_matrix_rows_are_true_classes(
    metrics: ClassificationMetrics,
) -> None:
    """Rows are truth, columns are predictions. The transpose is a real bug."""
    assert metrics.confusion_matrix == (
        (1, 1, 1),  # true a: one each of a, b, c
        (0, 2, 0),  # true b: both predicted b
        (1, 0, 0),  # true c: predicted a
    )


def test_confusion_matrix_labels_are_sorted_classes(
    metrics: ClassificationMetrics,
) -> None:
    assert metrics.labels == ("a", "b", "c")


def test_confusion_matrix_entries_are_builtin_ints(
    metrics: ClassificationMetrics,
) -> None:
    assert all(type(cell) is int for row in metrics.confusion_matrix for cell in row)


def test_scalars_returns_exactly_the_four_logged_metrics(
    metrics: ClassificationMetrics,
) -> None:
    """This dict is the MLflow log_metrics payload; it must stay flat."""
    scalars = metrics.scalars()

    assert set(scalars) == {"accuracy", "precision", "recall", "f1"}
    assert all(type(v) is float for v in scalars.values())


def test_to_dict_contains_only_builtin_types(metrics: ClassificationMetrics) -> None:
    """numpy scalars serialize fine until they reach json.dumps. Catch them here."""

    def walk(value: Any) -> None:
        assert type(value) in {bool, int, float, str, list, dict}, (
            f"{value!r} is {type(value)}, not a builtin"
        )
        if isinstance(value, dict):
            for key, item in value.items():
                assert type(key) is str
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(metrics.to_dict())


def test_to_dict_is_json_serializable(metrics: ClassificationMetrics) -> None:
    json.dumps(metrics.to_dict())


def test_write_json_roundtrips(tmp_path: Path, metrics: ClassificationMetrics) -> None:
    path = metrics.write_json(tmp_path / "metrics.json")

    assert path.exists()
    assert json.loads(path.read_text()) == metrics.to_dict()


def test_report_includes_every_class(metrics: ClassificationMetrics) -> None:
    assert {"a", "b", "c"} <= set(metrics.report)


def test_report_excludes_the_scalar_accuracy_entry(
    metrics: ClassificationMetrics,
) -> None:
    """sklearn puts a bare float under 'accuracy'; it lives on the dataclass instead."""
    assert "accuracy" not in metrics.report
    assert all(isinstance(v, dict) for v in metrics.report.values())


def test_evaluate_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError):
        evaluate(["a", "b"], ["a"])


def test_evaluate_emits_no_warning_for_unpredicted_class(
    recwarn: pytest.WarningsRecorder,
) -> None:
    """A class never predicted is 0/0 precision. zero_division=0, not a warning."""
    m = evaluate(["a", "a", "b", "b", "c", "c"], ["a", "a", "b", "b", "b", "b"])

    assert m.labels == ("a", "b", "c")
    assert [w.category.__name__ for w in recwarn] == []


def test_metrics_are_frozen(metrics: ClassificationMetrics) -> None:
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        metrics.accuracy = 0.0  # type: ignore[misc]
