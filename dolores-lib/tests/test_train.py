from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from dolores.data import Dataset
from dolores.experiment import TrainingResult, train
from dolores.inference import load_model, read_manifest

FEATURES = ["f0", "f1", "f2"]


@pytest.fixture
def result(splits: tuple[Dataset, Dataset]) -> TrainingResult:
    train_ds, test_ds = splits
    return train(LogisticRegression(max_iter=1000), train_ds, test_ds)


def test_train_returns_a_fitted_estimator(result: TrainingResult) -> None:
    assert hasattr(result.estimator, "classes_")


def test_train_does_not_mutate_the_datasets(
    splits: tuple[Dataset, Dataset],
) -> None:
    train_ds, test_ds = splits
    before = (train_ds.frame.copy(), test_ds.frame.copy())

    train(LogisticRegression(max_iter=1000), train_ds, test_ds)

    assert train_ds.frame.equals(before[0])
    assert test_ds.frame.equals(before[1])


def test_train_scores_the_test_split_not_the_training_one(
    splits: tuple[Dataset, Dataset], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scoring on train produces beautiful, meaningless numbers. Pin the split."""
    train_ds, test_ds = splits
    seen: dict[str, Any] = {}

    import dolores.experiment._train as module

    real = module.evaluate

    def spy(y_true: Any, y_pred: Any) -> Any:
        seen["y_true"] = list(y_true)
        return real(y_true, y_pred)

    monkeypatch.setattr(module, "evaluate", spy)

    train(LogisticRegression(max_iter=1000), train_ds, test_ds)

    assert seen["y_true"] == list(test_ds.y)
    assert len(seen["y_true"]) == 6


def test_train_result_name_defaults_to_the_estimator_class(
    result: TrainingResult,
) -> None:
    assert result.name == "LogisticRegression"


def test_train_result_honours_an_explicit_name(
    splits: tuple[Dataset, Dataset],
) -> None:
    train_ds, test_ds = splits

    result = train(LogisticRegression(max_iter=1000), train_ds, test_ds, name="logreg")

    assert result.name == "logreg"


def test_train_result_captures_params(result: TrainingResult) -> None:
    assert result.params["max_iter"] == 1000


def test_train_result_features_are_in_dataset_order(result: TrainingResult) -> None:
    assert result.features == ("f0", "f1", "f2")


def test_train_result_carries_the_target_name(result: TrainingResult) -> None:
    assert result.target == "label"


def test_train_is_reproducible(splits: tuple[Dataset, Dataset]) -> None:
    train_ds, test_ds = splits

    first = train(LogisticRegression(max_iter=1000), train_ds, test_ds)
    second = train(LogisticRegression(max_iter=1000), train_ds, test_ds)

    assert first.metrics.scalars() == second.metrics.scalars()


def test_train_rejects_a_non_estimator(splits: tuple[Dataset, Dataset]) -> None:
    train_ds, test_ds = splits

    with pytest.raises(TypeError, match="predict"):
        train(object(), train_ds, test_ds)  # type: ignore[arg-type]


def test_train_rejects_mismatched_feature_sets(
    splits: tuple[Dataset, Dataset], frame: pd.DataFrame
) -> None:
    train_ds, _ = splits
    other = Dataset(frame=frame.drop(columns=["f2"]), target="label")

    with pytest.raises(ValueError, match="f2"):
        train(LogisticRegression(max_iter=1000), train_ds, other)


def test_train_rejects_mismatched_targets(
    splits: tuple[Dataset, Dataset], frame: pd.DataFrame
) -> None:
    train_ds, _ = splits
    renamed = frame.rename(columns={"label": "species"})

    with pytest.raises(ValueError, match="species"):
        train(
            LogisticRegression(max_iter=1000),
            train_ds,
            Dataset(frame=renamed, target="species"),
        )


def test_save_bundle_roundtrip(
    result: TrainingResult, tmp_path: Path, payload: dict[str, float]
) -> None:
    """The seam between the two halves of the library."""
    path = result.save_bundle(tmp_path / "model.zip")

    prediction = load_model(path).predict(payload)

    assert prediction.label in result.metrics.labels


def test_save_bundle_embeds_the_metrics(result: TrainingResult, tmp_path: Path) -> None:
    path = result.save_bundle(tmp_path / "model.zip")

    assert read_manifest(path).metrics["f1"] == pytest.approx(result.metrics.f1)


def test_save_bundle_records_the_dataset_target(
    result: TrainingResult, tmp_path: Path
) -> None:
    path = result.save_bundle(tmp_path / "model.zip")

    assert read_manifest(path).target.name == "label"
    assert read_manifest(path).feature_names == FEATURES
