from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from dolores.data import Dataset, DatasetError, load_dataset


def test_load_dataset_reads_csv(csv_path: Path) -> None:
    ds = load_dataset(csv_path, target="label", drop=["Id"])

    assert ds.frame.shape == (30, 4)
    assert list(ds.frame.columns) == ["f0", "f1", "f2", "label"]


def test_load_dataset_drops_requested_columns(csv_path: Path) -> None:
    ds = load_dataset(csv_path, target="label", drop=["Id"])

    assert "Id" not in ds.frame.columns


def test_load_dataset_raises_when_target_missing(csv_path: Path) -> None:
    with pytest.raises(DatasetError) as excinfo:
        load_dataset(csv_path, target="nope")

    message = str(excinfo.value)
    assert "nope" in message
    assert "label" in message, "the error should name the columns that do exist"


def test_load_dataset_raises_when_drop_column_missing(csv_path: Path) -> None:
    with pytest.raises(DatasetError, match="ghost"):
        load_dataset(csv_path, target="label", drop=["ghost"])


def test_load_dataset_raises_when_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "absent.csv", target="label")


def test_load_dataset_rejects_unsupported_suffix(tmp_path: Path) -> None:
    path = tmp_path / "toy.txt"
    path.write_text("f0,label\n1.0,alpha\n")

    with pytest.raises(DatasetError, match=r"\.txt"):
        load_dataset(path, target="label")


def test_dataset_rejects_target_missing_from_frame(frame: pd.DataFrame) -> None:
    with pytest.raises(DatasetError, match="nope"):
        Dataset(frame=frame, target="nope")


def test_x_excludes_target(dataset: Dataset) -> None:
    assert dataset.target not in dataset.X.columns


def test_x_preserves_column_order(dataset: Dataset) -> None:
    assert list(dataset.X.columns) == ["f0", "f1", "f2"]


def test_y_is_the_target_column(dataset: Dataset) -> None:
    assert dataset.y.name == "label"
    assert len(dataset.y) == 30


def test_features_excludes_target(dataset: Dataset) -> None:
    assert dataset.features == ["f0", "f1", "f2"]


def test_dataset_is_frozen(dataset: Dataset) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        dataset.target = "f0"  # type: ignore[misc]


def test_len_is_row_count(dataset: Dataset) -> None:
    assert len(dataset) == 30
