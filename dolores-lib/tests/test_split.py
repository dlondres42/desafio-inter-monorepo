from __future__ import annotations

import pytest

from dolores.data import Dataset, DatasetError


def test_split_sizes(dataset: Dataset) -> None:
    train, test = dataset.split(test_size=0.2, seed=42)

    assert (len(train), len(test)) == (24, 6)
    assert len(train) + len(test) == len(dataset)


def test_split_partitions_without_overlap(dataset: Dataset) -> None:
    train, test = dataset.split()

    train_index, test_index = set(train.frame.index), set(test.frame.index)
    assert train_index & test_index == set()
    assert train_index | test_index == set(dataset.frame.index)


def test_split_is_reproducible_for_same_seed(dataset: Dataset) -> None:
    first, _ = dataset.split(seed=0)
    second, _ = dataset.split(seed=0)

    assert first.frame.equals(second.frame)


def test_split_differs_for_different_seed(dataset: Dataset) -> None:
    first, _ = dataset.split(seed=0)
    second, _ = dataset.split(seed=1)

    assert set(first.frame.index) != set(second.frame.index)


def test_split_stratifies_class_proportions(dataset: Dataset) -> None:
    _, test = dataset.split(test_size=0.2, seed=42)

    assert test.y.value_counts().to_dict() == {"alpha": 2, "beta": 2, "gamma": 2}


def test_split_without_stratify_still_partitions(dataset: Dataset) -> None:
    train, test = dataset.split(test_size=0.2, seed=42, stratify=False)

    assert (len(train), len(test)) == (24, 6)


def test_split_children_carry_target_and_features(dataset: Dataset) -> None:
    for child in dataset.split():
        assert child.target == dataset.target
        assert child.features == dataset.features


def test_split_children_are_datasets(dataset: Dataset) -> None:
    train, test = dataset.split()

    assert isinstance(train, Dataset)
    assert isinstance(test, Dataset)


@pytest.mark.parametrize("test_size", [0.0, 1.0, 1.5, -0.1])
def test_split_rejects_test_size_out_of_range(
    dataset: Dataset, test_size: float
) -> None:
    with pytest.raises(DatasetError, match="test_size"):
        dataset.split(test_size=test_size)


def test_split_does_not_mutate_parent(dataset: Dataset) -> None:
    before = dataset.frame.copy()

    dataset.split()

    assert dataset.frame.equals(before)
