"""The ``Dataset`` value object and the one function that builds it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetError(ValueError):
    """A frame, a path, or a split parameter is not usable."""


@dataclass(frozen=True, slots=True)
class Dataset:
    """A dataframe that knows which of its columns is the target.

    Pairing the two is the whole point: every downstream step — splitting,
    fitting, deriving a bundle manifest — needs to know which column is the
    label, and threading that name alongside a bare frame is how it ends up
    wrong somewhere.

    The target is validated on construction, so the invariant holds for split
    children too, and never needs rechecking.

    Attributes:
        frame: The data, target column included.
        target: Name of the target column.

    Raises:
        DatasetError: If ``target`` is not a column of ``frame``.
    """

    frame: pd.DataFrame
    target: str

    def __post_init__(self) -> None:
        if self.target not in self.frame.columns:
            available = ", ".join(str(column) for column in self.frame.columns)
            raise DatasetError(
                f"target column {self.target!r} is not in the frame; "
                f"available columns: {available}"
            )

    @property
    def X(self) -> pd.DataFrame:
        """The feature columns, in frame order."""
        return self.frame.drop(columns=[self.target])

    @property
    def y(self) -> pd.Series:
        """The target column."""
        return self.frame[self.target]

    @property
    def features(self) -> list[str]:
        """Feature column names, in the order estimators will receive them."""
        return [str(column) for column in self.frame.columns if column != self.target]

    def split(
        self,
        *,
        test_size: float = 0.2,
        seed: int = 42,
        stratify: bool = True,
    ) -> tuple[Dataset, Dataset]:
        """Partition into train and test datasets.

        The row index is preserved rather than reset, so a prediction can be
        traced back to the row it came from.

        Args:
            test_size: Fraction of rows held out, strictly between 0 and 1.
            seed: Passed to scikit-learn as ``random_state``. The default is
                fixed so that splits are reproducible unless asked otherwise.
            stratify: Preserve the target's class proportions in both halves.

        Returns:
            The train and test datasets, both carrying this dataset's target.

        Raises:
            DatasetError: If ``test_size`` is not strictly between 0 and 1.
        """
        if not 0.0 < test_size < 1.0:
            raise DatasetError(
                f"test_size must be strictly between 0 and 1, got {test_size}"
            )

        train_frame, test_frame = cast(
            tuple[pd.DataFrame, pd.DataFrame],
            train_test_split(
                self.frame,
                test_size=test_size,
                random_state=seed,
                stratify=self.y if stratify else None,
            ),
        )
        return (
            Dataset(frame=train_frame, target=self.target),
            Dataset(frame=test_frame, target=self.target),
        )

    def __len__(self) -> int:
        return len(self.frame)


def load_dataset(
    path: Path | str,
    *,
    target: str,
    drop: Sequence[str] = (),
) -> Dataset:
    """Read a CSV into a :class:`Dataset`.

    Args:
        path: Path to a ``.csv`` file.
        target: Name of the target column.
        drop: Columns to discard on the way in — identifiers and other
            bookkeeping the estimator must never see.

    Returns:
        The loaded dataset.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        DatasetError: If the suffix is not ``.csv``, if any column named in
            ``drop`` is absent, or if ``target`` is not a column.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such dataset: {path}")
    if path.suffix.lower() != ".csv":
        raise DatasetError(
            f"unsupported dataset format {path.suffix!r}; only .csv is supported"
        )

    frame = pd.read_csv(path)

    # A misspelled drop is a silent bug: the column stays, and the estimator is
    # fit on an identifier. Refuse instead.
    missing = [column for column in drop if column not in frame.columns]
    if missing:
        raise DatasetError(
            f"cannot drop {', '.join(repr(column) for column in missing)}: "
            f"not in {path.name}"
        )
    if drop:
        frame = frame.drop(columns=list(drop))

    return Dataset(frame=frame, target=target)
