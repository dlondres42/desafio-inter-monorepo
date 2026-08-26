"""Everything between a file on disk and a dataset ready to fit.

Loads tabular data from a path, wraps it in a ``Dataset`` alongside its target
column, and splits it. One call should take a caller from a path to a split
dataset; the format check, dataframe backend and path resolution behind that
stay internal.

Scope is tabular data only. Estimators and metrics are persisted by the packages
that produce and consume them — :mod:`dolores.experiment` and
:mod:`dolores.inference` — rather than through a shared wrapper here.
"""

from dolores.data._dataset import Dataset, DatasetError, load_dataset

__all__ = ["Dataset", "DatasetError", "load_dataset"]
