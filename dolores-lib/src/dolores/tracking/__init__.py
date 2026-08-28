"""Optional MLflow integration.

Ships behind the ``dolores-lib[tracking]`` extra and imports MLflow lazily, so
the library — and the inference server's container image — works without it.
Mandatory for the data scientist, absent from the serving path.

This package imports :mod:`dolores.experiment` and :mod:`dolores.inference`, and
neither imports it back. Nothing in the library depends on tracking; logging is
always an explicit call the caller makes, never a side effect of training.

The tracking URI is always read from the environment. It is never hardcoded here.
"""

from dolores.tracking._mlflow import log_run

__all__ = ["log_run"]
