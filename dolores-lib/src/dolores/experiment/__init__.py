"""Candidate estimators in, fitted models and metrics out.

The unit of work is a single candidate fitted, scored and reported — which is
also exactly one MLflow run. Training and evaluation live here together because
splitting them would make the caller wire up what this package exists to hand
them already assembled.

``train`` takes any object that can fit and predict, so the library never holds a
list of blessed estimators; choosing them is the caller's job. It hands back a
``TrainingResult`` that knows how to write itself out as a servable bundle.

``metrics`` is a public module rather than a private one: pure functions over
``y_true``/``y_pred``, importable on their own, so that scoring a model this
package did not train remains possible.

Imports :mod:`dolores.data` and :mod:`dolores.inference`, and neither imports this
package back. Runs without :mod:`dolores.tracking` — or MLflow — installed.
"""

from dolores.experiment._train import Estimator, TrainingResult, train

__all__ = ["Estimator", "TrainingResult", "train"]
