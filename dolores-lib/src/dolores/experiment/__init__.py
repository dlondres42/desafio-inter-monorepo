"""Candidate estimators in, fitted models and metrics out.

The unit of work is a single candidate fitted, scored and reported — which is
also exactly one MLflow run. Training and evaluation live here together because
splitting them would make the caller wire up what this package exists to hand
them already assembled.

Holds the estimator registry (an internal detail, not a public abstraction) and
``metrics``, a pure module of functions over ``y_true``/``y_pred``. Metrics stay
pure and importable on their own so that scoring a model this package did not
train remains possible.

Runs without :mod:`dolores.tracking` installed.
"""
