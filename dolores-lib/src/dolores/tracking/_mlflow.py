"""Ship one :class:`~dolores.experiment.TrainingResult` to MLflow as one run."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from dolores.experiment import TrainingResult
from dolores.inference import read_manifest


def _import_mlflow() -> Any:
    """Import MLflow, or explain how to get it.

    Imported here rather than at module scope so that ``dolores.tracking`` can be
    imported — and the rest of the library used — without the extra installed.
    """
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - exercised via sys.modules
        raise ImportError(
            "dolores.tracking requires MLflow, which ships as an optional extra. "
            "Install it with: pip install 'dolores-lib[tracking]'"
        ) from exc
    return mlflow


def log_run(
    result: TrainingResult,
    *,
    experiment: str,
    bundle: Path | None = None,
    tracking_uri: str | None = None,
) -> str:
    """Log a training result to MLflow as a single run.

    The four headline scores go to ``log_metrics``, which only accepts flat
    numbers; the confusion matrix and per-class report go alongside as a
    ``metrics.json`` artifact.

    When ``bundle`` is given it is logged as a plain artifact and its ``model_id``
    becomes a run tag. That is deliberate in place of ``mlflow.sklearn.log_model``:
    it keeps this module clear of ``mlflow.sklearn``, and it means the bytes MLflow
    shows are the same bytes the inference server loads, rather than a second,
    competing serialization of the same estimator.

    Args:
        result: The fitted, scored candidate to record.
        experiment: MLflow experiment name. Created if absent.
        bundle: Optional path to the bundle this result produced.
        tracking_uri: Overrides ``MLFLOW_TRACKING_URI``. When neither is set,
            MLflow's own default is left alone — the URI is never hardcoded here.

    Returns:
        The MLflow run id.

    Raises:
        ImportError: If MLflow is not installed.
    """
    mlflow = _import_mlflow()

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=result.name) as run:
        mlflow.log_params(result.params)
        mlflow.log_metrics(result.metrics.scalars())

        with tempfile.TemporaryDirectory() as staging:
            mlflow.log_artifact(
                str(result.metrics.write_json(Path(staging) / "metrics.json"))
            )

        if bundle is not None:
            mlflow.set_tag("model_id", read_manifest(bundle).model_id)
            mlflow.log_artifact(str(bundle))

        run_id: str = run.info.run_id
    return run_id
