from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dolores.experiment import TrainingResult
from dolores.tracking import log_run

# Blocks `import mlflow` in a subprocess, so the "works without the extra" claim
# is tested against absence rather than against a mock of absence.
BLOCKER = """
import sys
class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("mlflow is not installed")
        return None
sys.meta_path.insert(0, Blocker())
"""


class FakeMlflow:
    """Records what log_run asks of MLflow, without needing a server."""

    def __init__(self) -> None:
        self.tracking_uri: str | None = None
        self.experiment: str | None = None
        self.run_name: str | None = None
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        self.tags: dict[str, str] = {}
        self.artifacts: list[str] = []

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment = name

    @contextmanager
    def start_run(self, run_name: str | None = None) -> Iterator[Any]:
        self.run_name = run_name
        yield SimpleNamespace(info=SimpleNamespace(run_id="run-abc123"))

    def log_params(self, params: Mapping[str, Any]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: Mapping[str, Any]) -> None:
        self.metrics.update(metrics)

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def log_artifact(self, path: str) -> None:
        self.artifacts.append(Path(path).name)


@pytest.fixture
def mlflow(monkeypatch: pytest.MonkeyPatch) -> FakeMlflow:
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    return fake


def test_experiment_imports_without_mlflow() -> None:
    """The extra is optional in fact, not merely in the docstring."""
    result = subprocess.run(
        [sys.executable, "-c", BLOCKER + "import dolores.experiment"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_inference_imports_without_mlflow() -> None:
    """The serving path especially: MLflow never reaches the container image."""
    result = subprocess.run(
        [sys.executable, "-c", BLOCKER + "import dolores.inference"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_log_run_raises_a_helpful_error_when_mlflow_is_missing(
    monkeypatch: pytest.MonkeyPatch, training_result: TrainingResult
) -> None:
    """Setting a module to None in sys.modules makes `import` of it fail."""
    monkeypatch.setitem(sys.modules, "mlflow", None)

    with pytest.raises(ImportError, match=r"dolores-lib\[tracking\]"):
        log_run(training_result, experiment="x")


def test_log_run_sets_tracking_uri_from_env(
    mlflow: FakeMlflow, monkeypatch: pytest.MonkeyPatch, training_result: TrainingResult
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://example:5000")

    log_run(training_result, experiment="iris")

    assert mlflow.tracking_uri == "http://example:5000"


def test_log_run_does_not_set_uri_when_env_is_unset(
    mlflow: FakeMlflow, training_result: TrainingResult
) -> None:
    """Leave MLflow's own default alone rather than inventing one."""
    log_run(training_result, experiment="iris")

    assert mlflow.tracking_uri is None


def test_log_run_prefers_an_explicit_uri_over_the_environment(
    mlflow: FakeMlflow, monkeypatch: pytest.MonkeyPatch, training_result: TrainingResult
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://from-env:5000")

    log_run(training_result, experiment="iris", tracking_uri="http://explicit:5000")

    assert mlflow.tracking_uri == "http://explicit:5000"


def test_log_run_logs_params_and_scalar_metrics(
    mlflow: FakeMlflow, training_result: TrainingResult
) -> None:
    log_run(training_result, experiment="iris")

    assert mlflow.params["max_iter"] == 1000
    assert set(mlflow.metrics) == {"accuracy", "precision", "recall", "f1"}


def test_log_run_passes_only_floats_to_log_metrics(
    mlflow: FakeMlflow, training_result: TrainingResult
) -> None:
    """MLflow rejects nested structures; the matrix and report go as an artifact."""
    log_run(training_result, experiment="iris")

    assert all(type(value) is float for value in mlflow.metrics.values())


def test_log_run_writes_the_full_metrics_as_an_artifact(
    mlflow: FakeMlflow, training_result: TrainingResult
) -> None:
    log_run(training_result, experiment="iris")

    assert "metrics.json" in mlflow.artifacts


def test_log_run_uses_the_experiment_and_run_name(
    mlflow: FakeMlflow, training_result: TrainingResult
) -> None:
    log_run(training_result, experiment="iris-baseline")

    assert mlflow.experiment == "iris-baseline"
    assert mlflow.run_name == "logreg"


def test_log_run_returns_the_run_id(
    mlflow: FakeMlflow, training_result: TrainingResult
) -> None:
    assert log_run(training_result, experiment="iris") == "run-abc123"


def test_log_run_logs_the_bundle_and_tags_its_model_id(
    mlflow: FakeMlflow, training_result: TrainingResult, tmp_path: Path
) -> None:
    """The traceability link: this run produced the artifact the server loads."""
    from dolores.inference import read_manifest

    bundle = training_result.save_bundle(tmp_path / "model.zip")

    log_run(training_result, experiment="iris", bundle=bundle)

    assert "model.zip" in mlflow.artifacts
    assert mlflow.tags["model_id"] == read_manifest(bundle).model_id


def test_log_run_without_a_bundle_tags_nothing(
    mlflow: FakeMlflow, training_result: TrainingResult
) -> None:
    log_run(training_result, experiment="iris")

    assert "model_id" not in mlflow.tags


@pytest.mark.integration
def test_log_run_against_a_real_sqlite_store(
    training_result: TrainingResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One real MLflow round trip. Without it, everything above is mock theatre.

    sqlite rather than the filesystem store: MLflow 3.15 refuses the latter
    outright. It is also what `make mlflow-start` runs, so this exercises the
    backend the notebook actually talks to.
    """
    from mlflow.tracking import MlflowClient

    monkeypatch.chdir(tmp_path)  # artifacts land here, not in the repo
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_id = log_run(training_result, experiment="iris-it", tracking_uri=uri)

    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)
    assert run.data.metrics["f1"] == pytest.approx(training_result.metrics.f1)
    assert run.data.params["max_iter"] == "1000"
    assert run.info.run_name == "logreg"
