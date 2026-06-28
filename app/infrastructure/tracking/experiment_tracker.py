from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.infrastructure.tracking.mlflow_client import MLflowClient


class ExperimentTracker:
    def __init__(self, client: MLflowClient, experiment_name: str) -> None:
        self._client = client
        self._experiment_name = experiment_name
        self._configured = self._client.set_experiment(experiment_name)

    @property
    def enabled(self) -> bool:
        return self._configured and self._client.enabled

    @contextmanager
    def run(self, run_name: str) -> Iterator[object | None]:
        with self._client.start_run(run_name=run_name) as run:
            yield run

    def log_params(self, params: dict[str, object]) -> None:
        self._client.log_params(params)

    def log_metrics(self, metrics: dict[str, float | int]) -> None:
        self._client.log_metrics(metrics)

    def log_artifact(self, artifact_path: Path) -> None:
        self._client.log_artifact(artifact_path)
