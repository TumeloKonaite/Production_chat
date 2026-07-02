from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.infrastructure.tracking.setup import TrackingSetupError, configure_tracking_backend


class MLflowClient:
    def __init__(
        self,
        *,
        tracking_uri: str | None,
        enabled: bool,
        enable_dagshub_tracking: bool = False,
        dagshub_repo_owner: str | None = None,
        dagshub_repo_name: str | None = None,
        dagshub_token: str | None = None,
    ) -> None:
        self._tracking_uri = tracking_uri
        self._enabled = enabled
        self._enable_dagshub_tracking = enable_dagshub_tracking
        self._dagshub_repo_owner = dagshub_repo_owner
        self._dagshub_repo_name = dagshub_repo_name
        self._dagshub_token = dagshub_token
        self._mlflow: Any | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_experiment(self, experiment_name: str) -> bool:
        if not self._enabled:
            return False

        mlflow = self._load_mlflow()
        if mlflow is None:
            return False

        try:
            configure_tracking_backend(
                mlflow=mlflow,
                tracking_uri=self._tracking_uri,
                enable_dagshub_tracking=self._enable_dagshub_tracking,
                dagshub_repo_owner=self._dagshub_repo_owner,
                dagshub_repo_name=self._dagshub_repo_name,
                dagshub_token=self._dagshub_token,
            )
            mlflow.set_experiment(experiment_name)
            return True
        except TrackingSetupError:
            raise
        except Exception:
            self._enabled = False
            return False

    @contextmanager
    def start_run(self, run_name: str) -> Iterator[object | None]:
        if not self._enabled:
            yield None
            return

        mlflow = self._load_mlflow()
        if mlflow is None:
            yield None
            return

        try:
            run_context = mlflow.start_run(run_name=run_name)
        except Exception:
            self._enabled = False
            yield None
            return

        with run_context as run:
            yield run

    def log_params(self, params: dict[str, object]) -> None:
        if not self._enabled:
            return

        mlflow = self._load_mlflow()
        if mlflow is None:
            return

        try:
            mlflow.log_params(self._normalize_mapping(params))
        except Exception:
            self._enabled = False

    def log_metrics(self, metrics: dict[str, float | int]) -> None:
        if not self._enabled:
            return

        mlflow = self._load_mlflow()
        if mlflow is None:
            return

        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, float(value))
        except Exception:
            self._enabled = False

    def log_artifact(self, artifact_path: Path) -> None:
        if not self._enabled:
            return

        mlflow = self._load_mlflow()
        if mlflow is None:
            return

        try:
            mlflow.log_artifact(str(artifact_path))
        except Exception:
            self._enabled = False

    def _load_mlflow(self) -> Any | None:
        if self._mlflow is not None:
            return self._mlflow

        try:
            import mlflow
        except ImportError:
            self._enabled = False
            return None

        self._mlflow = mlflow
        return self._mlflow

    def _normalize_mapping(self, mapping: dict[str, object]) -> dict[str, str | float | int]:
        normalized: dict[str, str | float | int] = {}
        for key, value in mapping.items():
            if value is None:
                normalized[key] = "null"
            elif isinstance(value, (str, float, int)):
                normalized[key] = value
            else:
                normalized[key] = str(value)
        return normalized
