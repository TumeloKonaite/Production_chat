from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


class TrackingSetupError(RuntimeError):
    pass


def configure_tracking_backend(
    *,
    mlflow: Any,
    tracking_uri: str | None,
    tracking_username: str | None,
    tracking_password: str | None,
    enable_dagshub_tracking: bool,
    dagshub_repo_owner: str | None,
    dagshub_repo_name: str | None,
    dagshub_token: str | None,
) -> None:
    if enable_dagshub_tracking:
        _initialize_dagshub(
            repo_owner=dagshub_repo_owner,
            repo_name=dagshub_repo_name,
            token=dagshub_token,
        )
        return

    if tracking_username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = tracking_username
    if tracking_password:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = tracking_password
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def create_experiment_tracker(settings: Settings, experiment_name: str):
    from app.infrastructure.tracking.experiment_tracker import ExperimentTracker
    from app.infrastructure.tracking.mlflow_client import MLflowClient

    if settings.enable_dagshub_tracking and not settings.enable_mlflow_tracking:
        logger.warning(
            "Experiment tracking disabled: DagsHub tracking requires "
            "ENABLE_MLFLOW_TRACKING=true because MLflow remains the logging API."
        )
        return ExperimentTracker(
            MLflowClient(
                tracking_uri=None,
                tracking_username=None,
                tracking_password=None,
                enabled=False,
            ),
            experiment_name,
        )

    return ExperimentTracker(
        MLflowClient(
            tracking_uri=settings.mlflow_tracking_uri,
            tracking_username=settings.mlflow_tracking_username,
            tracking_password=settings.mlflow_tracking_password,
            enabled=settings.enable_mlflow_tracking,
            enable_dagshub_tracking=settings.enable_dagshub_tracking,
            dagshub_repo_owner=settings.dagshub_repo_owner,
            dagshub_repo_name=settings.dagshub_repo_name,
            dagshub_token=settings.dagshub_token,
        ),
        experiment_name=experiment_name,
    )


def _initialize_dagshub(
    *,
    repo_owner: str | None,
    repo_name: str | None,
    token: str | None,
) -> None:
    if not repo_owner or not repo_owner.strip():
        raise TrackingSetupError(
            "DagsHub tracking is enabled but DAGSHUB_REPO_OWNER is not set."
        )
    if not repo_name or not repo_name.strip():
        raise TrackingSetupError(
            "DagsHub tracking is enabled but DAGSHUB_REPO_NAME is not set."
        )

    try:
        import dagshub
    except ImportError as exc:
        raise TrackingSetupError(
            "DagsHub tracking is enabled but the dagshub package is not installed."
        ) from exc

    if token and not os.getenv("DAGSHUB_USER_TOKEN"):
        os.environ["DAGSHUB_USER_TOKEN"] = token

    try:
        dagshub.init(
            repo_owner=repo_owner.strip(),
            repo_name=repo_name.strip(),
            mlflow=True,
        )
    except Exception as exc:
        raise TrackingSetupError(
            "Unable to initialize DagsHub tracking. Check the repo settings and credentials."
        ) from exc
