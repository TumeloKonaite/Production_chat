from __future__ import annotations

from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    HTTPException,
    Query,
    Response,
    status,
)

from app.api.dependencies.common_dependencies import get_app_settings
from app.api.evals.schemas import (
    EvalRunListItem,
    EvalRunListResponse,
    EvalRunQueuedResponse,
    EvalRunStatusResponse,
    GenerationEvalRequest,
    MatrixDryRunResponse,
    MatrixEvalRequest,
    RagEvalRequest,
    RetrievalEvalRequest,
)
from app.config import Settings
from app.services.evals.eval_artifact_reader import EvalArtifactReader
from app.services.evals.eval_job_runner import EvalJobRunner
from app.services.evals.eval_run_service import (
    EvalArtifactNotFoundError,
    EvalRunNotFoundError,
    EvalRunService,
    EvalRunValidationError,
)

router = APIRouter(prefix="/api/evals", tags=["evals"])


def get_eval_run_service() -> EvalRunService:
    return EvalRunService()


def get_eval_artifact_reader(
    run_service: EvalRunService = Depends(get_eval_run_service),
) -> EvalArtifactReader:
    return EvalArtifactReader(run_service)


def get_eval_job_runner(
    run_service: EvalRunService = Depends(get_eval_run_service),
) -> EvalJobRunner:
    return EvalJobRunner(run_service=run_service)


def require_eval_admin_token(
    settings: Settings = Depends(get_app_settings),
    eval_admin_token: Annotated[str | None, Header(alias="X-Eval-Admin-Token")] = None,
) -> None:
    from app.api.evals_legacy_auth import validate_eval_admin_token

    validate_eval_admin_token(
        provided_token=eval_admin_token,
        expected_token=settings.eval_admin_token,
    )


@router.post(
    "/retrieval",
    response_model=EvalRunQueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_eval_admin_token)],
)
def create_retrieval_eval_run(
    payload: RetrievalEvalRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_app_settings),
    run_service: EvalRunService = Depends(get_eval_run_service),
    job_runner: EvalJobRunner = Depends(get_eval_job_runner),
) -> EvalRunQueuedResponse:
    record = run_service.create_run(mode="retrieval", config=payload.model_dump())
    background_tasks.add_task(
        job_runner.run_retrieval_job,
        run_id=record.run_id,
        payload=payload.model_dump(),
        settings=settings,
    )
    return _build_queued_response(run_id=record.run_id, mode="retrieval")


@router.post(
    "/generation",
    response_model=EvalRunQueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_eval_admin_token)],
)
def create_generation_eval_run(
    payload: GenerationEvalRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_app_settings),
    run_service: EvalRunService = Depends(get_eval_run_service),
    job_runner: EvalJobRunner = Depends(get_eval_job_runner),
) -> EvalRunQueuedResponse:
    record = run_service.create_run(mode="generation", config=payload.model_dump())
    background_tasks.add_task(
        job_runner.run_generation_job,
        run_id=record.run_id,
        payload=payload.model_dump(),
        settings=settings,
    )
    return _build_queued_response(run_id=record.run_id, mode="generation")


@router.post(
    "/rag",
    response_model=EvalRunQueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_eval_admin_token)],
)
def create_rag_eval_run(
    payload: RagEvalRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_app_settings),
    run_service: EvalRunService = Depends(get_eval_run_service),
    job_runner: EvalJobRunner = Depends(get_eval_job_runner),
) -> EvalRunQueuedResponse:
    record = run_service.create_run(mode="rag", config=payload.model_dump())
    background_tasks.add_task(
        job_runner.run_rag_job,
        run_id=record.run_id,
        payload=payload.model_dump(),
        settings=settings,
    )
    return _build_queued_response(run_id=record.run_id, mode="rag")


@router.post(
    "/matrix",
    response_model=EvalRunQueuedResponse | MatrixDryRunResponse,
    dependencies=[Depends(require_eval_admin_token)],
)
def create_matrix_eval_run(
    payload: MatrixEvalRequest,
    background_tasks: BackgroundTasks,
    response: Response,
    settings: Settings = Depends(get_app_settings),
    run_service: EvalRunService = Depends(get_eval_run_service),
    job_runner: EvalJobRunner = Depends(get_eval_job_runner),
) -> EvalRunQueuedResponse | MatrixDryRunResponse:
    try:
        plan = job_runner.resolve_matrix_plan(
            suite_name=payload.suite,
            dry_run=payload.dry_run,
            confirm_full_run=payload.confirm_full_run,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if payload.dry_run:
        return MatrixDryRunResponse(
            suite=plan.suite.name,
            mode=plan.suite.mode,
            dry_run=True,
            retrieval_combinations=len(plan.retrieval_combinations),
            generation_combinations=len(plan.generation_combinations),
            total_planned_runs=plan.total_planned_runs,
            max_combinations=plan.suite.max_combinations,
            status="ok",
        )

    record = run_service.create_run(
        mode="matrix",
        suite=payload.suite,
        config=payload.model_dump(),
        extra_fields={
            "total_planned_runs": plan.total_planned_runs,
            "successful_runs": 0,
            "failed_runs": 0,
        },
    )
    background_tasks.add_task(
        job_runner.run_matrix_job,
        run_id=record.run_id,
        payload=payload.model_dump(),
        settings=settings,
    )
    response.status_code = status.HTTP_202_ACCEPTED
    return _build_queued_response(run_id=record.run_id, mode="matrix", suite=payload.suite)


@router.get(
    "/runs",
    response_model=EvalRunListResponse,
    dependencies=[Depends(require_eval_admin_token)],
)
def list_eval_runs(
    mode: str | None = Query(default=None, pattern="^(retrieval|generation|rag|matrix)$"),
    limit: int = Query(default=20, ge=1, le=100),
    run_service: EvalRunService = Depends(get_eval_run_service),
) -> EvalRunListResponse:
    return EvalRunListResponse(
        runs=[_build_run_list_item(record) for record in run_service.list_runs(mode=mode, limit=limit)]
    )


@router.get(
    "/runs/{run_id}",
    response_model=EvalRunStatusResponse,
    dependencies=[Depends(require_eval_admin_token)],
)
def get_eval_run_status(
    run_id: str,
    run_service: EvalRunService = Depends(get_eval_run_service),
) -> EvalRunStatusResponse:
    try:
        record = run_service.get_run(run_id)
    except EvalRunValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except EvalRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return _build_run_status_response(record)


@router.get(
    "/runs/{run_id}/summary",
    dependencies=[Depends(require_eval_admin_token)],
)
def get_eval_run_summary(
    run_id: str,
    artifact_reader: EvalArtifactReader = Depends(get_eval_artifact_reader),
) -> dict[str, object]:
    try:
        return artifact_reader.read_summary(run_id)
    except EvalRunValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (EvalRunNotFoundError, EvalArtifactNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/runs/{run_id}/failures",
    dependencies=[Depends(require_eval_admin_token)],
)
def get_eval_run_failures(
    run_id: str,
    artifact_reader: EvalArtifactReader = Depends(get_eval_artifact_reader),
) -> dict[str, object]:
    try:
        return artifact_reader.read_failures(run_id)
    except EvalRunValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (EvalRunNotFoundError, EvalArtifactNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


def _build_queued_response(
    *,
    run_id: str,
    mode: str,
    suite: str | None = None,
) -> EvalRunQueuedResponse:
    return EvalRunQueuedResponse(
        run_id=run_id,
        status="queued",
        mode=mode,
        suite=suite,
        status_url=f"/api/evals/runs/{run_id}",
    )


def _build_run_list_item(record) -> EvalRunListItem:
    return EvalRunListItem(
        run_id=record.run_id,
        mode=record.mode,
        status=record.status,
        suite=record.suite,
        started_at=record.started_at,
        completed_at=record.completed_at,
        total_planned_runs=record.total_planned_runs,
        successful_runs=record.successful_runs,
        failed_runs=record.failed_runs,
    )


def _build_run_status_response(record) -> EvalRunStatusResponse:
    return EvalRunStatusResponse(
        run_id=record.run_id,
        mode=record.mode,
        status=record.status,
        suite=record.suite,
        triggered_by="api",
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        total_planned_runs=record.total_planned_runs,
        successful_runs=record.successful_runs,
        failed_runs=record.failed_runs,
        summary_url=f"/api/evals/runs/{record.run_id}/summary",
        failures_url=f"/api/evals/runs/{record.run_id}/failures",
        artifacts=record.artifacts,
        config=record.config,
    )
