from __future__ import annotations

from importlib import import_module
import logging

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.dependencies.common_dependencies import get_app_settings, get_db_session
from app.config import Settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


def _check_redis_readiness(settings: Settings) -> str | None:
    if not settings.redis_healthcheck_enabled:
        return None

    redis_url = settings.resolved_redis_url
    if redis_url is None:
        return "misconfigured"

    try:
        redis_module = import_module("redis")
        redis_client = redis_module.Redis.from_url(
            redis_url,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        try:
            redis_client.ping()
        finally:
            close = getattr(redis_client, "close", None)
            if callable(close):
                close()
    except Exception:
        logger.warning("Redis readiness check failed.", exc_info=True)
        return "unavailable"

    return "ok"


@router.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", response_model=None)
async def readiness_check(
    settings: Settings = Depends(get_app_settings),
    db_session: Session = Depends(get_db_session),
) -> dict[str, str]:
    try:
        db_session.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "degraded", "database": "unavailable"},
        )

    response: dict[str, str] = {"status": "ok", "database": "ok"}
    redis_status = _check_redis_readiness(settings)
    if redis_status is None:
        return response
    if redis_status != "ok":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "degraded",
                "database": "ok",
                "redis": redis_status,
            },
        )

    response["redis"] = "ok"
    return response
