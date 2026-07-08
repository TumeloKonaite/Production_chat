from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.dependencies.common_dependencies import get_app_settings, get_db_session
from app.config import Settings
from app.infrastructure.cache import build_cache_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


async def _check_redis_readiness(settings: Settings) -> str | None:
    if not settings.redis_healthcheck_enabled:
        return None

    if not settings.upstash_redis_configured:
        return "misconfigured"

    try:
        cache_client = build_cache_client(settings)
        await cache_client.get("__upstash_healthcheck__")
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
    except Exception:
        logger.warning("Database readiness check failed.", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "degraded", "database": "unavailable"},
        )

    response: dict[str, str] = {"status": "ok", "database": "ok"}
    redis_status = await _check_redis_readiness(settings)
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
