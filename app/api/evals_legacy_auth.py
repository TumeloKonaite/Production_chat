from __future__ import annotations

import secrets

from fastapi import HTTPException, status


def validate_eval_admin_token(
    *,
    provided_token: str | None,
    expected_token: str | None,
) -> None:
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="EVAL_ADMIN_TOKEN is not configured.",
        )
    if not provided_token or not secrets.compare_digest(provided_token, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid eval admin token.",
        )
