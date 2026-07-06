from __future__ import annotations

import json
from typing import Any

from app.services.evals.eval_run_service import (
    EvalArtifactNotFoundError,
    EvalRunService,
)


class EvalArtifactReader:
    def __init__(self, run_service: EvalRunService) -> None:
        self._run_service = run_service

    def read_summary(self, run_id: str) -> dict[str, Any]:
        summary_path = self._run_service.resolve_run_path(run_id, "summary.json")
        if not summary_path.exists():
            raise EvalArtifactNotFoundError(f"Summary not found for run: {run_id}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def read_failures(self, run_id: str) -> dict[str, Any]:
        failures_path = self._run_service.resolve_run_path(run_id, "failures.json")
        if not failures_path.exists():
            raise EvalArtifactNotFoundError(f"Failures not found for run: {run_id}")
        return json.loads(failures_path.read_text(encoding="utf-8"))
