from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

from evals.matrix import DEFAULT_EXPERIMENT_OUTPUT_DIR

RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-:.]+$")
TERMINAL_STATUSES = frozenset({"completed", "completed_with_failures", "failed", "cancelled"})


class EvalRunError(RuntimeError):
    """Base error for eval run state operations."""


class EvalRunNotFoundError(EvalRunError):
    """Raised when a run record cannot be located."""


class EvalRunValidationError(EvalRunError):
    """Raised when a run identifier or path is invalid."""


class EvalArtifactNotFoundError(EvalRunError):
    """Raised when an expected run artifact does not exist."""


@dataclass(frozen=True, slots=True)
class EvalRunRecord:
    run_id: str
    mode: str
    status: str
    suite: str | None
    triggered_by: str
    created_at: str | None
    started_at: str | None
    completed_at: str | None
    total_planned_runs: int | None
    successful_runs: int | None
    failed_runs: int | None
    config: dict[str, Any]
    artifacts: dict[str, str]
    run_dir: Path


class EvalRunService:
    def __init__(self, base_output_dir: Path = DEFAULT_EXPERIMENT_OUTPUT_DIR) -> None:
        self._base_output_dir = base_output_dir.resolve()
        self._base_output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_output_dir(self) -> Path:
        return self._base_output_dir

    def build_run_id(
        self,
        *,
        mode: str,
        suite: str | None = None,
        created_at: datetime | None = None,
    ) -> str:
        timestamp = (created_at or datetime.now().astimezone()).strftime("%Y-%m-%d_%H-%M-%S")
        suffix = self._normalize_run_token(suite if mode == "matrix" and suite else mode)
        candidate = f"{timestamp}_{suffix}"
        attempt = 1
        while (self._base_output_dir / candidate).exists():
            candidate = f"{timestamp}_{suffix}_{attempt:02d}"
            attempt += 1
        return candidate

    def create_run(
        self,
        *,
        mode: str,
        config: dict[str, Any],
        suite: str | None = None,
        run_id: str | None = None,
        created_at: datetime | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> EvalRunRecord:
        created_timestamp = created_at or datetime.now().astimezone()
        resolved_run_id = run_id or self.build_run_id(
            mode=mode,
            suite=suite,
            created_at=created_timestamp,
        )
        self.validate_run_id(resolved_run_id)
        run_dir = self._base_output_dir / resolved_run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        payload: dict[str, Any] = {
            "run_id": resolved_run_id,
            "mode": mode,
            "status": "queued",
            "suite": suite,
            "triggered_by": "api",
            "created_at": created_timestamp.replace(microsecond=0).isoformat(),
            "started_at": None,
            "completed_at": None,
            "total_planned_runs": None,
            "successful_runs": None,
            "failed_runs": None,
            "config": config,
            "artifacts": {},
        }
        if extra_fields:
            payload.update(extra_fields)

        self._write_json(run_dir / "manifest.json", payload)
        self._write_json(
            run_dir / "status.json",
            {
                "run_id": resolved_run_id,
                "mode": mode,
                "status": "queued",
                "suite": suite,
                "created_at": payload["created_at"],
                "started_at": None,
                "completed_at": None,
            },
        )
        self._write_json(run_dir / "failures.json", {"run_id": resolved_run_id, "failures": []})
        return self.get_run(resolved_run_id)

    def get_run(self, run_id: str) -> EvalRunRecord:
        manifest_path = self.resolve_run_path(run_id, "manifest.json")
        if not manifest_path.exists():
            raise EvalRunNotFoundError(f"Unknown eval run: {run_id}")
        payload = self._read_json(manifest_path)
        return EvalRunRecord(
            run_id=str(payload["run_id"]),
            mode=str(payload["mode"]),
            status=str(payload["status"]),
            suite=self._optional_str(payload.get("suite")),
            triggered_by=str(payload.get("triggered_by", "api")),
            created_at=self._optional_str(payload.get("created_at")),
            started_at=self._optional_str(payload.get("started_at")),
            completed_at=self._optional_str(payload.get("completed_at")),
            total_planned_runs=self._optional_int(payload.get("total_planned_runs")),
            successful_runs=self._optional_int(payload.get("successful_runs")),
            failed_runs=self._optional_int(payload.get("failed_runs")),
            config=dict(payload.get("config", {})),
            artifacts=dict(payload.get("artifacts", {})),
            run_dir=manifest_path.parent,
        )

    def list_runs(
        self,
        *,
        mode: str | None = None,
        limit: int = 20,
    ) -> list[EvalRunRecord]:
        records: list[EvalRunRecord] = []
        for manifest_path in self._base_output_dir.glob("*/manifest.json"):
            try:
                record = self.get_run(manifest_path.parent.name)
            except EvalRunError:
                continue
            if mode is not None and record.mode != mode:
                continue
            records.append(record)

        records.sort(
            key=lambda record: record.created_at or record.run_id,
            reverse=True,
        )
        return records[:limit]

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        summary_payload: dict[str, Any] | None = None,
        failures_payload: dict[str, Any] | None = None,
        **fields: Any,
    ) -> EvalRunRecord:
        record = self.get_run(run_id)
        payload = self._read_json(record.run_dir / "manifest.json")

        if status is not None:
            payload["status"] = status
            now_iso = datetime.now().astimezone().replace(microsecond=0).isoformat()
            if status == "running" and payload.get("started_at") is None:
                payload["started_at"] = now_iso
            if status in TERMINAL_STATUSES:
                payload["completed_at"] = now_iso

        payload.update(fields)
        self._write_json(record.run_dir / "manifest.json", payload)
        self._write_json(
            record.run_dir / "status.json",
            {
                "run_id": payload["run_id"],
                "mode": payload["mode"],
                "status": payload["status"],
                "suite": payload.get("suite"),
                "created_at": payload.get("created_at"),
                "started_at": payload.get("started_at"),
                "completed_at": payload.get("completed_at"),
            },
        )

        if summary_payload is not None:
            self._write_json(record.run_dir / "summary.json", summary_payload)
        if failures_payload is not None:
            self._write_json(record.run_dir / "failures.json", failures_payload)

        return self.get_run(run_id)

    def resolve_run_path(self, run_id: str, filename: str) -> Path:
        self.validate_run_id(run_id)
        if Path(filename).name != filename:
            raise EvalRunValidationError(f"Invalid run artifact name: {filename}")
        run_dir = (self._base_output_dir / run_id).resolve()
        try:
            run_dir.relative_to(self._base_output_dir)
        except ValueError as exc:  # pragma: no cover
            raise EvalRunValidationError(f"Unsafe run path for {run_id}") from exc
        resolved_path = (run_dir / filename).resolve()
        try:
            resolved_path.relative_to(run_dir)
        except ValueError as exc:
            raise EvalRunValidationError(f"Unsafe artifact path for {run_id}") from exc
        return resolved_path

    def validate_run_id(self, run_id: str) -> None:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise EvalRunValidationError(f"Invalid run ID: {run_id}")

    def _normalize_run_token(self, value: str) -> str:
        normalized = "".join(
            character if character.isalnum() or character in {"_", "-", ".", ":"} else "_"
            for character in value.strip()
        )
        return normalized or "run"

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    def _optional_int(self, value: object) -> int | None:
        if value is None:
            return None
        return int(value)

    def _optional_str(self, value: object) -> str | None:
        if value is None:
            return None
        return str(value)
