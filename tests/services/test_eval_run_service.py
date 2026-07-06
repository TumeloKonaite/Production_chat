from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.evals.eval_run_service import (
    EvalArtifactNotFoundError,
    EvalRunService,
    EvalRunValidationError,
)
from app.services.evals.eval_artifact_reader import EvalArtifactReader


def test_create_run_writes_manifest_and_status(tmp_path: Path) -> None:
    service = EvalRunService(base_output_dir=tmp_path / "runs")

    record = service.create_run(
        mode="retrieval",
        config={"top_k": 5},
        run_id="2026-07-04_15-20-11_retrieval",
    )

    assert record.run_id == "2026-07-04_15-20-11_retrieval"
    manifest = json.loads((record.run_dir / "manifest.json").read_text(encoding="utf-8"))
    status = json.loads((record.run_dir / "status.json").read_text(encoding="utf-8"))
    failures = json.loads((record.run_dir / "failures.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "queued"
    assert status["status"] == "queued"
    assert failures == {"run_id": "2026-07-04_15-20-11_retrieval", "failures": []}


def test_update_run_writes_summary_and_failures(tmp_path: Path) -> None:
    service = EvalRunService(base_output_dir=tmp_path / "runs")
    reader = EvalArtifactReader(service)
    service.create_run(mode="matrix", suite="rag_medium", config={"suite": "rag_medium"}, run_id="run_001")

    service.update_run(
        "run_001",
        status="completed_with_failures",
        total_planned_runs=4,
        successful_runs=3,
        failed_runs=1,
        summary_payload={"run_id": "run_001", "top_results": [{"rank": 1}]},
        failures_payload={"run_id": "run_001", "failures": [{"run_id": "run_003"}]},
        artifacts={"summary_json": "runner_output/rag_summary.json"},
    )

    record = service.get_run("run_001")
    assert record.status == "completed_with_failures"
    assert record.successful_runs == 3
    assert reader.read_summary("run_001") == {"run_id": "run_001", "top_results": [{"rank": 1}]}
    assert reader.read_failures("run_001") == {
        "run_id": "run_001",
        "failures": [{"run_id": "run_003"}],
    }


def test_validate_run_id_and_artifact_path_safety(tmp_path: Path) -> None:
    service = EvalRunService(base_output_dir=tmp_path / "runs")

    with pytest.raises(EvalRunValidationError):
        service.create_run(mode="retrieval", config={}, run_id="../bad")

    service.create_run(mode="retrieval", config={}, run_id="safe_run")
    with pytest.raises(EvalRunValidationError):
        service.resolve_run_path("safe_run", "../summary.json")


def test_reader_raises_for_missing_summary(tmp_path: Path) -> None:
    service = EvalRunService(base_output_dir=tmp_path / "runs")
    reader = EvalArtifactReader(service)
    service.create_run(mode="retrieval", config={}, run_id="safe_run")

    with pytest.raises(EvalArtifactNotFoundError):
        reader.read_summary("safe_run")
