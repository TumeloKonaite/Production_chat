from __future__ import annotations

import json

from scripts.run_chunking_experiment import (
    build_comparison_rows,
    parse_chunk_configs,
    write_comparison_artifacts,
)


def test_parse_chunk_configs_parses_multiple_values() -> None:
    assert parse_chunk_configs("300:50, 500:100") == [(300, 50), (500, 100)]


def test_parse_chunk_configs_rejects_invalid_values() -> None:
    try:
        parse_chunk_configs("300:300")
    except ValueError as exc:
        assert "chunk_overlap must be smaller than chunk_size" in str(exc)
    else:
        raise AssertionError("Expected invalid chunk config to raise ValueError.")


def test_write_comparison_artifacts_marks_best_run(tmp_path) -> None:
    rows = build_comparison_rows(
        [
            {
                "chunk_size": 300,
                "chunk_overlap": 50,
                "documents_loaded": 9,
                "chunks_indexed": 42,
                "k": 5,
                "hit_at_k": 1.0,
                "recall_at_k": 0.8,
                "mean_precision_at_k": 0.4,
                "mrr": 0.6,
                "run_output_dir": "run-a",
                "results_json": "run-a/results.json",
                "results_csv": "run-a/results.csv",
                "config_json": "run-a/config.json",
            },
            {
                "chunk_size": 500,
                "chunk_overlap": 100,
                "documents_loaded": 9,
                "chunks_indexed": 24,
                "k": 5,
                "hit_at_k": 1.0,
                "recall_at_k": 0.9,
                "mean_precision_at_k": 0.5,
                "mrr": 0.7,
                "run_output_dir": "run-b",
                "results_json": "run-b/results.json",
                "results_csv": "run-b/results.csv",
                "config_json": "run-b/config.json",
            },
        ]
    )

    artifact_paths = write_comparison_artifacts(tmp_path, rows=rows)

    payload = json.loads(artifact_paths["summary_json"].read_text(encoding="utf-8"))
    assert payload["best_configuration"]["chunk_size"] == 500
    assert payload["runs"][0]["is_best"] is True
    csv_text = artifact_paths["summary_csv"].read_text(encoding="utf-8")
    assert "is_best" in csv_text
    assert "500" in csv_text
    assert artifact_paths["ranking_md"].exists()
