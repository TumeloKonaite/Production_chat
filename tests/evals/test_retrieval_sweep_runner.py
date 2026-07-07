from __future__ import annotations

from pathlib import Path

from app.config import Settings
from evals.runners.retrieval_eval_runner import (
    RetrievalDatasetValidationSummary,
    RetrievalEvalRunResult,
)
from evals.runners.run_retrieval_sweep import (
    build_tracking_run_name_for_experiment,
    format_retrieval_sweep_summary,
    load_retrieval_sweep_config,
    run_retrieval_sweep,
)


def test_load_retrieval_sweep_config_parses_valid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "retrieval_sweep.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiments:",
                "  - name: Retrieval-Vector-K3",
                "    retriever_type: VECTOR",
                "    top_k: 3",
                "  - name: retrieval-keyword-k5",
                "    retriever_type: keyword",
                "    top_k: 5",
                "    chunk_size: 500",
                "    chunk_overlap: 100",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_retrieval_sweep_config(config_path)

    assert [experiment.name for experiment in config.experiments] == [
        "Retrieval-Vector-K3",
        "retrieval-keyword-k5",
    ]
    assert config.experiments[0].retriever_type == "vector"
    assert config.experiments[0].top_k == 3
    assert config.experiments[1].chunk_size == 500
    assert config.experiments[1].chunk_overlap == 100


def test_load_retrieval_sweep_config_rejects_invalid_retriever_type(tmp_path: Path) -> None:
    config_path = tmp_path / "retrieval_sweep.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiments:",
                "  - name: retrieval-invalid",
                "    retriever_type: lexical",
                "    top_k: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        load_retrieval_sweep_config(config_path)
    except ValueError as exc:
        assert str(exc) == "experiments[1].retriever_type must be one of: hybrid, keyword, vector."
    else:  # pragma: no cover
        raise AssertionError("Expected config parsing to fail for invalid retriever_type.")


def test_load_retrieval_sweep_config_rejects_invalid_top_k(tmp_path: Path) -> None:
    config_path = tmp_path / "retrieval_sweep.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiments:",
                "  - name: retrieval-vector-k0",
                "    retriever_type: vector",
                "    top_k: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        load_retrieval_sweep_config(config_path)
    except ValueError as exc:
        assert str(exc) == "experiments[1].top_k must be a positive integer."
    else:  # pragma: no cover
        raise AssertionError("Expected config parsing to fail for invalid top_k.")


def test_run_retrieval_sweep_calls_shared_runner_for_each_experiment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "retrieval_sweep.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiments:",
                "  - name: retrieval-vector-k3",
                "    retriever_type: vector",
                "    top_k: 3",
                "  - name: retrieval-keyword-k5",
                "    retriever_type: keyword",
                "    top_k: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sweep_config = load_retrieval_sweep_config(config_path)

    validation_summary = RetrievalDatasetValidationSummary(
        total_queries=2,
        queries_with_expected_sources=2,
        queries_without_expected_sources=0,
        missing_expected_source_ids=[],
    )
    monkeypatch.setattr(
        "evals.runners.run_retrieval_sweep.load_and_validate_dataset",
        lambda *args, **kwargs: (["example"], validation_summary),
    )
    monkeypatch.setattr(
        "evals.runners.run_retrieval_sweep.create_experiment_tracker",
        lambda *args, **kwargs: type("Tracker", (), {"enabled": True})(),
    )

    calls: list[dict[str, object]] = []

    def fake_run_retrieval_eval(**kwargs):
        calls.append(kwargs)
        run_output_dir = kwargs["output_root"] / kwargs["output_label"]
        summary = {
            "num_queries_total": 2,
            "num_queries_evaluated": 2,
            "num_queries_without_expected_source": 0,
            "num_queries_without_expected_sources": 0,
            "k": kwargs["top_k"],
            "hit_at_k": 1.0,
            "recall_at_k": 0.5,
            "precision_at_k": 0.25,
            "mean_precision_at_k": 0.25,
            "mrr": 0.75,
        }
        config = {
            "dataset_path": str(kwargs["dataset_path"]),
            "embedding_provider": kwargs["settings"].embedding_provider,
            "embedding_model": kwargs["settings"].knowledge_embedding_model,
            "embedding_dimension": kwargs["settings"].embedding_dimension,
            "git_commit_sha": "abc123",
        }
        artifact_paths = {
            "results_json": run_output_dir / "results.json",
            "results_csv": run_output_dir / "results.csv",
            "config_json": run_output_dir / "config.json",
        }
        return RetrievalEvalRunResult(
            run_name=str(kwargs["run_name"]),
            mlflow_run_id=None,
            output_dir=run_output_dir,
            summary=summary,
            results=[],
            config=config,
            artifact_paths=artifact_paths,
            validation_summary=validation_summary,
        )

    monkeypatch.setattr(
        "evals.runners.run_retrieval_sweep.run_retrieval_eval",
        fake_run_retrieval_eval,
    )

    rows, artifact_paths = run_retrieval_sweep(
        sweep_config=sweep_config,
        sweep_config_path=config_path,
        dataset_path=tmp_path / "dataset.jsonl",
        output_dir=tmp_path / "output",
        settings=_build_settings(),
        argv=["evals/runners/run_retrieval_sweep.py", "--config", str(config_path)],
    )

    assert len(calls) == 2
    assert calls[0]["settings"].retriever_type == "vector"
    assert calls[0]["settings"].retrieval_top_k == 3
    assert calls[1]["settings"].retriever_type == "keyword"
    assert calls[1]["settings"].retrieval_top_k == 5
    assert calls[0]["examples"] == ["example"]
    assert calls[0]["validation_summary"] == validation_summary
    assert rows[0]["run_name"].startswith("retrieval-vector-k3-")
    assert rows[1]["run_name"].startswith("retrieval-keyword-k5-")
    assert rows[0]["run_name"] != rows[1]["run_name"]
    assert rows[0]["rank"] == 1
    assert rows[0]["is_best"] is True
    assert artifact_paths["summary_json"].exists()
    assert artifact_paths["summary_csv"].exists()
    assert artifact_paths["ranking_md"].exists()
    assert artifact_paths["manifest_json"].exists()


def test_build_tracking_run_name_for_experiment_is_unique_and_meaningful() -> None:
    first_name = build_tracking_run_name_for_experiment(
        experiment=load_retrieval_sweep_config_from_text(
            """
experiments:
  - name: retrieval-vector-k3
    retriever_type: vector
    top_k: 3
"""
        ).experiments[0],
        timestamp_label="2026-07-03_170000_run_01_retrieval-vector-k3",
    )
    second_name = build_tracking_run_name_for_experiment(
        experiment=load_retrieval_sweep_config_from_text(
            """
experiments:
  - name: retrieval-vector-k3
    retriever_type: vector
    top_k: 3
"""
        ).experiments[0],
        timestamp_label="2026-07-03_170001_run_02_retrieval-vector-k3",
    )

    assert first_name == "retrieval-vector-k3-2026-07-03_170000_run_01_retrieval-vector-k3"
    assert second_name == "retrieval-vector-k3-2026-07-03_170001_run_02_retrieval-vector-k3"
    assert first_name != second_name


def test_format_retrieval_sweep_summary_renders_compact_table() -> None:
    table = format_retrieval_sweep_summary(
        [
            {
                "run_name": "retrieval-vector-k3-2026-07-03_170000",
                "retriever_type": "vector",
                "top_k": 3,
                "mrr": 0.396,
                "recall_at_k": 0.316,
                "mean_precision_at_k": 0.264,
            }
        ]
    )

    assert "run_name" in table
    assert "rank" in table
    assert "retrieval-vector-k3-2026-07-03_170000" in table
    assert "0.396" in table


def load_retrieval_sweep_config_from_text(payload: str):
    path = Path("unused.yaml")
    temp_dir = Path.cwd()
    temp_file = temp_dir / path
    temp_file.write_text(payload.strip() + "\n", encoding="utf-8")
    try:
        return load_retrieval_sweep_config(temp_file)
    finally:
        temp_file.unlink(missing_ok=True)


def _build_settings() -> Settings:
    return Settings(
        database_url="postgresql+psycopg://postgres:postgres@127.0.0.1:5434/test",
        openai_api_key=None,
        openai_base_url="https://api.openai.com/v1",
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        tavus_api_key=None,
        tavus_base_url="https://tavusapi.com",
        tavus_face_id=None,
        tavus_pal_id=None,
        public_backend_url=None,
        tavus_tool_secret=None,
        ingestion_api_secret=None,
        eval_admin_token=None,
        default_model_config_id="openai:gpt-4.1-mini",
        model_configs_json=None,
        embedding_provider="hf",
        knowledge_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        knowledge_collection_name="personal_knowledge_base",
        default_prompt_version="v1_professional",
        conversation_history_limit=10,
        retriever_type="vector",
        retrieval_top_k=5,
        retrieval_min_similarity=0.55,
        default_retrieval_config="default",
        enable_mlflow_tracking=False,
        mlflow_tracking_uri=None,
        mlflow_experiment_name="test-experiment",
        enable_dagshub_tracking=False,
        dagshub_repo_owner=None,
        dagshub_repo_name=None,
        dagshub_token=None,
    )
