from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from app.services.retrieval import RetrievedChunk
from scripts.run_embedding_experiment import (
    EmbeddingRunConfig,
    build_embedding_comparison_rows,
    load_embedding_experiment_config,
    run_embedding_experiment_matrix,
    _validate_vector_store_dimension,
    write_embedding_comparison_artifacts,
)


class FakeSession:
    def __enter__(self) -> object:
        return object()

    def __exit__(self, *_: object) -> None:
        return None


class FakeRetrievalService:
    def __init__(self, settings: Settings) -> None:
        self._provider = settings.embedding_provider
        self._model = settings.knowledge_embedding_model

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        del top_k
        if self._provider == "openrouter":
            responses = {
                "question 1": [
                    _build_chunk(chunk_id="projects.md::chunk-1", source="projects.md"),
                    _build_chunk(chunk_id="skills.md::chunk-1", source="skills.md"),
                ],
                "question 2": [
                    _build_chunk(chunk_id="skills.md::chunk-2", source="skills.md"),
                    _build_chunk(chunk_id="projects.md::chunk-2", source="projects.md"),
                ],
            }
        else:
            responses = {
                "question 1": [
                    _build_chunk(chunk_id="projects.md::chunk-1", source="projects.md"),
                    _build_chunk(chunk_id="skills.md::chunk-1", source="skills.md"),
                ],
                "question 2": [
                    _build_chunk(chunk_id="projects.md::chunk-2", source="projects.md"),
                    _build_chunk(chunk_id="skills.md::chunk-2", source="skills.md"),
                ],
            }
        return list(responses.get(query, []))

    def get_vector_store_dimension(self) -> int | None:
        return 384


def test_load_embedding_experiment_config_normalizes_and_sorts_values(tmp_path: Path) -> None:
    config_path = tmp_path / "embedding_config.json"
    config_path.write_text(
        json.dumps(
            {
                "embedding_runs": [
                    {
                        "provider": "HF",
                        "model": "sentence-transformers/all-MiniLM-L6-v2",
                        "dimension": 384,
                    }
                ],
                "k_values": [5, 1, 3, 3],
            }
        ),
        encoding="utf-8",
    )

    config = load_embedding_experiment_config(config_path)

    assert config.embedding_runs[0].provider == "hf"
    assert config.embedding_runs[0].model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.embedding_runs[0].dimension == 384
    assert config.k_values == [1, 3, 5]


def test_build_embedding_comparison_rows_uses_deterministic_ranking() -> None:
    rows = build_embedding_comparison_rows(
        [
            {
                "embedding_provider": "hf",
                "embedding_model": "model-a",
                "embedding_dimension": 384,
                "mrr": 0.7,
                "recall_at_1": 0.5,
                "recall_at_3": 0.8,
                "recall_at_5": 0.9,
            },
            {
                "embedding_provider": "openrouter",
                "embedding_model": "model-b",
                "embedding_dimension": 384,
                "mrr": 0.7,
                "recall_at_1": 0.6,
                "recall_at_3": 0.8,
                "recall_at_5": 0.9,
            },
        ],
        k_values=[1, 3, 5],
    )

    assert rows[0]["embedding_provider"] == "openrouter"
    assert rows[0]["is_best"] is True
    assert rows[0]["rank"] == 1
    assert rows[1]["rank"] == 2


def test_write_embedding_comparison_artifacts_persists_best_setup(tmp_path: Path) -> None:
    rows = build_embedding_comparison_rows(
        [
            {
                "embedding_provider": "hf",
                "embedding_model": "model-a",
                "embedding_dimension": 384,
                "documents_loaded": 9,
                "chunks_indexed": 20,
                "mrr": 0.6,
                "recall_at_1": 0.4,
                "recall_at_3": 0.7,
                "recall_at_5": 0.8,
                "run_output_dir": "run-a",
                "results_json": "run-a/results.json",
                "results_csv": "run-a/results.csv",
                "config_json": "run-a/config.json",
            },
            {
                "embedding_provider": "openrouter",
                "embedding_model": "model-b",
                "embedding_dimension": 384,
                "documents_loaded": 9,
                "chunks_indexed": 20,
                "mrr": 0.8,
                "recall_at_1": 0.7,
                "recall_at_3": 0.9,
                "recall_at_5": 1.0,
                "run_output_dir": "run-b",
                "results_json": "run-b/results.json",
                "results_csv": "run-b/results.csv",
                "config_json": "run-b/config.json",
            },
        ],
        k_values=[1, 3, 5],
    )

    manifest_path = tmp_path / "experiment_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    artifact_paths = write_embedding_comparison_artifacts(
        tmp_path,
        rows=rows,
        k_values=[1, 3, 5],
        experiment_manifest_path=manifest_path,
    )

    payload = json.loads(artifact_paths["summary_json"].read_text(encoding="utf-8"))
    assert payload["best_embedding_setup"]["embedding_provider"] == "openrouter"
    assert payload["ranking"]["primary_metric"] == "recall_at_k"
    csv_text = artifact_paths["summary_csv"].read_text(encoding="utf-8")
    assert "embedding_provider" in csv_text
    assert "openrouter" in csv_text
    assert artifact_paths["ranking_md"].exists()


def test_run_embedding_experiment_matrix_writes_ranked_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "embedding_config.json"
    config_path.write_text(
        json.dumps(
            {
                "embedding_runs": [
                    {
                        "provider": "hf",
                        "model": "sentence-transformers/all-MiniLM-L6-v2",
                        "dimension": 384,
                    },
                    {
                        "provider": "openrouter",
                        "model": "openai/text-embedding-3-small",
                        "dimension": 384,
                    },
                ],
                "k_values": [1, 3, 5],
            }
        ),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "q1",
                        "question": "question 1",
                        "expected_source_documents": ["projects.md"],
                    }
                ),
                json.dumps(
                    {
                        "id": "q2",
                        "question": "question 2",
                        "expected_source_documents": ["skills.md"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    experiment_config = load_embedding_experiment_config(config_path)

    monkeypatch.setattr(
        "scripts.run_embedding_experiment.get_session_factory",
        lambda: (lambda: FakeSession()),
    )
    monkeypatch.setattr(
        "scripts.run_embedding_experiment._get_database_vector_store_dimension",
        lambda: None,
    )
    monkeypatch.setattr(
        "scripts.run_embedding_experiment.RetrievalService",
        FakeRetrievalService,
    )
    monkeypatch.setattr(
        "scripts.run_embedding_experiment.ingest_knowledge",
        lambda *args, **kwargs: (
            [SimpleNamespace(source="projects.md"), SimpleNamespace(source="skills.md")],
            [
                SimpleNamespace(source="projects.md", chunk_count=2),
                SimpleNamespace(source="skills.md", chunk_count=1),
            ],
        ),
    )

    rows, artifact_paths = run_embedding_experiment_matrix(
        experiment_config=experiment_config,
        experiment_config_path=config_path,
        dataset_path=dataset_path,
        output_dir=tmp_path / "output",
        settings=_build_settings(),
        argv=["scripts/run_embedding_experiment.py", "--config", str(config_path)],
    )

    assert rows[0]["embedding_provider"] == "openrouter"
    assert rows[0]["embedding_model"] == "openai/text-embedding-3-small"
    assert rows[0]["embedding_dimension"] == 384
    assert rows[0]["recall_at_1"] == 1.0
    assert rows[1]["embedding_provider"] == "hf"

    payload = json.loads(artifact_paths["summary_json"].read_text(encoding="utf-8"))
    assert payload["best_embedding_setup"]["embedding_provider"] == "openrouter"
    assert payload["best_embedding_setup"]["mrr"] == 1.0
    assert payload["runs"][1]["embedding_provider"] == "hf"

    run_results = list((tmp_path / "output" / "runs").glob("*/results.json"))
    assert len(run_results) == 2


def test_validate_vector_store_dimension_fails_early_for_pgvector_mismatch() -> None:
    try:
        _validate_vector_store_dimension(
            embedding_run=EmbeddingRunConfig(
                provider="openrouter",
                model="openai/text-embedding-3-small",
                dimension=1536,
            ),
            database_vector_dimension=384,
        )
    except ValueError as exc:
        message = str(exc)
        assert "Database vector dimension mismatch" in message
        assert "Configured dimension: 1536" in message
        assert "Database vector store dimension: 384" in message
    else:
        raise AssertionError("Expected early pgvector dimension validation to fail.")


def _build_settings() -> Settings:
    return Settings(
        database_url="postgresql+psycopg://postgres:postgres@127.0.0.1:5434/test",
        openai_api_key=None,
        openai_base_url="https://api.openai.com/v1",
        openrouter_api_key="openrouter-test-key",
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


def _build_chunk(*, chunk_id: str, source: str, similarity: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        source=source,
        section="Section",
        content=f"content from {source}",
        similarity=similarity,
        metadata={"chunk_id": chunk_id, "source": source},
    )
