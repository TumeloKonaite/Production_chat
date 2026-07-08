from __future__ import annotations

from pathlib import Path
import tomllib

import modal

APP_NAME = "production-chatbot-api"
SECRET_NAME = "production-chatbot-api-secrets"
REMOTE_ROOT = "/root/project"
PYPROJECT_PATH = Path(__file__).with_name("pyproject.toml")
PREBAKED_HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FASTAPI_TIMEOUT_SECONDS = 300
INGESTION_TIMEOUT_SECONDS = 300
FALLBACK_RUNTIME_DEPENDENCIES = [
    "alembic>=1.16.0,<2.0.0",
    "boto3>=1.43.0,<2.0.0",
    "dagshub>=0.7.0,<1.0.0",
    "fastapi>=0.115.0,<1.0.0",
    "httpx>=0.28.0,<1.0.0",
    "langfuse>=4.13.0,<5.0.0",
    "langchain-huggingface>=1.2.2,<2.0.0",
    "langchain-postgres>=0.0.17,<1.0.0",
    "langchain-text-splitters>=0.3.0,<1.0.0",
    "mlflow>=2.16.0,<4.0.0",
    "psycopg[binary]>=3.2.0,<4.0.0",
    "python-multipart>=0.0.20,<1.0.0",
    "python-dotenv>=1.0.0,<2.0.0",
    "pytest>=8.0.0,<9.0.0",
    "sentence-transformers>=5.2.0,<6.0.0",
    "sqlalchemy>=2.0.0,<3.0.0",
    "uvicorn>=0.30.0,<1.0.0",
    "ruff>=0.15.20",
]


def _runtime_dependencies() -> list[str]:
    if not PYPROJECT_PATH.exists():
        return list(FALLBACK_RUNTIME_DEPENDENCIES)

    with PYPROJECT_PATH.open("rb") as pyproject_file:
        project_config = tomllib.load(pyproject_file)["project"]

    base_dependencies = list(project_config.get("dependencies", []))
    optional_dependencies = project_config.get("optional-dependencies", {})
    deploy_dependencies = list(optional_dependencies.get("deploy", []))
    return base_dependencies + deploy_dependencies


modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*_runtime_dependencies())
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "SENTENCE_TRANSFORMERS_HOME": "/cache/sentence_transformers",
        }
    )
    .run_commands(
        "python -c \"from sentence_transformers import SentenceTransformer; "
        f"SentenceTransformer('{PREBAKED_HF_EMBEDDING_MODEL}')\""
    )
    .add_local_dir("app", remote_path=f"{REMOTE_ROOT}/app")
    .add_local_dir("alembic", remote_path=f"{REMOTE_ROOT}/alembic")
    .add_local_dir("evals", remote_path=f"{REMOTE_ROOT}/evals")
    .add_local_file("alembic.ini", remote_path=f"{REMOTE_ROOT}/alembic.ini")
    .add_local_file("main.py", remote_path=f"{REMOTE_ROOT}/main.py")
)

app = modal.App(name=APP_NAME)


@app.function(
    image=modal_image,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    timeout=FASTAPI_TIMEOUT_SECONDS,
)
@modal.asgi_app()
def fastapi_app():
    import sys

    if REMOTE_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_ROOT)

    from app.main import app as api_app

    return api_app


@app.function(
    image=modal_image,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    timeout=INGESTION_TIMEOUT_SECONDS,
)
def run_ingestion_job(job_id: str) -> dict[str, object]:
    import sys

    if REMOTE_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_ROOT)

    from app.api.dependencies.knowledge_dependencies import build_knowledge_ingestion_job_worker
    from app.config import get_settings
    from app.repositories.db.session import get_session_factory

    settings = get_settings()
    worker = build_knowledge_ingestion_job_worker(settings)
    session_factory = get_session_factory()
    with session_factory() as session:
        result = worker.run_job(session, job_id=job_id)

    return {
        "job_id": result.job_id,
        "source_id": result.source_id,
        "status": result.status,
        "chunk_count": result.chunk_count,
        "embedding_provider": result.embedding_provider,
        "embedding_model": result.embedding_model,
        "embedding_dimension": result.embedding_dimension,
    }
