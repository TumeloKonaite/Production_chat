from __future__ import annotations

from pathlib import Path
import tomllib

import modal

APP_NAME = "production-chatbot-api"
SECRET_NAME = "production-chatbot-api-secrets"
REMOTE_ROOT = "/root/project"
PYPROJECT_PATH = Path(__file__).with_name("pyproject.toml")
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
    "redis>=5.2.0,<6.0.0",
    "redisvl>=0.8.0,<1.0.0",
]


def _runtime_dependencies() -> list[str]:
    if not PYPROJECT_PATH.exists():
        return list(FALLBACK_RUNTIME_DEPENDENCIES)

    with PYPROJECT_PATH.open("rb") as pyproject_file:
        project_config = tomllib.load(pyproject_file)["project"]

    base_dependencies = list(project_config.get("dependencies", []))
    optional_dependencies = project_config.get("optional-dependencies", {})
    response_cache_dependencies = list(optional_dependencies.get("response-cache", []))
    return base_dependencies + response_cache_dependencies


modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*_runtime_dependencies())
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
    timeout=150,
)
@modal.asgi_app()
def fastapi_app():
    import sys

    if REMOTE_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_ROOT)

    from app.main import app as api_app

    return api_app
