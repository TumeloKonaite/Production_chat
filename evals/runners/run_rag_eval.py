from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.infrastructure.llm import JudgeClient
from app.infrastructure.prompts import PromptLoader
from app.infrastructure.tracking.conventions import (
    build_generation_tracking_params,
    build_rag_tracking_metrics,
    build_retrieval_tracking_params,
    extract_model_provider,
    get_git_sha,
    resolve_prompt_template_path,
)
from app.infrastructure.tracking import create_experiment_tracker
from app.repositories import EvalRepository
from app.repositories.db.session import get_session_factory
from app.services.evals.rag_eval_service import RagEvalService
from app.services.llm import LLMService
from app.services.retrieval import RetrievalService
from evals.runners.query_rewriter import QueryRewriter

DEFAULT_DATASET_PATH = ROOT_DIR / "evals" / "datasets" / "portfolio_eval_dataset.jsonl"
DEFAULT_JUDGE_PROMPT_PATH = ROOT_DIR / "evals" / "prompts" / "judge_prompt_v1.md"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "evals" / "results"
DEFAULT_PROMPTS_DIR = ROOT_DIR / "app" / "infrastructure" / "prompts" / "templates"


@dataclass(frozen=True, slots=True)
class RagEvalRunResult:
    run_name: str
    dataset_path: Path
    prompt_version: str
    model_config_id: str
    judge_model_config_id: str | None
    top_k: int
    temperature: float
    max_tokens: int | None
    retrieval_config: dict[str, object]
    summary: object
    results: list[object]
    artifact_paths: dict[str, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval and answer-quality evaluation for the portfolio chatbot.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the RAG evaluation dataset JSONL file.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model config ID used to generate chatbot answers, for example openai:gpt-4.1-mini.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Optional model config ID used for LLM-as-a-judge scoring.",
    )
    parser.add_argument(
        "--prompt-version",
        required=True,
        help="Prompt version to use for answer generation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Retrieval top-k used during evaluation. Defaults to RETRIEVAL_TOP_K.",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Stable run label used for persistence and summary output.",
    )
    parser.add_argument(
        "--judge-prompt",
        type=Path,
        default=DEFAULT_JUDGE_PROMPT_PATH,
        help="Judge prompt template markdown file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where result artifacts will be written.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for answer generation.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional max_tokens override passed to the answer-generation model client.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Override the MLflow experiment name used for this evaluation run.",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip database persistence and only write local artifacts.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    try:
        result = await run_rag_eval(
            settings=get_settings(),
            dataset_path=args.dataset.resolve(),
            output_dir=args.output_dir.resolve(),
            model_config_id=args.model,
            judge_model_config_id=args.judge_model,
            prompt_version=args.prompt_version,
            top_k=args.top_k,
            run_name=args.run_name,
            judge_prompt_path=args.judge_prompt.resolve(),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            experiment_name=args.experiment_name,
            persist_results=not args.no_db,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print(Path(result.artifact_paths["summary_md"]).read_text(encoding="utf-8"), end="")
    print(f"Detailed results written to: {result.artifact_paths['results_json']}")


async def run_rag_eval(
    *,
    settings,
    dataset_path: Path,
    output_dir: Path,
    model_config_id: str,
    judge_model_config_id: str | None,
    prompt_version: str,
    top_k: int | None,
    run_name: str,
    judge_prompt_path: Path,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    experiment_name: str | None = None,
    persist_results: bool = True,
    tracker=None,
    argv: list[str] | None = None,
) -> RagEvalRunResult:
    resolved_top_k = top_k if top_k is not None else settings.retrieval_top_k
    prompt_loader = PromptLoader(prompts_dir=DEFAULT_PROMPTS_DIR)
    prompt_template_path = resolve_prompt_template_path(
        prompts_dir=DEFAULT_PROMPTS_DIR,
        prompt_version=prompt_version,
    )
    llm_service = LLMService(settings=settings)
    retrieval_service = RetrievalService(settings=settings)
    judge_client = JudgeClient(settings=settings)
    experiment_tracker = tracker or create_experiment_tracker(
        settings,
        experiment_name or settings.mlflow_experiment_name,
    )

    eval_repository = None
    session = None
    if persist_results:
        session = get_session_factory()()
        eval_repository = EvalRepository(session=session)

    try:
        rag_eval_service = RagEvalService(
            prompt_loader=prompt_loader,
            llm_service=llm_service,
            retrieval_service=retrieval_service,
            judge_client=judge_client,
            eval_repository=eval_repository,
            query_rewriter=QueryRewriter(settings=settings) if settings.enable_query_rewriting else None,
        )

        dataset = rag_eval_service.load_dataset(dataset_path)
        judge_prompt_template = judge_prompt_path.read_text(encoding="utf-8")
        retrieval_config = {
            "name": settings.default_retrieval_config,
            "retriever_type": settings.retriever_type,
            "top_k": resolved_top_k,
            "min_similarity": settings.retrieval_min_similarity,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.knowledge_embedding_model,
            "embedding_dimension": settings.embedding_dimension,
            "chunk_size": settings.knowledge_chunk_size,
            "chunk_overlap": settings.knowledge_chunk_overlap,
            "query_rewriting": settings.enable_query_rewriting,
            "query_rewrite_model": settings.query_rewrite_model if settings.enable_query_rewriting else None,
            "query_rewrite_prompt_version": (
                settings.query_rewrite_prompt_version if settings.enable_query_rewriting else None
            ),
            "reranker": settings.reranker_type if settings.enable_reranking else "none",
            "reranker_enabled": settings.enable_reranking,
            "reranker_model": settings.reranker_model if settings.enable_reranking else None,
            "reranker_initial_top_k": (
                settings.reranker_initial_top_k if settings.enable_reranking else None
            ),
            "reranker_final_top_k": resolved_top_k,
            "collection_name": settings.knowledge_collection_name,
        }

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        summary, results = await rag_eval_service.evaluate_dataset(
            dataset_name=dataset_path.name,
            examples=dataset,
            run_name=run_name,
            prompt_version=prompt_version,
            model_config_id=model_config_id,
            judge_model_config_id=judge_model_config_id,
            top_k=resolved_top_k,
            retrieval_config=retrieval_config,
            judge_prompt_template=judge_prompt_template,
            temperature=temperature,
            max_tokens=max_tokens,
            persist_results=persist_results,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        result_payload = {
            "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "summary": asdict(summary),
            "results": rag_eval_service.results_as_json(results),
        }
        result_path = output_dir / f"{run_name}_{timestamp}.json"
        result_path.write_text(
            json.dumps(result_payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

        summary_table = rag_eval_service.render_summary_table([summary])
        summary_path = output_dir / f"{run_name}_{timestamp}.md"
        summary_path.write_text(summary_table, encoding="utf-8")
        config_path = output_dir / f"{run_name}_{timestamp}_config.json"
        prompt_artifact_path: Path | None = None
        if prompt_template_path is not None:
            prompt_artifact_path = output_dir / f"{run_name}_{timestamp}_prompt.md"
            prompt_artifact_path.write_text(
                prompt_template_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        judge_prompt_artifact_path = output_dir / f"{run_name}_{timestamp}_judge_prompt.md"
        judge_prompt_artifact_path.write_text(judge_prompt_template, encoding="utf-8")
        config_path.write_text(
            json.dumps(
                {
                    "dataset_path": str(dataset_path),
                    "model_config_id": model_config_id,
                    "judge_model_config_id": judge_model_config_id,
                    "prompt_version": prompt_version,
                    "prompt_template_path": str(prompt_template_path) if prompt_template_path else None,
                    "top_k": resolved_top_k,
                    "judge_prompt_path": str(judge_prompt_path),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "retrieval_config": retrieval_config,
                    "persist_results": persist_results,
                    "git_sha": get_git_sha(),
                    "python_command_used": " ".join(argv or []),
                },
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )

        if experiment_tracker.enabled:
            with experiment_tracker.run(run_name):
                params = build_retrieval_tracking_params(
                    workflow="rag_eval",
                    experiment_family="rag_eval",
                    run_name=run_name,
                    settings=settings,
                    dataset_path=dataset_path,
                    top_k=resolved_top_k,
                    chunk_size=settings.knowledge_chunk_size,
                    chunk_overlap=settings.knowledge_chunk_overlap,
                    git_sha=get_git_sha(),
                    extra={
                        "persisted_to_db": persist_results,
                        "judge_prompt_path": str(judge_prompt_path),
                    },
                )
                params.update(
                    build_generation_tracking_params(
                        workflow="rag_eval",
                        experiment_family="rag_eval",
                        run_name=run_name,
                        dataset_path=dataset_path,
                        dataset_version=dataset_path.stem,
                        prompt_version=prompt_version,
                        prompt_template_path=prompt_template_path,
                        model_config_id=summary.model_config_id,
                        llm_provider=extract_model_provider(summary.model_config_id),
                        llm_model=summary.model_name,
                        llm_base_url=llm_service.get_model_base_url(summary.model_config_id),
                        temperature=temperature,
                        max_tokens=max_tokens,
                        git_sha=get_git_sha(),
                        judge_model_config_id=judge_model_config_id,
                        retrieval_config=str(retrieval_config.get("name", "default")),
                        context_top_k=resolved_top_k,
                    )
                )
                experiment_tracker.log_params(params)
                experiment_tracker.log_metrics(build_rag_tracking_metrics(summary))
                experiment_tracker.log_artifact(result_path)
                experiment_tracker.log_artifact(summary_path)
                experiment_tracker.log_artifact(config_path)
                experiment_tracker.log_artifact(judge_prompt_artifact_path)
                if prompt_artifact_path is not None:
                    experiment_tracker.log_artifact(prompt_artifact_path)

        return RagEvalRunResult(
            run_name=run_name,
            dataset_path=dataset_path,
            prompt_version=prompt_version,
            model_config_id=summary.model_config_id,
            judge_model_config_id=judge_model_config_id,
            top_k=resolved_top_k,
            temperature=temperature,
            max_tokens=max_tokens,
            retrieval_config=retrieval_config,
            summary=summary,
            results=list(results),
            artifact_paths={
                "results_json": result_path,
                "summary_md": summary_path,
                "config_json": config_path,
                "judge_prompt_md": judge_prompt_artifact_path,
                **({"prompt_template_md": prompt_artifact_path} if prompt_artifact_path else {}),
            },
        )
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    asyncio.run(main())
