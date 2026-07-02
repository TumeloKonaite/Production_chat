from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.infrastructure.llm import JudgeClient
from app.infrastructure.prompts import PromptLoader
from app.infrastructure.tracking import create_experiment_tracker
from app.repositories import EvalRepository
from app.repositories.db.session import get_session_factory
from app.services.evals.rag_eval_service import RagEvalService
from app.services.llm import LLMService
from app.services.retrieval import RetrievalService

DEFAULT_DATASET_PATH = ROOT_DIR / "evals" / "datasets" / "portfolio_eval_dataset.jsonl"
DEFAULT_JUDGE_PROMPT_PATH = ROOT_DIR / "evals" / "prompts" / "judge_prompt_v1.md"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "evals" / "results"
DEFAULT_PROMPTS_DIR = ROOT_DIR / "app" / "infrastructure" / "prompts" / "templates"


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
        default=5,
        help="Retrieval top-k used during evaluation.",
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
    settings = get_settings()
    prompt_loader = PromptLoader(prompts_dir=DEFAULT_PROMPTS_DIR)
    llm_service = LLMService(settings=settings)
    retrieval_service = RetrievalService(settings=settings)
    judge_client = JudgeClient(settings=settings)
    experiment_name = args.experiment_name or settings.mlflow_experiment_name
    tracker = create_experiment_tracker(settings, experiment_name)

    eval_repository = None
    session = None
    if not args.no_db:
        session = get_session_factory()()
        eval_repository = EvalRepository(session=session)

    try:
        rag_eval_service = RagEvalService(
            prompt_loader=prompt_loader,
            llm_service=llm_service,
            retrieval_service=retrieval_service,
            judge_client=judge_client,
            eval_repository=eval_repository,
        )

        dataset = rag_eval_service.load_dataset(args.dataset)
        judge_prompt_template = args.judge_prompt.read_text(encoding="utf-8")
        retrieval_config = {
            "name": settings.default_retrieval_config,
            "top_k": args.top_k,
            "min_similarity": settings.retrieval_min_similarity,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.knowledge_embedding_model,
            "embedding_dimension": settings.embedding_dimension,
            "collection_name": settings.knowledge_collection_name,
        }

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        summary, results = await rag_eval_service.evaluate_dataset(
            dataset_name=args.dataset.name,
            examples=dataset,
            run_name=args.run_name,
            prompt_version=args.prompt_version,
            model_config_id=args.model,
            judge_model_config_id=args.judge_model,
            top_k=args.top_k,
            retrieval_config=retrieval_config,
            judge_prompt_template=judge_prompt_template,
            temperature=args.temperature,
            persist_results=not args.no_db,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        result_payload = {
            "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "summary": asdict(summary),
            "results": rag_eval_service.results_as_json(results),
        }
        result_path = args.output_dir / f"{args.run_name}_{timestamp}.json"
        result_path.write_text(
            json.dumps(result_payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

        summary_table = rag_eval_service.render_summary_table([summary])
        summary_path = args.output_dir / f"{args.run_name}_{timestamp}.md"
        summary_path.write_text(summary_table, encoding="utf-8")

        if tracker.enabled:
            with tracker.run(args.run_name):
                tracker.log_params(
                    {
                        "run_name": args.run_name,
                        "dataset_name": args.dataset.name,
                        "dataset_path": str(args.dataset),
                        "model_config_id": args.model,
                        "model_name": summary.model_name,
                        "judge_model_config_id": args.judge_model,
                        "prompt_version": args.prompt_version,
                        "judge_prompt_path": str(args.judge_prompt),
                        "retrieval_config": retrieval_config,
                        "temperature": args.temperature,
                        "top_k": args.top_k,
                        "min_similarity": settings.retrieval_min_similarity,
                        "embedding_provider": settings.embedding_provider,
                        "embedding_model": settings.knowledge_embedding_model,
                        "embedding_dimension": settings.embedding_dimension,
                        "knowledge_collection_name": settings.knowledge_collection_name,
                        "persisted_to_db": not args.no_db,
                    }
                )
                tracker.log_metrics(
                    {
                        "total_questions": summary.total_questions,
                        "avg_precision_at_k": summary.avg_precision_at_k,
                        "avg_recall_at_k": summary.avg_recall_at_k,
                        "avg_mrr": summary.avg_mrr,
                        "avg_ndcg_at_k": summary.avg_ndcg_at_k,
                        "avg_context_relevance": summary.avg_context_relevance,
                        "avg_faithfulness": summary.avg_faithfulness,
                        "avg_answer_relevance": summary.avg_answer_relevance,
                    }
                )
                tracker.log_artifact(result_path)
                tracker.log_artifact(summary_path)

        print(summary_table, end="")
        print(f"Detailed results written to: {result_path}")
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    asyncio.run(main())
