from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.domain.evals import RagEvalQuestionResult, RagEvalRunSummary
from app.repositories.models import RagEvalResult, RagEvalRun


class EvalRepositoryError(Exception):
    """Raised when RAG evaluation persistence fails."""


class EvalRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create_run(
        self,
        *,
        summary: RagEvalRunSummary,
        results: Sequence[RagEvalQuestionResult],
    ) -> RagEvalRun:
        run = RagEvalRun(
            run_name=summary.run_name,
            model_name=summary.model_name,
            prompt_version=summary.prompt_version,
            retrieval_config=summary.retrieval_config,
            retrieval_top_k=summary.top_k,
            dataset_name=summary.dataset_name,
            total_questions=summary.total_questions,
            avg_precision_at_k=summary.avg_precision_at_k,
            avg_recall_at_k=summary.avg_recall_at_k,
            avg_mrr=summary.avg_mrr,
            avg_ndcg_at_k=summary.avg_ndcg_at_k,
            avg_context_relevance=summary.avg_context_relevance,
            avg_faithfulness=summary.avg_faithfulness,
            avg_answer_relevance=summary.avg_answer_relevance,
        )

        try:
            self._session.add(run)
            self._session.flush()
            self._session.add_all(
                [
                    RagEvalResult(
                        run_id=run.id,
                        question_id=result.question_id,
                        question=result.question,
                        generated_answer=result.generated_answer,
                        expected_source_documents=list(result.expected_source_documents),
                        retrieved_source_documents=list(result.retrieved_source_documents),
                        precision_at_k=result.retrieval_metrics.precision_at_k,
                        recall_at_k=result.retrieval_metrics.recall_at_k,
                        mrr=result.retrieval_metrics.mrr,
                        ndcg_at_k=result.retrieval_metrics.ndcg_at_k,
                        context_relevance_score=result.judge_evaluation.context_relevance.score,
                        context_relevance_reason=result.judge_evaluation.context_relevance.reason,
                        faithfulness_score=result.judge_evaluation.faithfulness.score,
                        faithfulness_reason=result.judge_evaluation.faithfulness.reason,
                        answer_relevance_score=result.judge_evaluation.answer_relevance.score,
                        answer_relevance_reason=result.judge_evaluation.answer_relevance.reason,
                        latency_ms=result.latency_ms,
                        token_usage=result.token_usage,
                    )
                    for result in results
                ]
            )
            self._session.commit()
            self._session.refresh(run)
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise EvalRepositoryError() from exc

        return run
