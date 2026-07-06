from __future__ import annotations

from evals.matrix.expander import expand_suite_plan, format_suite_plan
from evals.matrix.models import ExperimentSuiteConfig
from evals.matrix.runner import run_experiment_matrix


def test_expand_suite_plan_builds_cartesian_product_for_rag() -> None:
    suite = ExperimentSuiteConfig(
        name="rag_medium",
        mode="rag",
        description=None,
        max_combinations=8,
        retrieval={
            "retriever_type": ("vector", "hybrid"),
            "top_k": (3, 5),
        },
        generation={
            "llm_model": ("gpt-4.1-mini",),
            "prompt_version": ("v1_professional", "v2_warm_conversational"),
        },
    )

    plan = expand_suite_plan(suite)

    assert len(plan.retrieval_combinations) == 4
    assert len(plan.generation_combinations) == 2
    assert plan.total_planned_runs == 8
    assert plan.runs[0].run_id == "run_001"
    assert plan.runs[-1].run_id == "run_008"
    assert plan.runs[0].retrieval_config == {"retriever_type": "vector", "top_k": 3}
    assert plan.runs[1].generation_config == {
        "llm_model": "gpt-4.1-mini",
        "prompt_version": "v2_warm_conversational",
    }


def test_format_suite_plan_renders_generation_counts() -> None:
    suite = ExperimentSuiteConfig(
        name="generation_smoke",
        mode="generation",
        description=None,
        max_combinations=4,
        retrieval={},
        generation={
            "llm_model": ("gpt-4.1-mini",),
            "prompt_version": ("v1_professional",),
        },
    )

    plan = expand_suite_plan(suite)
    rendered = format_suite_plan(plan)

    assert "Suite: generation_smoke" in rendered
    assert "Mode: generation" in rendered
    assert "Generation combinations: 1" in rendered
    assert "Total planned runs: 1" in rendered

