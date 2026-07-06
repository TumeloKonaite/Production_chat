from __future__ import annotations

from itertools import product

from evals.matrix.models import ExperimentSuiteConfig, MatrixRunSpec, MatrixScalar, ResolvedSuitePlan


def expand_suite_plan(suite: ExperimentSuiteConfig) -> ResolvedSuitePlan:
    retrieval_combinations = expand_parameter_grid(suite.retrieval)
    generation_combinations = expand_parameter_grid(suite.generation)

    runs: list[MatrixRunSpec] = []
    if suite.mode == "retrieval":
        for index, retrieval_config in enumerate(retrieval_combinations, start=1):
            runs.append(
                MatrixRunSpec(
                    index=index,
                    run_id=f"run_{index:03d}",
                    mode=suite.mode,
                    retrieval_config=retrieval_config,
                    generation_config={},
                )
            )
    elif suite.mode == "generation":
        for index, generation_config in enumerate(generation_combinations, start=1):
            runs.append(
                MatrixRunSpec(
                    index=index,
                    run_id=f"run_{index:03d}",
                    mode=suite.mode,
                    retrieval_config={},
                    generation_config=generation_config,
                )
            )
    else:
        index = 1
        for retrieval_config in retrieval_combinations:
            for generation_config in generation_combinations:
                runs.append(
                    MatrixRunSpec(
                        index=index,
                        run_id=f"run_{index:03d}",
                        mode=suite.mode,
                        retrieval_config=retrieval_config,
                        generation_config=generation_config,
                    )
                )
                index += 1

    return ResolvedSuitePlan(
        suite=suite,
        retrieval_combinations=retrieval_combinations,
        generation_combinations=generation_combinations,
        runs=runs,
        total_planned_runs=len(runs),
        requires_confirmation=suite.require_confirmation or "full" in suite.name.casefold(),
    )


def expand_parameter_grid(
    axes: dict[str, tuple[MatrixScalar, ...]],
) -> list[dict[str, MatrixScalar]]:
    if not axes:
        return []

    keys = list(axes)
    combinations = product(*(axes[key] for key in keys))
    return [{key: value for key, value in zip(keys, combination, strict=True)} for combination in combinations]


def format_suite_plan(plan: ResolvedSuitePlan) -> str:
    lines = [
        f"Suite: {plan.suite.name}",
        f"Mode: {plan.suite.mode}",
        "",
    ]
    if plan.suite.mode == "retrieval":
        lines.extend(
            [
                f"Retrieval combinations: {len(plan.retrieval_combinations)}",
                f"Total planned runs: {plan.total_planned_runs}",
            ]
        )
    elif plan.suite.mode == "generation":
        lines.extend(
            [
                f"Generation combinations: {len(plan.generation_combinations)}",
                f"Total planned runs: {plan.total_planned_runs}",
            ]
        )
    else:
        lines.extend(
            [
                f"Retrieval combinations: {len(plan.retrieval_combinations)}",
                f"Generation combinations: {len(plan.generation_combinations)}",
                f"Total end-to-end RAG combinations: {plan.total_planned_runs}",
            ]
        )
    lines.extend(
        [
            "",
            f"Max allowed combinations: {plan.suite.max_combinations}",
            "Status: OK",
        ]
    )
    return "\n".join(lines)
