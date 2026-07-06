from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

MatrixScalar: TypeAlias = str | int | float | bool
MATRIX_MODES = frozenset({"retrieval", "generation", "rag"})


@dataclass(frozen=True, slots=True)
class ExperimentSuiteConfig:
    name: str
    mode: str
    description: str | None
    max_combinations: int
    retrieval: dict[str, tuple[MatrixScalar, ...]]
    generation: dict[str, tuple[MatrixScalar, ...]]
    require_confirmation: bool = False


@dataclass(frozen=True, slots=True)
class ExperimentMatrixConfig:
    suites: dict[str, ExperimentSuiteConfig]
    source_path: Path


@dataclass(frozen=True, slots=True)
class MatrixRunSpec:
    index: int
    run_id: str
    mode: str
    retrieval_config: dict[str, MatrixScalar]
    generation_config: dict[str, MatrixScalar]

    @property
    def combined_config(self) -> dict[str, MatrixScalar]:
        return {
            **{f"retrieval.{key}": value for key, value in self.retrieval_config.items()},
            **{f"generation.{key}": value for key, value in self.generation_config.items()},
        }


@dataclass(frozen=True, slots=True)
class ResolvedSuitePlan:
    suite: ExperimentSuiteConfig
    retrieval_combinations: list[dict[str, MatrixScalar]]
    generation_combinations: list[dict[str, MatrixScalar]]
    runs: list[MatrixRunSpec]
    total_planned_runs: int
    requires_confirmation: bool


@dataclass(frozen=True, slots=True)
class ExperimentMatrixRunResult:
    matrix_run_id: str
    suite_name: str
    mode: str
    output_dir: Path
    manifest_path: Path
    failures_path: Path
    summary_paths: dict[str, Path]
    successful_rows: list[dict[str, object]]
    failures: list[dict[str, object]]
    status: str
