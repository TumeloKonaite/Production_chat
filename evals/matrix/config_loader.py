from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.matrix.models import (
    ExperimentMatrixConfig,
    ExperimentSuiteConfig,
    MATRIX_MODES,
    MatrixScalar,
)

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to load experiment matrix configs.") from exc

SUPPORTED_RETRIEVAL_AXES = frozenset(
    {
        "chunk_overlap",
        "chunk_size",
        "embedding_dimension",
        "embedding_model",
        "embedding_provider",
        "query_rewrite_model",
        "query_rewrite_prompt_version",
        "query_rewrite_temperature",
        "query_rewriting",
        "query_rewriting_enabled",
        "reranker",
        "reranker_enabled",
        "reranker_initial_top_k",
        "reranker_model",
        "reranker_type",
        "retriever_type",
        "top_k",
    }
)
SUPPORTED_GENERATION_AXES = frozenset(
    {
        "dataset_version",
        "judge_model",
        "judge_model_config_id",
        "llm_model",
        "model_config_id",
        "prompt_version",
        "temperature",
    }
)


def load_experiment_matrix_config(path: Path) -> ExperimentMatrixConfig:
    payload = _load_payload(path)
    if not isinstance(payload, dict):
        raise ValueError("Experiment matrix config must be a YAML or JSON object.")

    raw_suites = payload.get("suites")
    if not isinstance(raw_suites, dict) or not raw_suites:
        raise ValueError("suites must be a non-empty object.")

    suites: dict[str, ExperimentSuiteConfig] = {}
    for suite_name, raw_suite in raw_suites.items():
        if not isinstance(suite_name, str) or not suite_name.strip():
            raise ValueError("Suite names must be non-empty strings.")
        if not isinstance(raw_suite, dict):
            raise ValueError(f"suites.{suite_name} must be an object.")

        normalized_name = suite_name.strip()
        mode = _require_mode(raw_suite.get("mode"), field_name=f"suites.{normalized_name}.mode")
        max_combinations = _require_positive_int(
            raw_suite.get("max_combinations"),
            field_name=f"suites.{normalized_name}.max_combinations",
        )
        description = _optional_string(
            raw_suite.get("description"),
            field_name=f"suites.{normalized_name}.description",
        )
        require_confirmation = _optional_bool(
            raw_suite.get("require_confirmation"),
            field_name=f"suites.{normalized_name}.require_confirmation",
            default=False,
        )
        retrieval_axes = _load_axes(
            raw_suite.get("retrieval"),
            field_name=f"suites.{normalized_name}.retrieval",
            supported_keys=SUPPORTED_RETRIEVAL_AXES,
        )
        generation_axes = _load_axes(
            raw_suite.get("generation"),
            field_name=f"suites.{normalized_name}.generation",
            supported_keys=SUPPORTED_GENERATION_AXES,
        )
        _validate_suite_sections(
            suite_name=normalized_name,
            mode=mode,
            retrieval_axes=retrieval_axes,
            generation_axes=generation_axes,
        )

        suites[normalized_name] = ExperimentSuiteConfig(
            name=normalized_name,
            mode=mode,
            description=description,
            max_combinations=max_combinations,
            retrieval=retrieval_axes,
            generation=generation_axes,
            require_confirmation=require_confirmation,
        )

    return ExperimentMatrixConfig(suites=suites, source_path=path.resolve())


def _load_payload(path: Path) -> object:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ValueError(f"Experiment matrix config not found: {path}") from None

    suffix = path.suffix.casefold()
    if suffix == ".json":
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Experiment matrix config must be valid JSON: {path}") from exc

    try:
        return yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Experiment matrix config must be valid YAML: {path}") from exc


def _load_axes(
    value: object,
    *,
    field_name: str,
    supported_keys: frozenset[str],
) -> dict[str, tuple[MatrixScalar, ...]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object when provided.")

    axes: dict[str, tuple[MatrixScalar, ...]] = {}
    for raw_key, raw_values in value.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ValueError(f"{field_name} keys must be non-empty strings.")
        key = raw_key.strip()
        if key not in supported_keys:
            supported = ", ".join(sorted(supported_keys))
            raise ValueError(f"{field_name}.{key} is not supported. Supported keys: {supported}.")
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(f"{field_name}.{key} must be a non-empty list.")

        normalized_values: list[MatrixScalar] = []
        for index, item in enumerate(raw_values, start=1):
            if isinstance(item, bool | str | int | float):
                normalized_values.append(item)
                continue
            raise ValueError(
                f"{field_name}.{key}[{index}] must be a string, number, or boolean."
            )
        axes[key] = tuple(normalized_values)
    return axes


def _validate_suite_sections(
    *,
    suite_name: str,
    mode: str,
    retrieval_axes: dict[str, tuple[MatrixScalar, ...]],
    generation_axes: dict[str, tuple[MatrixScalar, ...]],
) -> None:
    if mode == "retrieval":
        if not retrieval_axes:
            raise ValueError(f"suites.{suite_name}.retrieval is required for retrieval suites.")
        if generation_axes:
            raise ValueError(f"suites.{suite_name}.generation is not allowed for retrieval suites.")
        return

    if mode == "generation":
        if not generation_axes:
            raise ValueError(f"suites.{suite_name}.generation is required for generation suites.")
        if retrieval_axes:
            raise ValueError(f"suites.{suite_name}.retrieval is not allowed for generation suites.")
        return

    if not retrieval_axes:
        raise ValueError(f"suites.{suite_name}.retrieval is required for rag suites.")
    if not generation_axes:
        raise ValueError(f"suites.{suite_name}.generation is required for rag suites.")


def _require_mode(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    normalized = value.strip().casefold()
    if normalized not in MATRIX_MODES:
        supported = ", ".join(sorted(MATRIX_MODES))
        raise ValueError(f"{field_name} must be one of: {supported}.")
    return normalized


def _require_positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string when provided.")
    return value.strip()


def _optional_bool(value: object, *, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean when provided.")
    return value
