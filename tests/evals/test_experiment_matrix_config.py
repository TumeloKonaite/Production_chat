from __future__ import annotations

from pathlib import Path

from evals.matrix.config_loader import load_experiment_matrix_config


def test_load_experiment_matrix_config_parses_named_suites(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment_matrix.yaml"
    config_path.write_text(
        "\n".join(
            [
                "suites:",
                "  retrieval_smoke:",
                "    mode: retrieval",
                "    max_combinations: 4",
                "    retrieval:",
                "      retriever_type:",
                "        - vector",
                "      top_k:",
                "        - 5",
                "  rag_medium:",
                "    mode: rag",
                "    max_combinations: 8",
                "    retrieval:",
                "      retriever_type:",
                "        - vector",
                "      top_k:",
                "        - 3",
                "        - 5",
                "    generation:",
                "      llm_model:",
                "        - gpt-4.1-mini",
                "      prompt_version:",
                "        - v1_professional",
                "        - v2_warm_conversational",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_experiment_matrix_config(config_path)

    assert sorted(config.suites) == ["rag_medium", "retrieval_smoke"]
    assert config.suites["retrieval_smoke"].mode == "retrieval"
    assert config.suites["retrieval_smoke"].retrieval["retriever_type"] == ("vector",)
    assert config.suites["rag_medium"].generation["prompt_version"] == (
        "v1_professional",
        "v2_warm_conversational",
    )


def test_load_experiment_matrix_config_requires_sections_for_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment_matrix.yaml"
    config_path.write_text(
        "\n".join(
            [
                "suites:",
                "  generation_smoke:",
                "    mode: generation",
                "    max_combinations: 4",
                "    retrieval:",
                "      top_k:",
                "        - 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        load_experiment_matrix_config(config_path)
    except ValueError as exc:
        assert str(exc) == "suites.generation_smoke.generation is required for generation suites."
    else:  # pragma: no cover
        raise AssertionError("Expected generation suite validation to fail.")


def test_load_experiment_matrix_config_rejects_unsupported_axes(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment_matrix.yaml"
    config_path.write_text(
        "\n".join(
            [
                "suites:",
                "  generation_smoke:",
                "    mode: generation",
                "    max_combinations: 4",
                "    generation:",
                "      max_tokens:",
                "        - 512",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        load_experiment_matrix_config(config_path)
    except ValueError as exc:
        assert "suites.generation_smoke.generation.max_tokens is not supported." in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected unsupported generation axis validation to fail.")
