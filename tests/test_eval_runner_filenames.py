from evals.runners.run_generation_eval import _safe_file_stem as generation_safe_file_stem
from evals.runners.run_model_eval import _safe_file_stem as model_safe_file_stem


def test_generation_eval_safe_file_stem_sanitizes_provider_style_model_ids() -> None:
    assert (
        generation_safe_file_stem("openrouter:openai/gpt-oss-120b")
        == "openrouter_openai_gpt-oss-120b"
    )


def test_model_eval_safe_file_stem_sanitizes_provider_style_model_ids() -> None:
    assert (
        model_safe_file_stem("openrouter:anthropic/claude-3.5-sonnet")
        == "openrouter_anthropic_claude-3.5-sonnet"
    )
