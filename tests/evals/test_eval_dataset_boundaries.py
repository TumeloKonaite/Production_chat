from __future__ import annotations

import json
from pathlib import Path
import re

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "evals" / "datasets"
PORTFOLIO_DATASET_PATH = DATASET_DIR / "portfolio_eval_dataset.jsonl"
MODEL_DATASET_PATH = DATASET_DIR / "model_eval_dataset.jsonl"
PROMPT_DATASET_PATH = DATASET_DIR / "prompt_eval_questions.jsonl"


def _load_questions(path: Path) -> list[str]:
    questions: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        questions.append(str(payload["question"]))
    return questions


def _normalize_question(question: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", question.casefold()).strip()
    return re.sub(r"\s+", " ", normalized)


def test_eval_datasets_do_not_reuse_exact_questions() -> None:
    portfolio_questions = {
        _normalize_question(question) for question in _load_questions(PORTFOLIO_DATASET_PATH)
    }
    model_questions = {
        _normalize_question(question) for question in _load_questions(MODEL_DATASET_PATH)
    }
    prompt_questions = {
        _normalize_question(question) for question in _load_questions(PROMPT_DATASET_PATH)
    }

    assert portfolio_questions.isdisjoint(model_questions)
    assert portfolio_questions.isdisjoint(prompt_questions)
    assert model_questions.isdisjoint(prompt_questions)
