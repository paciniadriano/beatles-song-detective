"""Reads and writes evaluation question files: the retrieval pool as JSON
Lines (one EvalQuestion per line), the generation set as YAML (written and
edited by hand). Kept in one place so every script that loads questions
validates them the same way."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.schemas import EvalQuestion


def load_questions(path: Path) -> list[EvalQuestion]:
    """Validates every line against EvalQuestion. A duplicate question_id
    raises: the split and the audit key questions by id, so a duplicate would
    silently give two questions the same fate."""
    lines = path.read_text(encoding="utf-8").splitlines()
    questions = [EvalQuestion.model_validate_json(line) for line in lines if line.strip()]
    _refuse_duplicates(path, questions)
    return questions


def load_question_set(path: Path) -> list[EvalQuestion]:
    """The generation set: a YAML list of questions. Keys EvalQuestion does not
    know (`article`, `note`) are there for the person editing, and ignored."""
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))
    questions = [EvalQuestion.model_validate(entry) for entry in entries]
    _refuse_duplicates(path, questions)
    return questions


def _refuse_duplicates(path: Path, questions: list[EvalQuestion]) -> None:
    seen_ids: set[str] = set()
    for question in questions:
        if question.question_id in seen_ids:
            raise ValueError(f"{path}: duplicate question_id {question.question_id}")
        seen_ids.add(question.question_id)


def save_questions(path: Path, questions: list[EvalQuestion]) -> None:
    """Writes the whole file at once, in the order given."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(question.model_dump_json() + "\n" for question in questions)
    path.write_text(text, encoding="utf-8")
