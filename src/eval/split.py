"""Assigns every question in the retrieval pool to dev or test.

Decisions are made on dev only; test is run once, at the end (CLAUDE.md,
"Experiment order"). The split is stratified by doc_type, so every
per-doc_type row exists in both halves: a plain random split could leave
studio (~30 questions) nearly absent from one of them.

Deterministic for a given seed and pool, so running it again gives the same
split. It rewrites the pool file in place, filling each question's `split`.

    python -m src.eval.split"""

from __future__ import annotations

import random
from pathlib import Path

from src.config import Config
from src.eval.question_file import load_questions, save_questions
from src.schemas import EvalQuestion


def group_by_doc_type(questions: list[EvalQuestion]) -> dict[str, list[EvalQuestion]]:
    groups: dict[str, list[EvalQuestion]] = {}
    for question in questions:
        groups.setdefault(question.doc_type, []).append(question)
    return groups


def split_by_doc_type(questions: list[EvalQuestion], dev_ratio: float, seed: int) -> list[EvalQuestion]:
    """Within each doc_type: shuffle, then the first round(n * dev_ratio) go
    to dev and the rest to test. Returns updated copies in the original file
    order. doc_types are visited in sorted order so the result does not depend
    on which type happens to come first in the file."""
    rng = random.Random(seed)
    split_of: dict[str, str] = {}
    groups = group_by_doc_type(questions)
    for doc_type in sorted(groups):
        group = list(groups[doc_type])
        rng.shuffle(group)
        dev_count = round(len(group) * dev_ratio)
        for index, question in enumerate(group):
            split_of[question.question_id] = "dev" if index < dev_count else "test"
    return [question.model_copy(update={"split": split_of[question.question_id]}) for question in questions]


def split_table(questions: list[EvalQuestion]) -> str:
    """dev and test counts per doc_type, for a quick look after splitting."""
    lines = [f"{'doc_type':12s} {'dev':>5s} {'test':>5s}"]
    groups = group_by_doc_type(questions)
    for doc_type in sorted(groups):
        dev = sum(1 for question in groups[doc_type] if question.split == "dev")
        test = len(groups[doc_type]) - dev
        lines.append(f"{doc_type:12s} {dev:5d} {test:5d}")
    total_dev = sum(1 for question in questions if question.split == "dev")
    lines.append(f"{'total':12s} {total_dev:5d} {len(questions) - total_dev:5d}")
    return "\n".join(lines)


def main() -> None:
    config = Config.load(Path("config.yaml"))
    pool_path = config.eval.generate.output
    questions = load_questions(pool_path)
    split = split_by_doc_type(questions, config.eval.dev_ratio, config.eval.split_seed)
    save_questions(pool_path, split)
    print(f"{len(split)} questions split -> {pool_path}\n")
    print(split_table(split))


if __name__ == "__main__":
    main()
