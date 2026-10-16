"""Saves the questions asked on the Space to a private Hugging Face dataset.

The Space's disk is wiped on every restart, so each record is appended to a
local JSONL file and `CommitScheduler` pushes that folder to the dataset repo
every few minutes, in a background thread. One file per start of the app
(a random name), so two starts never overwrite each other's file. Only the
last few minutes before a restart can be lost.

The repo is created as private on the first push; the Space's `HF_TOKEN`
secret must be allowed to write to it. Used only on the Space (app.py)."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import CommitScheduler


def open_question_log(repo_id: str, every_minutes: float, folder: Path) -> Callable[[dict], None]:
    """Returns the function chat.respond() calls with each record."""
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / f"questions_{uuid.uuid4().hex}.jsonl"
    scheduler = CommitScheduler(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        folder_path=folder,
        path_in_repo="questions",
        every=every_minutes,
    )

    def save(record: dict) -> None:
        # The lock keeps a push from reading the file halfway through a line.
        with scheduler.lock:
            with file.open("a", encoding="utf-8") as out:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    return save
