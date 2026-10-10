"""Finds the Qdrant store the app searches.

- On the Space, the environment variable named by `space.dataset_repo_env`
  holds the Hugging Face dataset repo with the published store
  (src.index.publish). It is downloaded and copied to
  `space.qdrant_runtime_path`: the download cache holds links to read-only
  files, and local Qdrant needs a folder it can write its `.lock` into
  (CLAUDE.md, trap 8).
- Locally, with the variable unset, the app opens `paths.qdrant_store`, the
  folder publish.py wrote, and the key in `.env` is used when the key box is
  empty (app.py). The Space never reads a key from its environment: every
  visitor pays with their own."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from src.config import Config
from src.index.collections import LOCK_FILE

STORE_FOLDER = "qdrant_store"  # its name inside the dataset repo


def runs_on_space(config: Config) -> bool:
    """The Space is told apart from a local run by the dataset repo variable:
    only the Space sets it."""
    return bool(os.getenv(config.space.dataset_repo_env))


def prepare_store(config: Config) -> Path:
    """Returns the folder to open with open_searcher()."""
    if not runs_on_space(config):
        store = config.paths.qdrant_store
        if not store.exists():
            raise FileNotFoundError(f"{store} does not exist: run python -m src.index.publish first")
        return store

    repo = os.environ[config.space.dataset_repo_env]
    downloaded = Path(snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=f"{STORE_FOLDER}/*"))
    destination = config.space.qdrant_runtime_path
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(downloaded / STORE_FOLDER, destination)  # copies the files the links point to
    (destination / LOCK_FILE).unlink(missing_ok=True)
    return destination
