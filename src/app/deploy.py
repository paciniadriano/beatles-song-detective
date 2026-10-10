"""Uploads the app's code to the Hugging Face Space (`space.repo_id`).

    python -m src.app.deploy

Only what the Space runs goes up: app.py, config.yaml, requirements.txt,
README.md (its front matter holds the Space's settings), LICENSE and src/,
plus results.md and CLAUDE.md, which the README links to.
Data, evaluation files, tests and local secrets (.env) stay on this
machine. The search index is not here: it lives in the dataset repo,
uploaded separately with its card (src/index/dataset_card.md).

Why a script and not `hf upload ... --include "src/*"`: on Windows the CLI
expands "src/*" into the names of local files before reading it, and the
command fails with "unexpected extra arguments".

Needs a Hugging Face token with write access (`hf auth login`)."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import filter_repo_objects

from src.config import Config

# Glob patterns over paths relative to the project root, "/"-separated.
UPLOAD = ["app.py", "config.yaml", "requirements.txt", "README.md", "LICENSE", "results.md", "CLAUDE.md", "src/*"]
SKIP = ["*__pycache__*", "*.pyc"]


def files_to_upload(root: Path) -> list[str]:
    """The files the upload will send, with the same filter the Hub client
    applies, so they can be printed and checked before anything is sent."""
    paths = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()]
    return sorted(filter_repo_objects(paths, allow_patterns=UPLOAD, ignore_patterns=SKIP))


def main() -> None:
    root = Path(".")
    config = Config.load(root / "config.yaml")
    files = files_to_upload(root)
    print(f"Uploading {len(files)} files to the Space {config.space.repo_id}:")
    for name in files:
        print(f"  {name}")

    commit = HfApi().upload_folder(
        repo_id=config.space.repo_id,
        repo_type="space",
        folder_path=root,
        allow_patterns=UPLOAD,
        ignore_patterns=SKIP,
        commit_message="Update the app",
    )
    print(f"Done: {commit.commit_url}")


if __name__ == "__main__":
    main()
