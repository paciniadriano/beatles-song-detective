"""Tests for deploy.py: what goes up to the Space, on a small fake project,
so no token or network is needed."""

from pathlib import Path

from src.app.deploy import files_to_upload


def _touch(root: Path, *names: str) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")


def test_only_what_the_space_runs_is_uploaded(tmp_path: Path) -> None:
    _touch(
        tmp_path,
        "app.py", "config.yaml", "requirements.txt", "README.md", "LICENSE",
        "results.md", "CLAUDE.md", "src/__init__.py", "src/app/ui.py", "src/rag/prompt.md",
        # must stay on this machine
        ".env", "notes.md", "scratch.py", "requirements-dev.txt",
        "data/qdrant_store/meta.json", "eval/generation/set_v1.yaml", "tests/test_app.py",
        "src/__pycache__/config.cpython-312.pyc", "src/app/__pycache__/ui.cpython-312.pyc",
    )
    assert files_to_upload(tmp_path) == [
        "CLAUDE.md", "LICENSE", "README.md", "app.py", "config.yaml", "requirements.txt",
        "results.md", "src/__init__.py", "src/app/ui.py", "src/rag/prompt.md",
    ]
