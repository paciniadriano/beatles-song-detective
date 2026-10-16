"""Downloads the questions saved by the Space and shows them.

    python -m src.app.show_questions

Prints one line per question, oldest first, and writes every field to
`data/questions.csv` (gitignored: visitors' questions stay off GitHub), which
opens in Excel. Needs a Hugging Face login with access to the private
dataset (`hf auth login`). Nothing here runs on the Space.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from src.config import Config

CSV_PATH = Path("data") / "questions.csv"
COLUMNS = ["time", "question", "answer", "queries", "cost_usd", "seconds", "completed"]


def load_questions(folder: Path) -> list[dict]:
    """Every record in every file (one file per start of the Space), oldest first."""
    records: list[dict] = []
    for file in sorted(folder.glob("questions/*.jsonl")):
        with file.open(encoding="utf-8") as lines:
            for line in lines:
                if line.strip():
                    records.append(json.loads(line))
    records.sort(key=lambda record: record["time"])
    return records


def write_csv(records: list[dict], path: Path) -> None:
    """utf-8-sig so Excel reads the accents and dashes correctly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "queries": " | ".join(record.get("queries", []))})


def main() -> None:
    config = Config.load(Path("config.yaml"))
    folder = Path(snapshot_download(config.space.questions_repo, repo_type="dataset"))
    records = load_questions(folder)

    for record in records:
        mark = "" if record["completed"] else "  (not completed)"
        print(f"{record['time']}  {record['question']}{mark}")
    write_csv(records, CSV_PATH)
    print(f"\n{len(records)} questions; all fields in {CSV_PATH}")


if __name__ == "__main__":
    main()
