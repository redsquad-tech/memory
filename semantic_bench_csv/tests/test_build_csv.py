from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchprep.sources import SourceSpec
from build_csv import build_tasks
from validate_csv import validate


def json_source(path: Path, rows: list[dict[str, str]], key: str) -> SourceSpec:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    return SourceSpec(
        dataset="babilong",
        config="0k",
        split="qa1",
        kind="json",
        path=path,
        source_key=key,
    )


def test_atomic_failure_preserves_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "tasks.csv"
    output.write_text("old-result\n", encoding="utf-8")
    good = json_source(
        tmp_path / "good.json",
        [{"input": "context", "question": "question", "target": "answer"}],
        "data/qa1/0k.json",
    )
    bad = json_source(
        tmp_path / "bad.json",
        [{"input": "context", "question": "question", "target": ""}],
        "data/qa2/0k.json",
    )
    with pytest.raises(ValueError, match="empty fields"):
        build_tasks(
            output=output,
            names=["babilong"],
            sources_by_dataset={"babilong": [good, bad]},
            variants=["full"],
            mrcr_needles=[],
            enforce_expected_counts=False,
        )
    assert output.read_text(encoding="utf-8") == "old-result\n"
    assert not list(tmp_path.glob(".tasks.csv.*"))


def test_large_csv_field_builds_and_validates(tmp_path: Path) -> None:
    source = json_source(
        tmp_path / "large.json",
        [{"input": "x" * 200_000, "question": "question", "target": "answer"}],
        "data/qa1/0k.json",
    )
    output = tmp_path / "tasks.csv"
    total, counts = build_tasks(
        output=output,
        names=["babilong"],
        sources_by_dataset={"babilong": [source]},
        variants=["full"],
        mrcr_needles=[],
        enforce_expected_counts=False,
    )
    assert total == 1
    assert counts["babilong"] == 1
    assert validate(output) == (1, [])
