from __future__ import annotations

import json
import math
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from assocmem.bench_checkpoint import checkpoint_content_hash

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters" / "seqbench" / "assocmem.py"
LEARN = "learn"
INFER = "infer"
MODEL = ROOT / "adapters" / "seqbench" / "model"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def run(operation: str, *args: Path) -> subprocess.CompletedProcess[str]:
    if operation == LEARN:
        command = [
            sys.executable,
            str(ADAPTER),
            "learn",
            "--model-in",
            str(args[0]),
            "--examples",
            str(args[1]),
            "--model-out",
            str(args[2]),
            "--budget",
            "/dev/null",
            "--seed",
            "42",
        ]
    else:
        command = [
            sys.executable,
            str(ADAPTER),
            "infer",
            "--model",
            str(args[0]),
            "--requests",
            str(args[1]),
            "--output",
            str(args[2]),
            "--budget",
            "/dev/null",
        ]
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_process_contract_roundtrip_and_checkpoint_purity(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    learned = tmp_path / "learned"
    write_jsonl(
        examples,
        [
            {"id": "a", "input": "alpha input", "target": "alpha"},
            {"id": "b", "input": "beta input", "target": "beta"},
            {"id": "c", "input": "alpha again", "target": "alpha"},
        ],
    )
    result = run(LEARN, MODEL, examples, learned)
    assert result.returncode == 0, result.stderr
    before = checkpoint_content_hash(learned)

    requests = tmp_path / "requests.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        requests,
        [
            {
                "id": "generate",
                "mode": "generate",
                "input": "alpha input",
                "seed": 42,
                "max_output_tokens": 8,
            },
            {"id": "alpha", "mode": "score", "input": "alpha input", "value": "alpha"},
            {"id": "beta", "mode": "score", "input": "alpha input", "value": "beta"},
            {"id": "unknown", "mode": "score", "input": "alpha input", "value": "gamma"},
        ],
    )
    result = run(INFER, learned, requests, predictions)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["generate", "alpha", "beta", "unknown"]
    assert rows[0]["output"] == "alpha"
    assert rows[1]["log_probability"] > rows[2]["log_probability"]
    assert rows[3]["log_probability"] is None
    assert math.exp(rows[1]["log_probability"]) + math.exp(
        rows[2]["log_probability"]
    ) == pytest.approx(1.0, abs=1e-6)
    assert checkpoint_content_hash(learned) == before


def test_candidate_scores_are_absolute_and_batch_independent(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    model = tmp_path / "model"
    write_jsonl(
        examples,
        [
            {"id": "1", "input": "one", "target": "x"},
            {"id": "2", "input": "two", "target": "y"},
            {"id": "3", "input": "three", "target": "z"},
        ],
    )
    result = run(LEARN, MODEL, examples, model)
    assert result.returncode == 0, result.stderr

    single = tmp_path / "single.jsonl"
    batch = tmp_path / "batch.jsonl"
    single_out = tmp_path / "single-out.jsonl"
    batch_out = tmp_path / "batch-out.jsonl"
    write_jsonl(single, [{"id": "x", "mode": "score", "input": "one", "value": "x"}])
    write_jsonl(
        batch,
        [
            {"id": "x", "mode": "score", "input": "one", "value": "x"},
            {"id": "y", "mode": "score", "input": "one", "value": "y"},
            {"id": "z", "mode": "score", "input": "one", "value": "z"},
        ],
    )
    assert run(INFER, model, single, single_out).returncode == 0
    assert run(INFER, model, batch, batch_out).returncode == 0
    one = json.loads(single_out.read_text(encoding="utf-8"))
    many = [json.loads(line) for line in batch_out.read_text(encoding="utf-8").splitlines()]
    assert one["log_probability"] == many[0]["log_probability"]
    assert sum(math.exp(row["log_probability"]) for row in many) == pytest.approx(
        1.0, abs=1e-6
    )


def test_malformed_jsonl_leaves_no_output(tmp_path: Path) -> None:
    examples = tmp_path / "bad.jsonl"
    examples.write_text('{"id":"x","input":"x","target":"y"}\nnot-json\n', encoding="utf-8")
    output = tmp_path / "output"
    result = run(LEARN, MODEL, examples, output)
    assert result.returncode != 0
    assert not output.exists()


def test_continued_learning_appends_targets_and_preserves_input_model(
    tmp_path: Path,
) -> None:
    first_examples = tmp_path / "first.jsonl"
    second_examples = tmp_path / "second.jsonl"
    first_model = tmp_path / "first-model"
    second_model = tmp_path / "second-model"
    write_jsonl(first_examples, [{"id": "1", "input": "one", "target": "first"}])
    write_jsonl(second_examples, [{"id": "2", "input": "two", "target": "second"}])
    assert run(LEARN, MODEL, first_examples, first_model).returncode == 0
    before = checkpoint_content_hash(first_model)
    result = run(LEARN, first_model, second_examples, second_model)
    assert result.returncode == 0, result.stderr
    assert checkpoint_content_hash(first_model) == before
    connection = sqlite3.connect(second_model / "targets.sqlite3")
    try:
        rows = connection.execute(
            "SELECT target_id, value FROM targets ORDER BY target_id"
        ).fetchall()
    finally:
        connection.close()
    assert rows == [(0, "first"), (1, "second")]
