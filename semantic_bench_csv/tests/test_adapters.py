from benchprep.adapters import ADAPTERS, semantic_parsing_row


def test_babi_emits_full_and_oracle() -> None:
    row = {
        "story": [
            {"id": "1", "type": 0, "text": "Mary went to the kitchen.", "supporting_ids": [], "answer": ""},
            {"id": "2", "type": 0, "text": "John went to the office.", "supporting_ids": [], "answer": ""},
            {"id": "3", "type": 1, "text": "Where is Mary?", "supporting_ids": ["1"], "answer": "kitchen"},
        ]
    }
    rows = list(
        ADAPTERS["babi"].convert(
            row,
            config="en-valid-qa1",
            split="test",
            source_key="en-valid/qa1_test.txt",
            source_index=0,
            variants={"full", "oracle"},
        )
    )
    assert [item.variant for item in rows] == ["full", "oracle"]
    assert "John went" in rows[0].input
    assert "John went" not in rows[1].input
    assert rows[0].expected_output == "kitchen"


def test_clutrr_query_and_oracle() -> None:
    row = {
        "id": "x",
        "story": "Noise. Alice is Bob's mother.",
        "clean_story": "Alice is Bob's mother.",
        "query": ["Bob", "Alice"],
        "target_text": "mother",
        "task_name": "task_1.2",
    }
    rows = list(
        ADAPTERS["clutrr"].convert(
            row,
            config="gen",
            split="test",
            source_key="gen/test/0000.parquet",
            source_index=0,
            variants={"full", "oracle"},
        )
    )
    assert len(rows) == 2
    assert rows[1].variant == "oracle"
    assert "Noise" not in rows[1].input
    assert rows[0].expected_output == "mother"


def test_proofwriter_normalizes_label() -> None:
    row = {"id": "p", "theory": "Bob is blue.", "question": "Bob is blue.", "answer": True, "maxD": 0}
    result = list(
        ADAPTERS["proofwriter"].convert(
            row,
            config="default",
            split="train",
            source_key="data/train-0.parquet",
            source_index=0,
            variants={"full"},
        )
    )[0]
    assert result.expected_output == "true"
    assert result.expected_probability == 1.0


def test_semantic_parsing_row() -> None:
    row = semantic_parsing_row(
        dataset="slog",
        config="cogs_LF",
        split="generalization",
        source_index=3,
        fields=["A dog ran.", "run.agent(x1,dog)", "long_distance"],
        source_path="data/example.tsv",
    )
    assert row is not None
    assert row.task == "long_distance"
    assert row.expected_output == "run.agent(x1,dog)"


def test_mrcr_serializes_messages() -> None:
    row = {
        "id": "m1",
        "prompt": '[{"role":"user","content":"Remember alpha."},{"role":"user","content":"What was it?"}]',
        "answer": "alpha",
        "random_string_to_prepend": "xyz",
    }
    result = list(
        ADAPTERS["mrcr"].convert(
            row,
            config="2needle",
            split="test",
            source_key="2needle/2needle_0.parquet",
            source_index=0,
            variants={"full"},
        )
    )[0]
    assert "[USER]" in result.input
    assert "Remember alpha" in result.input
    assert result.expected_output == "alpha"
    assert result.split == "test"


def test_structured_ids_include_source_location() -> None:
    row = {"prompt": "Remember alpha.", "answer": "alpha"}
    left = list(
        ADAPTERS["mrcr"].convert(
            row,
            config="2needle",
            split="test",
            source_key="2needle/2needle_0.parquet",
            source_index=0,
            variants={"full"},
        )
    )[0]
    right = list(
        ADAPTERS["mrcr"].convert(
            row,
            config="2needle",
            split="test",
            source_key="2needle/2needle_1.parquet",
            source_index=0,
            variants={"full"},
        )
    )[0]
    assert left.id != right.id


def test_proofwriter_uses_row_config_and_all_proofs() -> None:
    row = {
        "id": "theory-1",
        "config": "depth-5",
        "theory": "Bob is blue.",
        "question": "Bob is blue.",
        "answer": True,
        "maxD": 5,
        "allProofs": "proof payload",
    }
    result = list(
        ADAPTERS["proofwriter"].convert(
            row,
            config="row_config",
            split="train",
            source_key="data/train-0.parquet",
            source_index=8,
            variants={"full"},
        )
    )[0]
    assert result.config == "depth-5"
    assert result.metadata["proof"] == "proof payload"


def test_babilong_converts_record() -> None:
    row = {
        "input": "Mary went to the kitchen. John went to the hall.",
        "question": "Where is Mary?",
        "target": "kitchen",
    }
    result = list(
        ADAPTERS["babilong"].convert(
            row,
            config="8k",
            split="qa1",
            source_key="data/qa1/8k.json",
            source_index=7,
            variants={"full"},
        )
    )[0]
    assert result.config == "8k"
    assert result.task == "qa1"
    assert result.split == "test"
    assert result.expected_output == "kitchen"
