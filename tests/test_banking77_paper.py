import csv

import pytest
import torch

from assocmem.config import TextEncoderConfig
from assocmem.encoding import SignedHashTextEncoder
from assocmem.experiments.banking77_paper import (
    MODEL_ORDER,
    RAW_FIELDS,
    SUMMARY_FIELDS,
    PaperExperimentConfig,
    _build_model,
    _encode,
    _run_model,
    aggregate_rows,
    duplicate_test_ids,
    knn_capacity_for_budget,
    paired_interval,
    run_paper_experiment,
    select_shared_birth_ids,
)
from assocmem.experiments.cli import main as experiment_cli


def _measurement(value):
    from assocmem.experiments.banking77_paper import METRIC_FIELDS

    return {field: value for field in METRIC_FIELDS}


def test_aggregation_and_paired_interval_use_sample_variation():
    rows = []
    for model in MODEL_ORDER:
        for seed, value in enumerate((1.0, 3.0)):
            rows.append({"model": model, "seed": seed, **_measurement(value)})
    result = aggregate_rows(rows)
    assert len(result) == len(MODEL_ORDER) + 4
    assert result[0]["prequential_nll_bits"] == "2.000000 ± 1.414214"
    assert paired_interval([1.0, 3.0])[1] < 0 < paired_interval([1.0, 3.0])[2]
    shared = result[len(MODEL_ORDER)]
    assert shared["row_type"] == "comparison"
    assert shared["prequential_nll_advantage_bits"] == "0.000000"


def test_small_end_to_end_writes_summary_and_raw_csv(tmp_path):
    train = [
        ("alpha one", 0),
        ("alpha two", 0),
        ("beta one", 1),
        ("beta two", 1),
    ]
    test = [("alpha one", 0), ("beta test", 1)]
    summary = tmp_path / "banking77_results_v2.csv"
    raw = tmp_path / "banking77_runs_v2.csv"
    config = PaperExperimentConfig(
        seeds=(0,),
        dimension=32,
        max_features=12,
        capacity=4,
        top_k=2,
        key_nnz=12,
        key_scale=4.0,
        byte_budget=1024 * 1024,
        torch_threads=1,
    )
    rows = run_paper_experiment(
        train_records=train,
        test_records=test,
        output=summary,
        raw_output=raw,
        config=config,
    )
    assert len(rows) == len(MODEL_ORDER) + 4
    with summary.open(encoding="utf-8", newline="") as handle:
        saved_summary = list(csv.DictReader(handle))
    with raw.open(encoding="utf-8", newline="") as handle:
        saved_raw = list(csv.DictReader(handle))
    assert tuple(saved_summary[0]) == SUMMARY_FIELDS
    assert tuple(saved_raw[0]) == RAW_FIELDS
    assert len(saved_raw) == len(MODEL_ORDER)
    assert {row["excluded_duplicate_test_examples"] for row in saved_raw} == {"1"}
    assert {row["hash_seed"] for row in saved_raw} == {"0"}
    assert sorted(tmp_path.iterdir()) == sorted([summary, raw])

    shared = {row["model"]: row for row in saved_raw if row["model"].startswith("shared_")}
    assert shared["shared_learned"]["active_atoms"] == "4"
    assert shared["shared_frozen"]["active_atoms"] == "4"
    assert shared["shared_learned"]["shared_birth_count"] == "4"


def test_duplicate_filter_is_normalized_and_label_independent():
    train = [("Card—PAYMENT!", 0), ("other", 1)]
    test = [("card payment", 1), ("new", 0)]
    assert duplicate_test_ids(train, test) == {0}


def test_knn_capacity_is_byte_fair_and_can_retain_full_banking_train():
    config = PaperExperimentConfig()
    assert knn_capacity_for_budget(config, 77, 10_003) == 10_003


def test_shared_branches_receive_same_births_and_key_supports():
    records = [("alpha one", 0), ("beta two", 1), ("mixed three", 0)]
    config = PaperExperimentConfig(
        seeds=(0,),
        dimension=32,
        max_features=8,
        capacity=2,
        top_k=2,
        key_nnz=8,
        key_scale=4.0,
        byte_budget=1024 * 1024,
        torch_threads=1,
    )
    encoder = SignedHashTextEncoder(TextEncoderConfig(dimension=32, max_features=8, hash_seed=0))
    examples = _encode(records, encoder)
    births = select_shared_birth_ids(examples, config.capacity, config.shared_birth_seed)
    models = {}
    for name in ("shared_learned", "shared_frozen"):
        model = _build_model(
            name,
            encoder=encoder,
            num_classes=2,
            train_examples=len(examples),
            config=config,
        )
        _run_model(name, model, examples, examples, examples, 2, births)
        models[name] = model
    learned = models["shared_learned"].memory
    frozen = models["shared_frozen"].memory
    assert set(learned.origin_id[: learned.size].tolist()) == births
    assert torch.equal(
        learned.W[: learned.size] != 0,
        frozen.W[: frozen.size] != 0,
    )


def test_atomic_writer_preserves_previous_result_on_failure(tmp_path, monkeypatch):
    from assocmem.experiments import banking77_paper

    output = tmp_path / "result.csv"
    output.write_text("previous", encoding="utf-8")

    def fail(*args, **kwargs):
        raise RuntimeError("write failed")

    monkeypatch.setattr(csv.DictWriter, "writerows", fail)
    with pytest.raises(RuntimeError, match="write failed"):
        banking77_paper._write_csv_atomic(output, [{"model": "x"}])
    assert output.read_text(encoding="utf-8") == "previous"


def test_public_cli_has_no_suite_or_preflight(capsys):
    with pytest.raises(SystemExit) as error:
        experiment_cli(["--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "suite" not in help_text
    assert "preflight" not in help_text
