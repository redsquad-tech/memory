import math

from assocmem.experiments.banking77_mixture import (
    EXPECTED_VALIDATION_HASH,
    GEOMETRY_FIELDS,
    METRIC_FIELDS,
    PROTOCOL_VERSION,
    aggregate_rows,
    stratified_validation_split,
)
from assocmem.experiments.banking77_paper import _normalize_text
from assocmem.experiments.datasets import iter_banking77


def _row(*, seed, key_mode, nll, accuracy, geometry):
    row = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": "B",
        "stage": "validation",
        "key_mode": key_mode,
        "training_decoder": "categorical_mixture",
        "eval_decoder": "categorical_mixture",
        "training_prior": "uniform",
        "eval_prior": "uniform",
        "seed": seed,
    }
    row.update({field: math.nan for field in METRIC_FIELDS})
    row.update(
        {
            "eval_nll_bits": nll,
            "eval_accuracy_pct": accuracy,
            **{field: geometry for field in GEOMETRY_FIELDS},
        }
    )
    return row


def test_official_validation_split_is_frozen_and_has_no_text_leakage():
    records = list(iter_banking77("data", "train"))
    training, validation, digest = stratified_validation_split(records)
    assert (len(training), len(validation)) == (8011, 1992)
    assert digest == EXPECTED_VALIDATION_HASH
    assert set(training).isdisjoint(validation)
    assert set(training) | set(validation) == set(range(len(records)))
    train_texts = {_normalize_text(records[row_id][0]) for row_id in training}
    validation_texts = {_normalize_text(records[row_id][0]) for row_id in validation}
    assert train_texts.isdisjoint(validation_texts)


def test_mixture_aggregation_uses_paired_learned_advantages():
    rows = []
    for seed in (0, 1):
        rows.append(
            _row(
                seed=seed,
                key_mode="learned",
                nll=1.0 + seed,
                accuracy=80.0 + seed,
                geometry=0.7 + 0.1 * seed,
            )
        )
        rows.append(
            _row(
                seed=seed,
                key_mode="frozen",
                nll=1.5 + seed,
                accuracy=78.0 + seed,
                geometry=0.5 + 0.1 * seed,
            )
        )
    summary = aggregate_rows(rows)
    comparison = next(row for row in summary if row["row_type"] == "comparison")
    assert comparison["nll_advantage_bits"] == "0.500000"
    assert comparison["accuracy_advantage_pp"] == "2.000000"
    assert comparison["nll_favorable_seeds"] == "2"
    assert comparison["accuracy_favorable_seeds"] == "2"
    assert comparison["purity_advantage"] == "0.200000"


def test_posthoc_aggregation_reports_decoder_swap_effect():
    rows = []
    for seed in (0, 1):
        for decoder, nll, accuracy in (
            ("legacy_gated_logit", 4.0 + seed, 60.0 + seed),
            ("categorical_mixture", 2.0 + seed, 70.0 + seed),
        ):
            row = {
                "protocol_version": PROTOCOL_VERSION,
                "experiment": "A",
                "stage": "official_test",
                "key_mode": "learned",
                "training_decoder": "legacy_gated_logit",
                "eval_decoder": decoder,
                "training_prior": "empirical",
                "eval_prior": "empirical",
                "seed": seed,
            }
            row.update({field: math.nan for field in METRIC_FIELDS})
            row.update(
                {
                    "eval_nll_bits": nll,
                    "eval_accuracy_pct": accuracy,
                }
            )
            rows.append(row)
    summary = aggregate_rows(rows)
    comparison = next(row for row in summary if row["row_type"] == "comparison")
    assert comparison["comparison"] == "categorical_mixture_vs_legacy_decoder"
    assert comparison["nll_advantage_bits"] == "2.000000"
    assert comparison["accuracy_advantage_pp"] == "10.000000"
    assert comparison["nll_favorable_seeds"] == "2"
