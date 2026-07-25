import math

import pytest
import torch

from assocmem.config import InsertionConfig, MemoryConfig, UpdateConfig
from assocmem.encoding import TernaryQuery
from assocmem.memory import AssociativeMemory


def query(dimension=16):
    return TernaryQuery(
        dimension,
        torch.tensor([1, 4, 9], dtype=torch.int64),
        torch.tensor([1, -1, 1], dtype=torch.float32),
    )


def config(**overrides):
    values = {
        "dimension": 16,
        "num_classes": 3,
        "capacity": 4,
        "top_k": 2,
        "key_nnz": 3,
        "key_scale": 6.0,
        "insertion": InsertionConfig(
            minimum_surprise_bits=0.1,
            surprise_margin_from_uniform_bits=4,
            min_gain_bits=0,
            value_scale=4,
        ),
    }
    values.update(overrides)
    return MemoryConfig(**values)


def test_empty_memory_is_prior_and_read_is_pure():
    memory = AssociativeMemory(config())
    before = (memory.step, memory.revision, memory.base_counts.clone())
    result = memory.read(query())
    assert torch.allclose(result.probabilities, torch.full((3,), 1 / 3))
    assert result.background_responsibility == 1
    assert before[:2] == (memory.step, memory.revision)
    assert torch.equal(before[2], memory.base_counts)


def test_inserted_key_has_zero_energy_and_fixed_norm():
    memory = AssociativeMemory(config())
    report = memory.observe(query(), 2)
    assert report.insertion.inserted
    assert memory.size == 1
    assert torch.isclose(memory.W[0].abs().sum(), torch.tensor(6.0))
    assert torch.count_nonzero(memory.W[0]) <= 3
    assert torch.isclose(memory.read(query()).energies[0], torch.tensor(0.0), atol=1e-6)


def test_second_observation_does_not_increase_current_loss():
    memory = AssociativeMemory(config())
    memory.observe(query(), 2)
    read = memory.read(query())
    report = memory.observe(query(), 2, pre_read=read)
    assert report.learn.loss_after_bits <= report.learn.loss_before_bits + 1e-7
    assert memory.base_counts.tolist() == [0, 0, 2]


def test_stale_read_is_rejected():
    memory = AssociativeMemory(config())
    stale = memory.read(query())
    memory.observe(query(), 1, pre_read=stale)
    with pytest.raises(RuntimeError, match="stale"):
        memory.observe(query(), 1, pre_read=stale)


def test_full_capacity_evicts_without_zero_key():
    memory = AssociativeMemory(config(capacity=1))
    memory.observe(query(), 0)
    other = TernaryQuery(16, torch.tensor([2, 3]), torch.tensor([-1.0, 1.0], dtype=torch.float32))
    memory.observe(other, 2)
    assert memory.size == 1
    assert torch.isclose(memory.W[0].abs().sum(), torch.tensor(6.0), atol=1e-5)
    assert math.isfinite(float(memory.V[0].norm()))


def test_frozen_keys_only_updates_values():
    memory = AssociativeMemory(
        config(update=UpdateConfig(train_keys=False, learning_rate_value=1.0))
    )
    memory.observe(query(), 0)
    old = memory.W.clone()
    memory.observe(query(), 1)
    assert torch.equal(old[0], memory.W[0])


def test_read_owner_snapshot_fingerprint_and_origin():
    cfg = config(encoder_fingerprint="encoder-a")
    tagged = TernaryQuery(
        16,
        torch.tensor([1, 4, 9]),
        torch.tensor([1.0, -1.0, 1.0]),
        "encoder-a",
    )
    first = AssociativeMemory(cfg)
    second = AssociativeMemory(cfg)
    read = first.read(tagged)
    tagged.indices[0] = 2
    assert read.query.indices.tolist() == [1, 4, 9]
    with pytest.raises(RuntimeError, match="another memory"):
        second.observe(read.query, 0, pre_read=read)
    report = first.observe(read.query, 0, pre_read=read, origin_id=123)
    assert report.read_after.origin_ids.tolist() == [123]
    mutated = first.read(report.read_after.query)
    mutated.query.values[0] *= -1
    with pytest.raises(RuntimeError, match="mutated"):
        first.observe(mutated.query, 0, pre_read=mutated)
    wrong = TernaryQuery(16, torch.tensor([1]), torch.tensor([1.0]), "encoder-b")
    with pytest.raises(ValueError, match="fingerprint"):
        first.read(wrong)


def test_observe_rolls_back_if_insertion_raises(monkeypatch):
    memory = AssociativeMemory(config())
    memory.observe(query(), 0)
    before = memory.clone()

    def fail(*args, **kwargs):
        raise RuntimeError("injected")

    monkeypatch.setattr(memory, "_insert", fail)
    with pytest.raises(RuntimeError, match="injected"):
        memory.observe(query(), 1)
    for name in ("W", "V", "usage", "origin_id", "atom_uid", "base_counts"):
        assert torch.equal(getattr(memory, name), getattr(before, name))
    assert (memory.size, memory.step, memory.revision, memory.next_atom_uid) == (
        before.size,
        before.step,
        before.revision,
        before.next_atom_uid,
    )


def test_novelty_gate_does_not_insert_label_noise_and_support_can_turn_over():
    memory = AssociativeMemory(
        config(
            update=UpdateConfig(
                learning_rate_key=0.5,
                learning_rate_value=0.2,
                support_replacements=1,
            )
        )
    )
    memory.observe(query(), 0)
    changed = TernaryQuery(
        16,
        torch.tensor([1, 4, 10]),
        torch.tensor([1.0, -1.0, 1.0]),
    )
    report = memory.observe(changed, 1)
    assert report.learn.support_added == report.learn.support_removed == 1
    noise_memory = AssociativeMemory(config())
    noise_memory.observe(query(), 0)
    size = noise_memory.size
    for target in (2, 0, 1, 2):
        report = noise_memory.observe(query(), target)
        assert not report.insertion.inserted
        assert report.insertion.reason in {
            "below-threshold",
            "not-novel-energy",
            "not-novel-background",
        }
    assert noise_memory.size == size


def test_forced_and_skipped_insertion_modes_are_explicit():
    memory = AssociativeMemory(
        config(insertion=InsertionConfig(enabled=False, minimum_surprise_bits=100))
    )
    skipped = memory.observe(query(), 0, insertion_mode="skip")
    assert skipped.insertion.reason == "skipped"
    assert memory.size == 0
    reference = torch.tensor([0.2, 0.3, 0.5])
    forced = memory.observe(
        query(),
        1,
        insertion_mode="force",
        insertion_reference_probabilities=reference,
        origin_id=17,
    )
    assert forced.insertion.inserted
    assert forced.insertion.reason == "forced"
    assert memory.origin_id[0] == 17


def test_fixed_support_changes_weights_without_changing_coordinates():
    memory = AssociativeMemory(
        config(
            update=UpdateConfig(
                learning_rate_key=1.0,
                learning_rate_value=0.2,
                support_replacements=0,
                fixed_key_support=True,
            )
        )
    )
    memory.observe(query(), 0, insertion_mode="force")
    old = memory.W[0].clone()
    changed = TernaryQuery(
        16,
        torch.tensor([1, 4]),
        torch.tensor([-1.0, -1.0]),
    )
    memory.observe(changed, 1, insertion_mode="skip")
    assert torch.equal(old != 0, memory.W[0] != 0)
    assert not torch.equal(old, memory.W[0])
