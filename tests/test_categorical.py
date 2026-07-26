from dataclasses import replace

import pytest
import torch

from assocmem.baselines import ExactKNNBaseline
from assocmem.categorical import (
    CategoricalAssociativeMemory,
    CategoricalMemoryConfig,
    CategoricalMixtureDecoder,
    CategoricalUpdateConfig,
)
from assocmem.encoding import TernaryQuery


def query(indices, values, dimension=16):
    return TernaryQuery(
        dimension,
        torch.tensor(indices, dtype=torch.int64),
        torch.tensor(values, dtype=torch.float32),
    )


def config(*, learned=True, capacity=4):
    return CategoricalMemoryConfig(
        dimension=16,
        num_classes=3,
        capacity=capacity,
        top_k=2,
        key_nnz=4,
        key_scale=6.0,
        prior_mass=1.0,
        prior_mode="uniform",
        update=CategoricalUpdateConfig(
            learning_rate_key=0.5,
            train_keys=learned,
        ),
    )


def test_decoder_matches_manual_mixture_and_handles_empty():
    decoder = CategoricalMixtureDecoder(prior_mass=1.0)
    prior = torch.tensor([0.2, 0.3, 0.5])
    energies = torch.tensor([0.0, 1.0, 1.0])
    labels = torch.tensor([0, 1, 1], dtype=torch.int16)
    result = decoder.decode(energies, labels, prior)
    expected = torch.tensor([(0.2 + 1.0) / 3.0, (0.3 + 1.0) / 3.0, 0.5 / 3.0])
    assert torch.allclose(result.probabilities, expected, atol=1e-7)
    assert torch.isclose(
        result.background_responsibility + result.responsibilities.sum(),
        torch.tensor(1.0),
    )
    empty = decoder.decode(torch.empty(0), torch.empty(0, dtype=torch.int16), prior)
    assert torch.allclose(empty.probabilities, prior)


def test_decoder_energy_gradient_attracts_target_and_repels_wrong_label():
    decoder = CategoricalMixtureDecoder()
    energies = torch.tensor([1.0, 1.0], requires_grad=True)
    result = decoder.decode(
        energies,
        torch.tensor([1, 0], dtype=torch.int16),
        torch.tensor([0.5, 0.5]),
    )
    loss = -result.probabilities[1].log()
    gradient = torch.autograd.grad(loss, energies)[0]
    assert gradient[0] > 0
    assert gradient[1] < 0


def test_frozen_mixture_matches_same_energy_subset_knn():
    memory = CategoricalAssociativeMemory(config(learned=False))
    knn = ExactKNNBaseline(
        16,
        3,
        capacity=4,
        top_k=2,
        key_scale=6.0,
        key_nnz=4,
        prior_mass=1.0,
        prior_mode="uniform",
    )
    atoms = [
        (query([1, 3, 5], [1, -1, 1]), 0),
        (query([2, 3, 7], [1, -1, -1]), 1),
        (query([1, 4], [1, 1]), 2),
    ]
    for example_id, (atom_query, target) in enumerate(atoms):
        read = memory.read(atom_query)
        memory.observe(
            atom_query,
            target,
            pre_read=read,
            origin_id=example_id,
            insertion_mode="force",
        )
        prediction = knn.predict(atom_query)
        knn.observe(atom_query, target, prediction)
    test_query = query([1, 3, 7], [1, -1, -1])
    difference = (
        memory.read(test_query).probabilities - knn.predict(test_query).probabilities
    ).abs()
    assert float(difference.max()) < 1e-6


def test_key_only_learning_preserves_labels_and_support():
    memory = CategoricalAssociativeMemory(config(learned=True))
    first = query([1, 4, 9], [1, -1, 1])
    second = query([2, 4, 10], [-1, -1, 1])
    memory.observe(first, 0, insertion_mode="force", origin_id=10)
    memory.observe(second, 1, insertion_mode="force", origin_id=11)
    old_keys = memory.W.clone()
    old_support = memory.W != 0
    old_labels = memory.birth_label.clone()
    changed = query([1, 4, 10], [-1, -1, 1])
    report = memory.observe(changed, 0, insertion_mode="skip")
    assert report.learn.loss_after_bits <= report.learn.loss_before_bits + 1e-7
    assert torch.equal(memory.birth_label, old_labels)
    assert torch.equal(memory.W != 0, old_support)
    assert not torch.equal(memory.W, old_keys)
    assert not hasattr(memory, "V")


def test_frozen_keys_do_not_change_and_auto_insertion_is_rejected():
    memory = CategoricalAssociativeMemory(config(learned=False))
    atom = query([1, 4, 9], [1, -1, 1])
    memory.observe(atom, 0, insertion_mode="force")
    old = memory.W.clone()
    memory.observe(query([1, 4], [-1, -1]), 1, insertion_mode="skip")
    assert torch.equal(memory.W, old)
    with pytest.raises(ValueError, match="only force or skip"):
        memory.observe(atom, 0, insertion_mode="auto")  # type: ignore[arg-type]


def test_observe_rolls_back_categorical_state(monkeypatch):
    memory = CategoricalAssociativeMemory(config(learned=True))
    atom = query([1, 4, 9], [1, -1, 1])
    memory.observe(atom, 0, insertion_mode="force")
    before = memory.clone()

    def fail(*args, **kwargs):
        raise RuntimeError("injected")

    monkeypatch.setattr(memory, "_insert", fail)
    with pytest.raises(RuntimeError, match="injected"):
        memory.observe(query([2, 5], [1, -1]), 1, insertion_mode="force")
    for name in (
        "W",
        "birth_label",
        "usage",
        "origin_id",
        "atom_uid",
        "base_counts",
    ):
        assert torch.equal(getattr(memory, name), getattr(before, name))
    assert (memory.size, memory.step, memory.revision, memory.next_atom_uid) == (
        before.size,
        before.step,
        before.revision,
        before.next_atom_uid,
    )


def test_full_memory_eviction_rolls_back_every_possible_key(monkeypatch):
    memory = CategoricalAssociativeMemory(config(learned=True, capacity=2))
    memory.observe(
        query([1, 4, 9], [1, -1, 1]),
        0,
        insertion_mode="force",
    )
    memory.observe(
        query([2, 5, 10], [1, -1, 1]),
        1,
        insertion_mode="force",
    )
    before = memory.clone()
    original_insert = memory._insert

    def insert_then_fail(*args, **kwargs):
        original_insert(*args, **kwargs)
        raise RuntimeError("after eviction")

    monkeypatch.setattr(memory, "_insert", insert_then_fail)
    with pytest.raises(RuntimeError, match="after eviction"):
        memory.observe(
            query([3, 6, 11], [1, -1, 1]),
            2,
            insertion_mode="force",
        )
    for name in (
        "W",
        "birth_label",
        "usage",
        "origin_id",
        "atom_uid",
        "base_counts",
    ):
        assert torch.equal(getattr(memory, name), getattr(before, name))
    assert (memory.size, memory.step, memory.revision, memory.next_atom_uid) == (
        before.size,
        before.step,
        before.revision,
        before.next_atom_uid,
    )


def test_class_space_expands_without_dense_decoding(monkeypatch):
    memory = CategoricalAssociativeMemory(config())
    memory.expand_classes(400_000, reserve=524_288)
    assert memory.config.num_classes == 400_000
    assert memory.birth_label.dtype == torch.int64
    assert memory.base_counts.numel() == 524_288

    atom = query([1, 4, 9], [1, -1, 1])
    memory.observe_compact(atom, 399_999, insertion_mode="force")

    def fail(*args, **kwargs):
        raise AssertionError("dense decoder must not be used")

    monkeypatch.setattr(memory.decoder, "decode", fail)
    scores = memory.log_probabilities_for(atom, [399_999, 1])
    assert scores.shape == (2,)
    assert scores[0] > scores[1]
    assert memory.predict_class(atom) == 399_999


@pytest.mark.parametrize("prior_mode", ["uniform", "empirical"])
def test_compact_scores_and_updates_match_dense_path(prior_mode):
    dense_config = replace(config(), prior_mode=prior_mode)
    dense = CategoricalAssociativeMemory(dense_config)
    compact = CategoricalAssociativeMemory(dense_config)
    stream = [
        (query([1, 4, 9], [1, -1, 1]), 0, "force"),
        (query([2, 5, 10], [1, -1, 1]), 1, "force"),
        (query([1, 5, 10], [1, -1, 1]), 0, "skip"),
    ]
    for example_id, (item, target, insertion_mode) in enumerate(stream):
        dense.observe(
            item,
            target,
            origin_id=example_id,
            insertion_mode=insertion_mode,
        )
        compact.observe_compact(
            item,
            target,
            origin_id=example_id,
            insertion_mode=insertion_mode,
        )
        for name in ("W", "birth_label", "usage", "origin_id", "atom_uid"):
            assert torch.allclose(getattr(dense, name), getattr(compact, name))
        assert torch.equal(dense.base_counts, compact.base_counts)
        probe = query([1, 4, 10], [1, -1, 1])
        dense_log = dense.read(probe).probabilities.to(torch.float64).log()
        compact_log = compact.log_probabilities_for(probe, [0, 1, 2])
        assert torch.allclose(dense_log, compact_log, atol=1e-6)
