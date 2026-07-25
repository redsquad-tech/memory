import torch

from assocmem.config import TextEncoderConfig
from assocmem.encoding import SignedHashTextEncoder


def test_encoder_is_ternary_bounded_and_deterministic():
    config = TextEncoderConfig(dimension=128, max_features=24, hash_seed=17)
    first = SignedHashTextEncoder(config).encode("Transfer pending — AGAIN!")
    second = SignedHashTextEncoder(config).encode("Transfer pending — AGAIN!")
    assert torch.equal(first.indices, second.indices)
    assert torch.equal(first.values, second.values)
    assert first.nnz <= 24
    assert torch.all((first.values == -1) | (first.values == 1))


def test_task_namespace_changes_features():
    encoder = SignedHashTextEncoder(TextEncoderConfig(dimension=256))
    assert not torch.equal(
        encoder.encode("same text", task_id="a").to_dense(),
        encoder.encode("same text", task_id="b").to_dense(),
    )
