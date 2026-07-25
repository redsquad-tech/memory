import json

import pytest
import torch

from assocmem.checkpoint import load_checkpoint, save_checkpoint
from assocmem.config import InsertionConfig, MemoryConfig
from assocmem.encoding import TernaryQuery
from assocmem.memory import AssociativeMemory


def test_checkpoint_roundtrip(tmp_path):
    config = MemoryConfig(
        dimension=8,
        num_classes=2,
        capacity=3,
        key_nnz=2,
        key_scale=4,
        insertion=InsertionConfig(
            minimum_surprise_bits=0.1, surprise_margin_from_uniform_bits=2, min_gain_bits=0
        ),
    )
    memory = AssociativeMemory(config)
    query = TernaryQuery(8, torch.tensor([0, 3]), torch.tensor([1.0, -1.0], dtype=torch.float32))
    memory.observe(query, 1)
    path = tmp_path / "checkpoint"
    save_checkpoint(memory, path)
    restored = load_checkpoint(path)
    assert restored.step == memory.step
    assert restored.revision == memory.revision
    assert torch.equal(restored.read(query).probabilities, memory.read(query).probabilities)


def test_legacy_checkpoint_requires_explicit_opt_in(tmp_path):
    memory = AssociativeMemory(MemoryConfig(dimension=8, num_classes=2, capacity=2, key_nnz=2))
    path = tmp_path / "checkpoint"
    save_checkpoint(memory, path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="allow_legacy"):
        load_checkpoint(path)
    assert load_checkpoint(path, allow_legacy=True).size == 0


def test_checkpoint_rejects_wrong_dtype(tmp_path):
    memory = AssociativeMemory(MemoryConfig(dimension=8, num_classes=2, capacity=2, key_nnz=2))
    path = tmp_path / "checkpoint"
    save_checkpoint(memory, path)
    tensors = torch.load(path / "tensors.pt", weights_only=True)
    tensors["base_counts"] = tensors["base_counts"].to(torch.float32)
    torch.save(tensors, path / "tensors.pt")
    with pytest.raises(ValueError, match="dtype"):
        load_checkpoint(path)
