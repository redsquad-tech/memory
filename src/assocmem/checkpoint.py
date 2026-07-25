from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import MemoryConfig
from .memory import AssociativeMemory

FORMAT_VERSION = 2


def save_checkpoint(
    memory: AssociativeMemory,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        manifest = {
            "format_version": FORMAT_VERSION,
            "config": asdict(memory.config),
            "size": memory.size,
            "step": memory.step,
            "revision": memory.revision,
            "next_atom_uid": memory.next_atom_uid,
            "metadata": metadata or {},
        }
        tensors = {
            "W": memory.W[: memory.size].detach().cpu().clone(),
            "V": memory.V[: memory.size].detach().cpu().clone(),
            "usage": memory.usage[: memory.size].detach().cpu().clone(),
            "origin_id": memory.origin_id[: memory.size].detach().cpu().clone(),
            "atom_uid": memory.atom_uid[: memory.size].detach().cpu().clone(),
            "base_counts": memory.base_counts.detach().cpu().clone(),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        torch.save(tensors, temporary / "tensors.pt")
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    allow_legacy: bool = False,
) -> AssociativeMemory:
    source = Path(path)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    version = manifest.get("format_version")
    if version == 1 and not allow_legacy:
        raise ValueError("legacy checkpoint requires allow_legacy=True")
    if version not in ({1, FORMAT_VERSION} if allow_legacy else {FORMAT_VERSION}):
        raise ValueError("unsupported checkpoint format")
    raw = manifest["config"]
    config = MemoryConfig.from_dict(raw)
    memory = AssociativeMemory(config, device=device)
    tensors = torch.load(source / "tensors.pt", map_location=device, weights_only=True)
    size = int(manifest["size"])
    if not 0 <= size <= config.capacity:
        raise ValueError("invalid checkpoint size")
    expected = {
        "W": (size, config.dimension),
        "V": (size, config.num_classes),
        "usage": (size,),
        "origin_id": (size,),
        "atom_uid": (size,),
        "base_counts": (config.num_classes,),
    }
    expected_dtypes = {
        "W": torch.float32,
        "V": torch.float32,
        "usage": torch.float32,
        "origin_id": torch.int64,
        "atom_uid": torch.int64,
        "base_counts": torch.int64,
    }
    for name, shape in expected.items():
        if name not in tensors or tuple(tensors[name].shape) != shape:
            raise ValueError(f"invalid tensor {name}")
        if tensors[name].dtype != expected_dtypes[name]:
            raise ValueError(f"invalid dtype for tensor {name}")
        if not bool(torch.isfinite(tensors[name]).all()):
            raise ValueError(f"non-finite tensor {name}")
    for counter in ("step", "revision", "next_atom_uid"):
        if not isinstance(manifest.get(counter), int) or manifest[counter] < 0:
            raise ValueError(f"invalid checkpoint counter {counter}")
    if bool((tensors["usage"] < 0).any()) or bool((tensors["base_counts"] < 0).any()):
        raise ValueError("checkpoint contains negative counts")
    if size:
        uids = tensors["atom_uid"]
        if bool((uids < 0).any()) or int(torch.unique(uids).numel()) != size:
            raise ValueError("checkpoint atom uids must be unique and non-negative")
        if int(uids.max()) >= int(manifest["next_atom_uid"]):
            raise ValueError("next_atom_uid does not exceed active atom uids")
    with torch.no_grad():
        memory.W[:size] = tensors["W"]
        memory.V[:size] = tensors["V"]
        memory.usage[:size] = tensors["usage"]
        memory.origin_id[:size] = tensors["origin_id"]
        memory.atom_uid[:size] = tensors["atom_uid"]
        memory.base_counts[:] = tensors["base_counts"]
    memory.size = size
    memory.step = int(manifest["step"])
    memory.revision = int(manifest["revision"])
    memory.next_atom_uid = int(manifest["next_atom_uid"])
    key_norms = memory.W[:size].abs().sum(dim=1)
    if size and not torch.allclose(
        key_norms,
        torch.full_like(key_norms, config.key_scale),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("checkpoint violates key norm invariant")
    if size and bool((memory.W[:size].count_nonzero(dim=1) > config.key_nnz).any()):
        raise ValueError("checkpoint violates key sparsity invariant")
    value_means = memory.V[:size].mean(dim=1) if size else torch.empty(0)
    if size and not torch.allclose(value_means, torch.zeros_like(value_means), atol=1e-5):
        raise ValueError("checkpoint violates centered value invariant")
    if size and bool((memory.V[:size].norm(dim=1) > config.update.value_max_norm + 1e-5).any()):
        raise ValueError("checkpoint violates value norm invariant")
    return memory
