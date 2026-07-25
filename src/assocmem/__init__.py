"""Predictive associative memory."""

from .checkpoint import load_checkpoint, save_checkpoint
from .config import (
    EvictionConfig,
    InsertionConfig,
    MemoryConfig,
    TextEncoderConfig,
    UpdateConfig,
)
from .encoding import SignedHashTextEncoder, TernaryQuery
from .memory import AssociativeMemory, ObserveReport, ReadResult

__all__ = [
    "AssociativeMemory",
    "EvictionConfig",
    "InsertionConfig",
    "MemoryConfig",
    "ObserveReport",
    "ReadResult",
    "SignedHashTextEncoder",
    "TernaryQuery",
    "TextEncoderConfig",
    "UpdateConfig",
    "load_checkpoint",
    "save_checkpoint",
]

__version__ = "0.1.0"
