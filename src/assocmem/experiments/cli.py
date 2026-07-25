from __future__ import annotations

import argparse
import json
import random
import tomllib
from dataclasses import asdict
from pathlib import Path

import torch

from ..baselines import (
    CentroidBaseline,
    ExactKNNBaseline,
    FrequencyBaseline,
    OnlineLinearBaseline,
    OnlineMLPBaseline,
)
from ..checkpoint import load_checkpoint, save_checkpoint
from ..config import (
    EvictionConfig,
    InsertionConfig,
    MemoryConfig,
    TextEncoderConfig,
    UpdateConfig,
)
from ..encoding import SignedHashTextEncoder
from ..memory import AssociativeMemory
from ..online import MemoryOnlineModel, OnlineExample, evaluate_online
from .datasets import (
    iter_banking77,
    load_clinc150,
    prepare_banking77,
    prepare_clinc150,
    prepare_enwik8,
)
from .synthetic import PredictiveSemanticStream, SyntheticConfig


def _config(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _synthetic_run(args: argparse.Namespace) -> None:
    raw = _config(args.config)
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    encoder_cfg = TextEncoderConfig(**{**raw.get("encoder", {}), "hash_seed": seed})
    encoder = SignedHashTextEncoder(encoder_cfg)
    synthetic_cfg = SyntheticConfig(**raw.get("synthetic", {}), seed=seed)
    model_name = raw.get("run", {}).get("model", "memory")
    classes = synthetic_cfg.num_outputs
    if model_name == "memory":
        update_raw = raw.get("update", {})
        memory_raw = dict(raw.get("memory", {}))
        memory = AssociativeMemory(
            MemoryConfig(
                dimension=encoder_cfg.dimension,
                num_classes=classes,
                capacity=memory_raw.pop("capacity", 256),
                update=UpdateConfig(**update_raw),
                insertion=InsertionConfig(**raw.get("insertion", {})),
                eviction=EvictionConfig(**raw.get("eviction", {})),
                encoder_fingerprint=encoder.fingerprint,
                **memory_raw,
            ),
            device=args.device,
        )
        model = MemoryOnlineModel(memory)
    elif model_name == "frequency":
        model = FrequencyBaseline(classes)
    elif model_name == "knn":
        model = ExactKNNBaseline(encoder_cfg.dimension, classes, capacity=256)
    elif model_name == "centroid":
        model = CentroidBaseline(encoder_cfg.dimension, classes)
    elif model_name == "linear":
        model = OnlineLinearBaseline(encoder_cfg.dimension, classes)
    elif model_name == "mlp":
        model = OnlineMLPBaseline(encoder_cfg.dimension, classes)
    else:
        raise ValueError(f"unknown model: {model_name}")
    output = Path(args.output) / f"{model_name}-seed-{seed}"
    summary = evaluate_online(
        model, PredictiveSemanticStream(synthetic_cfg, encoder), output_dir=output
    )
    if model_name == "memory":
        save_checkpoint(model.memory, output / "checkpoint", metadata={"summary": summary})
    (output / "config.resolved.json").write_text(
        json.dumps(
            {"encoder": asdict(encoder_cfg), "synthetic": asdict(synthetic_cfg), "raw": raw},
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2))


def _intent_run(args: argparse.Namespace, raw: dict, task: str) -> None:
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    encoder_cfg = TextEncoderConfig(**{**raw.get("encoder", {}), "hash_seed": seed})
    encoder = SignedHashTextEncoder(encoder_cfg)
    data_dir = raw.get("run", {}).get("data_dir", "data")
    split = raw.get("run", {}).get("split", "train")
    if task == "banking77":
        records = list(iter_banking77(data_dir, split))
        num_classes = 77
    elif task == "clinc150":
        dataset = load_clinc150(data_dir)
        label_names = sorted(
            {label for key in ("train", "val", "test") for _, label in dataset.get(key, [])}
        )
        mapping = {label: index for index, label in enumerate(label_names)}
        key = {"validation": "val"}.get(split, split)
        records = [(text, mapping[label]) for text, label in dataset[key]]
        num_classes = len(label_names)
    else:
        raise ValueError(task)
    random.Random(seed).shuffle(records)
    max_examples = raw.get("run", {}).get("max_examples")
    if max_examples is not None:
        records = records[: int(max_examples)]
    examples = (
        OnlineExample(encoder.encode(text), target, example_id=index)
        for index, (text, target) in enumerate(records)
    )
    memory_raw = dict(raw.get("memory", {}))
    memory = AssociativeMemory(
        MemoryConfig(
            dimension=encoder_cfg.dimension,
            num_classes=num_classes,
            capacity=memory_raw.pop("capacity", 2048),
            update=UpdateConfig(**raw.get("update", {})),
            insertion=InsertionConfig(**raw.get("insertion", {})),
            eviction=EvictionConfig(**raw.get("eviction", {})),
            encoder_fingerprint=encoder.fingerprint,
            **memory_raw,
        ),
        device=args.device,
    )
    model = MemoryOnlineModel(memory)
    output = Path(args.output) / f"{task}-memory-seed-{seed}"
    summary = evaluate_online(model, examples, output_dir=output)
    save_checkpoint(memory, output / "checkpoint", metadata={"summary": summary, "task": task})
    (output / "config.resolved.json").write_text(
        json.dumps({"raw": raw, "seed": seed, "task": task}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def _run(args: argparse.Namespace) -> None:
    raw = _config(args.config)
    task = raw.get("run", {}).get("task", "synthetic")
    if task == "synthetic":
        _synthetic_run(args)
    else:
        _intent_run(args, raw, task)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="assocmem-exp")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--device", default="cpu")
    run.add_argument("--output", default="runs")
    prepare = sub.add_parser("prepare-data")
    prepare.add_argument("dataset", choices=("banking77", "clinc150", "enwik8"))
    prepare.add_argument("--data-dir", default="data")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("checkpoint")
    sweep = sub.add_parser("sweep")
    sweep.add_argument("--config", required=True)
    sweep.add_argument("--seeds", default="0,1,2,3,4")
    sweep.add_argument("--device", default="cpu")
    sweep.add_argument("--output", default="runs")
    args = parser.parse_args(argv)
    if args.command == "run":
        _run(args)
    elif args.command == "prepare-data":
        functions = {
            "banking77": prepare_banking77,
            "clinc150": prepare_clinc150,
            "enwik8": prepare_enwik8,
        }
        print(functions[args.dataset](args.data_dir))
    elif args.command == "inspect":
        memory = load_checkpoint(args.checkpoint)
        print(
            json.dumps(
                {
                    "size": memory.size,
                    "step": memory.step,
                    "revision": memory.revision,
                    "allocated_bytes": memory.allocated_bytes(),
                    "active_logical_bytes": memory.active_logical_bytes(),
                },
                indent=2,
            )
        )
    elif args.command == "sweep":
        for seed in (int(value) for value in args.seeds.split(",")):
            _run(
                argparse.Namespace(
                    config=args.config,
                    seed=seed,
                    device=args.device,
                    output=args.output,
                )
            )
