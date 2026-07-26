#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from assocmem.bench_protocol import run_infer, run_learn


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="operation", required=True)
    learn = commands.add_parser("learn")
    learn.add_argument("--model-in", type=Path, required=True)
    learn.add_argument("--examples", type=Path, required=True)
    learn.add_argument("--model-out", type=Path, required=True)
    learn.add_argument("--budget", type=Path, required=True)
    learn.add_argument("--seed", type=int, required=True)
    infer = commands.add_parser("infer")
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--requests", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--budget", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.operation == "learn":
            run_learn(
                args.model_in,
                args.examples,
                args.model_out,
                seed=args.seed,
            )
        else:
            run_infer(args.model, args.requests, args.output)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

