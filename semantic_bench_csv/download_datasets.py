#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import snapshot_download

from benchprep.sources import BABI_FAMILIES, discover_sources

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "datasets_manifest.json"
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_DATASETS = ["mrcr", "babi", "babilong", "clutrr", "proofwriter", "recogs", "slog"]


def comma_values(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate values are not allowed: {value}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pinned benchmark sources.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--datasets", default="all", help="Comma-separated names or 'all'.")
    parser.add_argument("--mrcr-needles", default="2needle,4needle,8needle")
    parser.add_argument("--babilong-lengths", default="0k,1k,2k,4k,8k,16k")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--list", action="store_true", help="List available datasets and exit.")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("Manifest root must be an object")
    missing = set(DEFAULT_DATASETS) - set(payload)
    if missing:
        raise ValueError(f"Manifest is missing datasets: {', '.join(sorted(missing))}")
    return payload


def selected_names(value: str, manifest: dict[str, Any]) -> list[str]:
    if value.strip().lower() == "all":
        return list(manifest)
    names = comma_values(value)
    unknown = sorted(set(names) - set(manifest))
    if unknown:
        raise ValueError(f"Unknown datasets: {', '.join(unknown)}")
    return names


def allow_patterns(name: str, mrcr_needles: list[str], babilong_lengths: list[str]) -> list[str] | None:
    if name == "mrcr":
        return ["README.md", *[f"{config}/**" for config in mrcr_needles]]
    if name == "babilong":
        return ["README.md", *[f"data/qa*/{length}.json" for length in babilong_lengths]]
    return None


def staging_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.partial-{uuid.uuid4().hex[:8]}")


def write_source_metadata(staging: Path, payload: dict[str, Any]) -> None:
    (staging / ".source.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def commit_staging(staging: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    staging.replace(destination)


def download_hf(
    name: str,
    spec: dict[str, Any],
    destination: Path,
    *,
    mrcr_needles: list[str],
    babilong_lengths: list[str],
) -> None:
    staging = staging_path(destination)
    try:
        revision = spec["revision"]
        snapshot_download(
            repo_id=spec["repo_id"],
            repo_type="dataset",
            revision=revision,
            local_dir=staging,
            allow_patterns=allow_patterns(name, mrcr_needles, babilong_lengths),
        )
        write_source_metadata(
            staging,
            {
                "kind": "huggingface",
                "repo_id": spec["repo_id"],
                "resolved_revision": revision,
                "mrcr_needles": mrcr_needles if name == "mrcr" else None,
                "babilong_lengths": babilong_lengths if name == "babilong" else None,
            },
        )
        commit_staging(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def download_git(spec: dict[str, Any], destination: Path) -> None:
    staging = staging_path(destination)
    try:
        staging.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(staging)], check=True)
        subprocess.run(
            ["git", "-C", str(staging), "remote", "add", "origin", spec["url"]],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(staging), "fetch", "-q", "--depth", "1", "origin", spec["commit"]],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(staging), "checkout", "-q", "--detach", "FETCH_HEAD"],
            check=True,
        )
        write_source_metadata(
            staging,
            {"kind": "git", "url": spec["url"], "commit": spec["commit"]},
        )
        commit_staging(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _download_file(url: str, destination: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    request = urllib.request.Request(url, headers={"User-Agent": "semantic-bench-csv/0.1"})
    with urllib.request.urlopen(request) as response:
        with destination.open("wb") as fh:
            while chunk := response.read(1024 * 1024):
                fh.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    return size, digest.hexdigest()


def _is_babi_member(name: str) -> bool:
    path = PurePosixPath(name)
    if len(path.parts) != 3 or path.parts[0] != "tasks_1-20_v1-2":
        return False
    if path.parts[1] not in BABI_FAMILIES:
        return False
    filename = path.parts[2]
    return filename.startswith("qa") and filename.endswith(".txt")


def download_babi_archive(spec: dict[str, Any], destination: Path) -> None:
    staging = staging_path(destination)
    archive_path = staging.with_name(f"{staging.name}.tar.gz")
    try:
        staging.mkdir(parents=True)
        size, digest = _download_file(spec["url"], archive_path)
        if size != spec["size"]:
            raise ValueError(f"bAbI archive size differs: expected {spec['size']}, got {size}")
        if digest != spec["sha256"]:
            raise ValueError(
                f"bAbI archive checksum differs: expected {spec['sha256']}, got {digest}"
            )
        extracted = 0
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                if not member.isfile() or not _is_babi_member(member.name):
                    continue
                relative = PurePosixPath(member.name)
                target = staging.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Cannot read bAbI archive member: {member.name}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted += 1
        if extracted != 360:
            raise ValueError(f"Expected 360 canonical bAbI split files, extracted {extracted}")
        write_source_metadata(
            staging,
            {
                "kind": "archive",
                "url": spec["url"],
                "size": size,
                "sha256": digest,
                "families": BABI_FAMILIES,
            },
        )
        commit_staging(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        archive_path.unlink(missing_ok=True)


def metadata_matches(destination: Path, spec: dict[str, Any]) -> bool:
    source_file = destination / ".source.json"
    if not destination.is_dir() or not source_file.is_file():
        return False
    try:
        payload = json.loads(source_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if spec["kind"] == "huggingface":
        return (
            payload.get("repo_id") == spec["repo_id"]
            and payload.get("resolved_revision") == spec["revision"]
        )
    if spec["kind"] == "git":
        return payload.get("url") == spec["url"] and payload.get("commit") == spec["commit"]
    if spec["kind"] == "archive":
        return (
            payload.get("url") == spec["url"]
            and payload.get("size") == spec["size"]
            and payload.get("sha256") == spec["sha256"]
        )
    return False


def sources_are_complete(
    name: str,
    destination: Path,
    *,
    mrcr_needles: list[str],
    babilong_lengths: list[str],
) -> bool:
    try:
        sources = discover_sources(
            name,
            destination,
            mrcr_needles=mrcr_needles,
            babilong_lengths=babilong_lengths,
        )
    except (FileNotFoundError, ValueError, zipfile.BadZipFile):
        return False
    expected_counts = {
        "babi": 360,
        "clutrr": 18,
        "proofwriter": 4,
        "recogs": 68,
        "slog": 8,
    }
    if name == "mrcr":
        return len(sources) == 2 * len(mrcr_needles)
    if name == "babilong":
        return bool(sources)
    return len(sources) == expected_counts[name]


def ensure_datasets(
    *,
    manifest: dict[str, dict[str, Any]],
    names: list[str],
    raw_dir: Path,
    mrcr_needles: list[str],
    babilong_lengths: list[str],
    force_download: bool = False,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        spec = manifest[name]
        destination = raw_dir / name
        reusable = (
            not force_download
            and metadata_matches(destination, spec)
            and sources_are_complete(
                name,
                destination,
                mrcr_needles=mrcr_needles,
                babilong_lengths=babilong_lengths,
            )
        )
        if reusable:
            print(f"[{name}] using cached pinned source")
            continue
        print(f"[{name}] downloading pinned source")
        if spec["kind"] == "huggingface":
            download_hf(
                name,
                spec,
                destination,
                mrcr_needles=mrcr_needles,
                babilong_lengths=babilong_lengths,
            )
        elif spec["kind"] == "git":
            download_git(spec, destination)
        elif spec["kind"] == "archive":
            download_babi_archive(spec, destination)
        else:
            raise ValueError(f"Unsupported source kind for {name}: {spec['kind']}")


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    if args.list:
        for name, spec in manifest.items():
            print(f"{name:12} {spec.get('description', '')}")
        return 0
    names = selected_names(args.datasets, manifest)
    ensure_datasets(
        manifest=manifest,
        names=names,
        raw_dir=args.output,
        mrcr_needles=comma_values(args.mrcr_needles),
        babilong_lengths=comma_values(args.babilong_lengths),
        force_download=args.force_download,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
