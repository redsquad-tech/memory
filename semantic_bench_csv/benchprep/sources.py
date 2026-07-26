from __future__ import annotations

import csv
import io
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .utils import iter_json_records, iter_parquet_rows

SourceKind = Literal["parquet", "json", "tsv", "zip_tsv", "babi_txt"]

BABI_FAMILIES = [
    "en",
    "en-10k",
    "en-valid",
    "en-valid-10k",
    "hn",
    "hn-10k",
    "shuffled",
    "shuffled-10k",
]

CLUTRR_CONFIGS = [
    "gen_train23_test2to10",
    "gen_train234_test2to10",
    "rob_train_clean_23_test_all_23",
    "rob_train_disc_23_test_all_23",
    "rob_train_irr_23_test_all_23",
    "rob_train_sup_23_test_all_23",
]

RECOGS_STANDARD_CONFIGS = [
    "cogs",
    "cogs_participle_verb",
    "cogs_participle_verb_easy",
    "cogs_preposing",
    "cogs_preposing+sprinkles",
    "recogs_positional_index",
    "recogs_v1",
    "recogs_v2",
    "variable_free",
]

RECOGS_CONCAT_SIZES = ["256", "512", "1024", "2048", "3072"]
RECOGS_TOKEN_FORMS = ["remove_x_", "remove_x_()", "remove_x_(,)"]
SPLIT_FILES = [
    ("train", "train"),
    ("validation", "dev"),
    ("test", "test"),
    ("generalization", "gen"),
]


@dataclass(frozen=True, slots=True)
class SourceSpec:
    dataset: str
    config: str
    split: str
    kind: SourceKind
    path: Path
    source_key: str
    member: str | None = None


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required source file is missing: {path}")
    return path


def _spec(
    dataset: str,
    root: Path,
    relative_path: str,
    *,
    config: str,
    split: str,
    kind: SourceKind,
    member: str | None = None,
) -> SourceSpec:
    path = _require_file(root / relative_path)
    source_key = relative_path if member is None else f"{relative_path}!{member}"
    return SourceSpec(
        dataset=dataset,
        config=config,
        split=split,
        kind=kind,
        path=path,
        source_key=source_key,
        member=member,
    )


def mrcr_sources(root: Path, needles: list[str]) -> list[SourceSpec]:
    sources: list[SourceSpec] = []
    for config in needles:
        for shard in (0, 1):
            relative = f"{config}/{config}_{shard}.parquet"
            sources.append(
                _spec("mrcr", root, relative, config=config, split="test", kind="parquet")
            )
    return sources


def babi_sources(root: Path) -> list[SourceSpec]:
    base = root / "tasks_1-20_v1-2"
    sources: list[SourceSpec] = []
    for family in BABI_FAMILIES:
        for task_no in range(1, 21):
            split_names = [("train", "train")]
            if "valid" in family:
                split_names.append(("validation", "valid"))
            split_names.append(("test", "test"))
            for split, source_split in split_names:
                directory = base / family
                if "valid" in family:
                    candidates = [directory / f"qa{task_no}_{source_split}.txt"]
                else:
                    candidates = sorted(directory.glob(f"qa{task_no}_*_{source_split}.txt"))
                existing = [path for path in candidates if path.is_file()]
                if len(existing) != 1:
                    raise FileNotFoundError(
                        f"Expected one bAbI source for {family}/qa{task_no}/{split}, "
                        f"found {len(existing)}"
                    )
                path = existing[0]
                relative = path.relative_to(root).as_posix()
                sources.append(
                    SourceSpec(
                        dataset="babi",
                        config=f"{family}-qa{task_no}",
                        split=split,
                        kind="babi_txt",
                        path=path,
                        source_key=relative,
                    )
                )
    return sources


def babilong_sources(root: Path, lengths: list[str]) -> list[SourceSpec]:
    sources: list[SourceSpec] = []
    for length in lengths:
        matches: list[tuple[int, Path]] = []
        for path in (root / "data").glob(f"qa*/{length}.json"):
            match = re.fullmatch(r"qa(\d+)", path.parent.name)
            if match and path.is_file():
                matches.append((int(match.group(1)), path))
        if not matches:
            raise FileNotFoundError(f"No BABILong sources found for length {length!r}")
        for _, path in sorted(matches):
            relative = path.relative_to(root).as_posix()
            sources.append(
                SourceSpec(
                    dataset="babilong",
                    config=length,
                    split=path.parent.name,
                    kind="json",
                    path=path,
                    source_key=relative,
                )
            )
    return sources


def clutrr_sources(root: Path) -> list[SourceSpec]:
    sources: list[SourceSpec] = []
    for config in CLUTRR_CONFIGS:
        for split in ("train", "validation", "test"):
            relative = f"{config}/{split}/0000.parquet"
            sources.append(
                _spec("clutrr", root, relative, config=config, split=split, kind="parquet")
            )
    return sources


def proofwriter_sources(root: Path) -> list[SourceSpec]:
    files = [
        ("train", "data/train-00000-of-00002-6176cf3d78f48858.parquet"),
        ("train", "data/train-00001-of-00002-901eaba2151bde61.parquet"),
        ("validation", "data/validation-00000-of-00001-8f79b25dd5b0f2c3.parquet"),
        ("test", "data/test-00000-of-00001-3e27b013c60e12d8.parquet"),
    ]
    return [
        _spec(
            "proofwriter",
            root,
            relative,
            config="row_config",
            split=split,
            kind="parquet",
        )
        for split, relative in files
    ]


def _recogs_split_source(
    root: Path,
    *,
    config: str,
    directory: str,
    split: str,
    stem: str,
) -> SourceSpec:
    relative = f"{directory}/{stem}.tsv"
    return _spec("recogs", root, relative, config=config, split=split, kind="tsv")


def recogs_sources(root: Path) -> list[SourceSpec]:
    sources: list[SourceSpec] = []
    for directory in RECOGS_STANDARD_CONFIGS:
        for split, stem in SPLIT_FILES:
            sources.append(
                _recogs_split_source(
                    root, config=directory, directory=directory, split=split, stem=stem
                )
            )
    for size in RECOGS_CONCAT_SIZES:
        config = f"cogs_concat/k_{size}"
        for split, stem in SPLIT_FILES:
            sources.append(
                _recogs_split_source(
                    root,
                    config=config,
                    directory="cogs_concat",
                    split=split,
                    stem=f"{stem}_k_{size}",
                )
            )
    for form in RECOGS_TOKEN_FORMS:
        config = f"cogs_token_removal/{form}"
        for split, stem in SPLIT_FILES:
            sources.append(
                _recogs_split_source(
                    root,
                    config=config,
                    directory="cogs_token_removal",
                    split=split,
                    stem=f"{stem}_{form}",
                )
            )
    return sources


def slog_sources(root: Path) -> list[SourceSpec]:
    sources: list[SourceSpec] = []
    archive_relative = "data/generalization_sets.zip"
    archive = _require_file(root / archive_relative)
    members = {
        "cogs_LF": "generalization_sets/gen_cogsLF.tsv",
        "varfree_LF": "generalization_sets/gen_varfreeLF.tsv",
    }
    for config in ("cogs_LF", "varfree_LF"):
        for split, stem in SPLIT_FILES[:3]:
            relative = f"data/{config}/{stem}.tsv"
            sources.append(
                _spec("slog", root, relative, config=config, split=split, kind="tsv")
            )
        member = members[config]
        sources.append(
            SourceSpec(
                dataset="slog",
                config=config,
                split="generalization",
                kind="zip_tsv",
                path=archive,
                source_key=f"{archive_relative}!{member}",
                member=member,
            )
        )
    return sources


def discover_sources(
    dataset: str,
    root: Path,
    *,
    mrcr_needles: list[str],
    babilong_lengths: list[str],
) -> list[SourceSpec]:
    if not root.is_dir():
        raise FileNotFoundError(f"Raw dataset directory is missing: {root}")
    if dataset == "mrcr":
        return mrcr_sources(root, mrcr_needles)
    if dataset == "babi":
        return babi_sources(root)
    if dataset == "babilong":
        return babilong_sources(root, babilong_lengths)
    if dataset == "clutrr":
        return clutrr_sources(root)
    if dataset == "proofwriter":
        return proofwriter_sources(root)
    if dataset == "recogs":
        return recogs_sources(root)
    if dataset == "slog":
        return slog_sources(root)
    raise ValueError(f"Unsupported dataset: {dataset}")


def iter_babi_stories(path: Path) -> Iterator[dict[str, Any]]:
    story: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            line_no, body = line.split(" ", 1)
            if line_no == "1" and story:
                yield {"story": story}
                story = []
            fields = body.split("\t")
            if len(fields) > 1:
                story.append(
                    {
                        "id": line_no,
                        "type": 1,
                        "text": fields[0].strip(),
                        "answer": fields[1].strip(),
                        "supporting_ids": fields[-1].strip().split(),
                    }
                )
            else:
                story.append(
                    {
                        "id": line_no,
                        "type": 0,
                        "text": fields[0].strip(),
                        "answer": "",
                        "supporting_ids": [],
                    }
                )
    if story:
        yield {"story": story}


def iter_structured_records(source: SourceSpec) -> Iterator[dict[str, Any]]:
    if source.kind == "parquet":
        yield from iter_parquet_rows(source.path)
        return
    if source.kind == "json":
        yield from iter_json_records(source.path)
        return
    if source.kind == "babi_txt":
        yield from iter_babi_stories(source.path)
        return
    raise ValueError(f"Source {source.source_key} is not structured: {source.kind}")


def _iter_tsv_stream(stream: io.TextIOBase) -> Iterator[list[str]]:
    reader = csv.reader(stream, delimiter="\t")
    for row in reader:
        if row and any(cell.strip() for cell in row):
            yield [cell.strip() for cell in row]


def iter_semantic_fields(source: SourceSpec) -> Iterator[list[str]]:
    if source.kind == "tsv":
        with source.path.open("r", encoding="utf-8-sig", newline="") as fh:
            yield from _iter_tsv_stream(fh)
        return
    if source.kind == "zip_tsv":
        assert source.member is not None
        with zipfile.ZipFile(source.path) as archive:
            with archive.open(source.member, pwd=b"SLOG") as raw:
                with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as fh:
                    yield from _iter_tsv_stream(fh)
        return
    raise ValueError(f"Source {source.source_key} is not semantic TSV: {source.kind}")
