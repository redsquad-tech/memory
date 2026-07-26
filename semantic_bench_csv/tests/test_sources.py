from __future__ import annotations

import zipfile
from pathlib import Path

from benchprep.sources import (
    BABI_FAMILIES,
    CLUTRR_CONFIGS,
    RECOGS_CONCAT_SIZES,
    RECOGS_STANDARD_CONFIGS,
    RECOGS_TOKEN_FORMS,
    SPLIT_FILES,
    babi_sources,
    babilong_sources,
    clutrr_sources,
    iter_babi_stories,
    recogs_sources,
    slog_sources,
)


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_babilong_discovery_ignores_cache_json(tmp_path: Path) -> None:
    source = tmp_path / "data/qa1/0k.json"
    touch(source)
    touch(tmp_path / ".cache/huggingface/tree.json")
    touch(tmp_path / ".source.json")
    sources = babilong_sources(tmp_path, ["0k"])
    assert [item.source_key for item in sources] == ["data/qa1/0k.json"]


def test_babi_manifest_has_all_160_configs_and_exact_splits(tmp_path: Path) -> None:
    for family in BABI_FAMILIES:
        for task_no in range(1, 21):
            if "valid" in family:
                for split in ("train", "valid", "test"):
                    touch(tmp_path / f"tasks_1-20_v1-2/{family}/qa{task_no}_{split}.txt")
            else:
                for split in ("train", "test"):
                    touch(
                        tmp_path
                        / f"tasks_1-20_v1-2/{family}/qa{task_no}_description_{split}.txt"
                    )
    sources = babi_sources(tmp_path)
    assert len(sources) == 360
    assert len({source.config for source in sources}) == 160
    train = next(source for source in sources if source.config == "en-valid-qa1")
    assert train.split == "train"


def test_babi_text_parser_preserves_questions_and_supports(tmp_path: Path) -> None:
    path = tmp_path / "qa1_train.txt"
    path.write_text(
        "1 Mary went to the kitchen.\n"
        "2 Where is Mary?\tkitchen\t1\n"
        "1 John went to the office.\n"
        "2 Where is John?\toffice\t1\n",
        encoding="utf-8",
    )
    stories = list(iter_babi_stories(path))
    assert len(stories) == 2
    assert stories[0]["story"][1]["answer"] == "kitchen"
    assert stories[0]["story"][1]["supporting_ids"] == ["1"]


def test_clutrr_split_comes_from_partition_not_config_name(tmp_path: Path) -> None:
    for config in CLUTRR_CONFIGS:
        for split in ("train", "validation", "test"):
            touch(tmp_path / config / split / "0000.parquet")
    sources = clutrr_sources(tmp_path)
    source = next(
        item
        for item in sources
        if item.config == "gen_train23_test2to10" and item.split == "train"
    )
    assert source.source_key == "gen_train23_test2to10/train/0000.parquet"


def test_recogs_whitelist_contains_all_official_variants(tmp_path: Path) -> None:
    for config in RECOGS_STANDARD_CONFIGS:
        for _, stem in SPLIT_FILES:
            touch(tmp_path / config / f"{stem}.tsv")
    for size in RECOGS_CONCAT_SIZES:
        for _, stem in SPLIT_FILES:
            touch(tmp_path / "cogs_concat" / f"{stem}_k_{size}.tsv")
    for form in RECOGS_TOKEN_FORMS:
        for _, stem in SPLIT_FILES:
            touch(tmp_path / "cogs_token_removal" / f"{stem}_{form}.tsv")
    touch(tmp_path / "model/not_a_dataset.tsv")
    sources = recogs_sources(tmp_path)
    assert len(sources) == 68
    assert len({source.config for source in sources}) == 17
    assert all("model/" not in source.source_key for source in sources)


def test_slog_reads_only_two_official_representations(tmp_path: Path) -> None:
    for config in ("cogs_LF", "varfree_LF"):
        for stem in ("train", "dev", "test"):
            touch(tmp_path / f"data/{config}/{stem}.tsv")
    archive_path = tmp_path / "data/generalization_sets.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("generalization_sets/gen_cogsLF.tsv", "x\ty\tz\n")
        archive.writestr("generalization_sets/gen_varfreeLF.tsv", "x\ty\tz\n")
    touch(tmp_path / "generation_scripts/duplicate.tsv")
    sources = slog_sources(tmp_path)
    assert len(sources) == 8
    assert {source.config for source in sources} == {"cogs_LF", "varfree_LF"}
    assert sum(source.kind == "zip_tsv" for source in sources) == 2
