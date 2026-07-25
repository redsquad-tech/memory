from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path

BANKING_COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
BANKING_BASE = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    f"{BANKING_COMMIT}/banking_data"
)
BANKING_SHA256 = {
    "categories.json": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}
CLINC_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
ENWIK8_URL = "https://mattmahoney.net/dc/enwik8.zip"


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(destination)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_banking77(data_dir: str | Path) -> Path:
    destination = Path(data_dir) / "banking77"
    for filename, expected in BANKING_SHA256.items():
        path = destination / filename
        if not path.exists():
            _download(f"{BANKING_BASE}/{filename}", path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"BANKING77 checksum mismatch for {filename}: {actual} != {expected}")
    return destination


def iter_banking77(data_dir: str | Path, split: str) -> Iterator[tuple[str, int]]:
    root = prepare_banking77(data_dir)
    categories = json.loads((root / "categories.json").read_text())
    mapping = {name: index for index, name in enumerate(categories)}
    with (root / f"{split}.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            yield row["text"], mapping[row["category"]]


def prepare_clinc150(data_dir: str | Path) -> Path:
    destination = Path(data_dir) / "clinc150" / "data_full.json"
    if not destination.exists():
        _download(CLINC_URL, destination)
    return destination


def load_clinc150(data_dir: str | Path) -> dict[str, list[list[str]]]:
    return json.loads(prepare_clinc150(data_dir).read_text(encoding="utf-8"))


def prepare_enwik8(data_dir: str | Path) -> Path:
    root = Path(data_dir) / "enwik8"
    target = root / "enwik8"
    if target.exists():
        return target
    archive = root / "enwik8.zip"
    if not archive.exists():
        _download(ENWIK8_URL, archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extract("enwik8", root)
    return target
