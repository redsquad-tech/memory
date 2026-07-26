from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

CSV_COLUMNS = [
    "id",
    "dataset",
    "config",
    "split",
    "task",
    "variant",
    "input",
    "expected_output",
    "expected_probability",
    "answer_candidates_json",
    "metadata_json",
]


@dataclass(frozen=True, slots=True)
class TaskRow:
    id: str
    dataset: str
    config: str
    split: str
    task: str
    variant: str
    input: str
    expected_output: str
    expected_probability: float = 1.0
    answer_candidates: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_csv_dict(self) -> dict[str, str | float]:
        data = asdict(self)
        data.pop("answer_candidates")
        data.pop("metadata")
        data["answer_candidates_json"] = (
            json.dumps(self.answer_candidates, ensure_ascii=False, separators=(",", ":"))
            if self.answer_candidates is not None
            else ""
        )
        data["metadata_json"] = json.dumps(
            self.metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        )
        return data
