from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from .schema import TaskRow
from .utils import (
    clean_text,
    first_present,
    jsonish,
    normalize_sequence_of_records,
    stable_id,
)

CLUTRR_LABELS = [
    "aunt", "son-in-law", "grandfather", "brother", "sister", "father", "mother",
    "grandmother", "uncle", "daughter-in-law", "grandson", "granddaughter",
    "father-in-law", "mother-in-law", "nephew", "son", "daughter", "niece",
    "husband", "wife", "sister-in-law",
]

PROOFWRITER_LABELS = ["true", "false", "unknown"]


def serialize_messages(value: Any) -> str:
    messages = jsonish(value, default=value)
    if not isinstance(messages, list):
        return clean_text(value)
    chunks: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            chunks.append(clean_text(message))
            continue
        role = clean_text(message.get("role", "message")).upper()
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                clean_text(part.get("text", part)) if isinstance(part, Mapping) else clean_text(part)
                for part in content
            )
        chunks.append(f"[{role}]\n{clean_text(content)}")
    return "\n\n".join(chunks)


def normalize_label(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and value in (0, 1):
        return "true" if int(value) == 1 else "false"
    text = clean_text(value).lower()
    mapping = {
        "1": "true", "yes": "true", "entailed": "true", "entailment": "true",
        "0": "false", "no": "false", "contradiction": "false",
        "both unknown": "unknown", "unknown": "unknown", "uncertain": "unknown",
    }
    return mapping.get(text, text)


class Adapter:
    name: str

    def convert(
        self,
        row: Mapping[str, Any],
        *,
        config: str,
        split: str,
        source_key: str,
        source_index: int,
        variants: set[str],
    ) -> Iterable[TaskRow]:
        raise NotImplementedError


class MRCRAdapter(Adapter):
    name = "mrcr"

    def convert(
        self,
        row: Mapping[str, Any],
        *,
        config: str,
        split: str,
        source_key: str,
        source_index: int,
        variants: set[str],
    ) -> Iterable[TaskRow]:
        if "full" not in variants:
            return
        answer = clean_text(first_present(row, "answer", "target", "expected_output"))
        prompt = serialize_messages(first_present(row, "prompt", "messages", "input"))
        source_id = clean_text(first_present(row, "id", "sample_id", default=source_index))
        metadata = {
            "source_id": source_id,
            "source_key": source_key,
            "random_string_to_prepend": clean_text(row.get("random_string_to_prepend")),
            "date_added": clean_text(row.get("date_added")),
            "needle_config": config,
            "n_needles": row.get("n_needles"),
            "desired_msg_index": row.get("desired_msg_index"),
            "total_messages": row.get("total_messages"),
            "n_chars": row.get("n_chars"),
        }
        yield TaskRow(
            id=stable_id(self.name, source_key, source_index, "full"),
            dataset=self.name,
            config=config,
            split=split,
            task="multi_round_coreference_retrieval",
            variant="full",
            input=prompt,
            expected_output=answer,
            metadata=metadata,
        )


class BabiAdapter(Adapter):
    name = "babi"

    def convert(
        self,
        row: Mapping[str, Any],
        *,
        config: str,
        split: str,
        source_key: str,
        source_index: int,
        variants: set[str],
    ) -> Iterable[TaskRow]:
        story = normalize_sequence_of_records(row.get("story"))
        contexts: list[tuple[str, str]] = []
        task = next((part for part in config.split("-") if re.fullmatch(r"qa\d+", part)), config)
        question_no = 0
        for event in story:
            event_type = event.get("type")
            is_question = event_type in (1, "1", "question") or bool(clean_text(event.get("answer")))
            event_id = clean_text(event.get("id"))
            text = clean_text(event.get("text"))
            if not is_question:
                contexts.append((event_id, text))
                continue
            question_no += 1
            answer = clean_text(event.get("answer"))
            support_ids = [clean_text(item) for item in (event.get("supporting_ids") or [])]
            variants_to_emit: list[str] = []
            if "full" in variants:
                variants_to_emit.append("full")
            if "oracle" in variants and support_ids:
                variants_to_emit.append("oracle")
            for variant in variants_to_emit:
                selected = contexts
                if variant == "oracle":
                    support_set = set(support_ids)
                    selected = [item for item in contexts if item[0] in support_set]
                context_text = "\n".join(text for _, text in selected)
                prompt = f"Context:\n{context_text}\n\nQuestion:\n{text}\n\nAnswer:"
                source_id = f"{source_index}:{question_no}:{event_id}"
                yield TaskRow(
                    id=stable_id(self.name, source_key, source_id, variant),
                    dataset=self.name,
                    config=config,
                    split=split,
                    task=task,
                    variant=variant,
                    input=prompt,
                    expected_output=answer,
                    metadata={
                        "source_id": source_id,
                        "source_key": source_key,
                        "supporting_ids": support_ids,
                        "context_statement_count": len(selected),
                    },
                )


class BABILongAdapter(Adapter):
    name = "babilong"

    def convert(
        self,
        row: Mapping[str, Any],
        *,
        config: str,
        split: str,
        source_key: str,
        source_index: int,
        variants: set[str],
    ) -> Iterable[TaskRow]:
        if "full" not in variants:
            return
        context = clean_text(first_present(row, "input", "context", "passage", "story"))
        question = clean_text(first_present(row, "question", "query"))
        target = clean_text(first_present(row, "target", "answer", "label"))
        task = clean_text(first_present(row, "task", default=split if split.startswith("qa") else "babi"))
        normalized_split = "test" if split.startswith("qa") or split == "unknown" else split
        prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        source_id = clean_text(first_present(row, "id", default=source_index))
        yield TaskRow(
            id=stable_id(self.name, source_key, source_index, "full"),
            dataset=self.name,
            config=config,
            split=normalized_split,
            task=task,
            variant="full",
            input=prompt,
            expected_output=target,
            metadata={
                "source_id": source_id,
                "source_key": source_key,
                "context_chars": len(context),
                "source_partition": split,
            },
        )


class CLUTRRAdapter(Adapter):
    name = "clutrr"

    @staticmethod
    def query_text(value: Any) -> str:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return f"What is the relation of {value[0]} to {value[1]}?"
        text = clean_text(value)
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, tuple) and len(parsed) >= 2:
                return f"What is the relation of {parsed[0]} to {parsed[1]}?"
        except (ValueError, SyntaxError):
            pass
        return text

    def convert(
        self,
        row: Mapping[str, Any],
        *,
        config: str,
        split: str,
        source_key: str,
        source_index: int,
        variants: set[str],
    ) -> Iterable[TaskRow]:
        story = clean_text(row.get("story"))
        clean_story = clean_text(row.get("clean_story"))
        query = self.query_text(row.get("query"))
        target = clean_text(first_present(row, "target_text", "label", "answer", "target"))
        source_id = clean_text(first_present(row, "id", default=source_index))
        variants_to_emit: list[str] = []
        if "full" in variants:
            variants_to_emit.append("full")
        if "oracle" in variants and clean_story:
            variants_to_emit.append("oracle")
        for variant in variants_to_emit:
            selected_story = clean_story if variant == "oracle" else story
            prompt = f"Story:\n{selected_story}\n\nQuestion:\n{query}\n\nAnswer:"
            yield TaskRow(
                id=stable_id(self.name, source_key, source_index, variant),
                dataset=self.name,
                config=config,
                split=split,
                task=clean_text(row.get("task_name")) or "kinship_relation",
                variant=variant,
                input=prompt,
                expected_output=target,
                answer_candidates=CLUTRR_LABELS,
                metadata={
                    "source_id": source_id,
                    "source_key": source_key,
                    "f_comb": clean_text(row.get("f_comb")),
                    "edge_types": row.get("edge_types"),
                    "query": row.get("query"),
                    "proof_state": row.get("proof_state"),
                    "task_split": clean_text(row.get("task_split")),
                },
            )


class ProofWriterAdapter(Adapter):
    name = "proofwriter"

    def convert(
        self,
        row: Mapping[str, Any],
        *,
        config: str,
        split: str,
        source_key: str,
        source_index: int,
        variants: set[str],
    ) -> Iterable[TaskRow]:
        if "full" not in variants:
            return
        row_config = clean_text(row.get("config")) or config
        theory = clean_text(first_present(row, "theory", "context", "rules"))
        question = clean_text(first_present(row, "question", "query", "hypothesis"))
        answer = normalize_label(first_present(row, "answer", "label", "target", "correct"))
        source_id = clean_text(first_present(row, "id", "theory_id", default=source_index))
        max_depth = first_present(row, "maxD", "depth", "QDep", default="")
        prompt = f"Theory:\n{theory}\n\nQuestion:\n{question}\n\nAnswer (true, false, or unknown):"
        metadata = {
            "source_id": source_id,
            "source_key": source_key,
            "max_depth": max_depth,
            "n_facts": first_present(row, "NFact", "n_facts"),
            "n_rules": first_present(row, "NRule", "n_rules"),
            "proof": first_present(row, "allProofs", "proof", "proofs"),
            "world_assumption": first_present(row, "world_assumption", "assumption"),
        }
        yield TaskRow(
            id=stable_id(self.name, source_key, source_index, "full"),
            dataset=self.name,
            config=row_config,
            split=split,
            task=f"deduction_depth_{max_depth}" if max_depth != "" else "deduction",
            variant="full",
            input=prompt,
            expected_output=answer,
            answer_candidates=PROOFWRITER_LABELS,
            metadata=metadata,
        )


ADAPTERS: dict[str, Adapter] = {
    "mrcr": MRCRAdapter(),
    "babi": BabiAdapter(),
    "babilong": BABILongAdapter(),
    "clutrr": CLUTRRAdapter(),
    "proofwriter": ProofWriterAdapter(),
}


def semantic_parsing_row(
    *, dataset: str, config: str, split: str, source_index: int, fields: list[str], source_path: str
) -> TaskRow | None:
    if len(fields) < 2:
        return None
    source = clean_text(fields[0])
    target = clean_text(fields[1])
    if not source or not target:
        return None
    category = clean_text(fields[2]) if len(fields) > 2 else ""
    task = category or "semantic_parsing"
    return TaskRow(
        id=stable_id(dataset, config, split, source_path, source_index),
        dataset=dataset,
        config=config,
        split=split,
        task=task,
        variant="full",
        input=source,
        expected_output=target,
        metadata={"source_key": source_path, "category": category},
    )
