#!/usr/bin/env python3
"""Shared frozen data/output contract for ENSEMBLE-v1 T2 labeling."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

LABELS = "ABCD"
EXPECTED_TRAIN = "ee11c843242316f864de291dfd357fed59f4248d40c5d99d2be2968c127f39e9"
SYSTEM_PROMPT = (
    "Rispondi al quesito a scelta multipla. Valuta il contenuto delle opzioni "
    "e restituisci soltanto la lettera corretta, senza spiegazioni."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_rows(path: Path) -> list[dict[str, Any]]:
    if sha256(path) != EXPECTED_TRAIN:
        raise RuntimeError("T2 frozen payload hash contract failed")
    rows = read_jsonl(path)
    if len(rows) != 21_000:
        raise RuntimeError("T2 frozen payload row-count contract failed")
    for row in rows:
        if str(row.get("source", "")).casefold() == "italic" or str(row.get("id", "")).casefold().startswith("italic:"):
            raise RuntimeError("Official row reached T2")
        if len(row["options"]) != 4:
            raise RuntimeError(f"Malformed T2 row: {row.get('id')}")
        frozen = row["teacher_supervision"]["teachers"][0]["views"]
        if [int(item["view"]) for item in frozen] != list(range(4)):
            raise RuntimeError(f"Malformed frozen views: {row['id']}")
        if any(sorted(map(int, item["semantic_order_by_display_slot"])) != list(range(4)) for item in frozen):
            raise RuntimeError(f"Malformed frozen order: {row['id']}")
    return rows


def orders(row: dict[str, Any]) -> list[tuple[int, ...]]:
    return [tuple(map(int, item["semantic_order_by_display_slot"])) for item in row["teacher_supervision"]["teachers"][0]["views"]]


def question_text(row: dict[str, Any]) -> str:
    question = str(row["question"]).strip()
    return f"{str(row['context']).strip()}\n\n{question}" if row.get("context") else question


def user_prompt(row: dict[str, Any], order: Sequence[int]) -> str:
    options = "\n".join(f"{LABELS[display]}. {row['options'][semantic]}" for display, semantic in enumerate(order))
    return f"Domanda: {question_text(row)}\n\nOpzioni:\n{options}\n\nRisposta:"


def conversation(row: dict[str, Any], order: Sequence[int]) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt(row, order)}]


def semantic_probabilities(display_log_probs: Sequence[float], order: Sequence[int]) -> list[float]:
    semantic = [0.0] * 4
    for display, semantic_index in enumerate(order):
        semantic[semantic_index] = math.exp(float(display_log_probs[display]))
    total = sum(semantic)
    return [value / total for value in semantic]


def completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    result = {str(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("Duplicate row in resumable T2 output")
    return result


def row_result(row: dict[str, Any], views: list[dict[str, Any]]) -> dict[str, Any]:
    views.sort(key=lambda item: item["view"])
    if [item["view"] for item in views] != list(range(4)):
        raise RuntimeError(f"Incomplete T2 views: {row['id']}")
    mean = [sum(item["probabilities"][option] for item in views) / 4 for option in range(4)]
    return {
        "id": str(row["id"]), "domain": str(row["domain"]), "views": views,
        "probabilities": mean, "top1_semantic": max(range(4), key=mean.__getitem__),
    }


def write_manifest(output_dir: Path, model: dict[str, Any], environment: dict[str, Any], expected_rows: int = 21_000, row_range: list[int] | None = None) -> dict[str, Any]:
    output_path = output_dir / "predictions.jsonl"
    rows = read_jsonl(output_path)
    if len(rows) != expected_rows or len({row["id"] for row in rows}) != expected_rows:
        raise RuntimeError("T2 completion contract failed")
    counts = {str(option): sum(row["top1_semantic"] == option for row in rows) for option in range(4)}
    manifest = {
        "protocol": "zagreus-ensemble-v1-t2-single-question-four-view",
        "official_italic_read_or_used": False,
        "model": model, "environment": environment,
        "input_sha256": EXPECTED_TRAIN, "rows": len(rows), "views": len(rows) * 4,
        "semantic_top1_counts": counts, "predictions_sha256": sha256(output_path),
    }
    if row_range is not None:
        manifest["source_row_range_half_open"] = row_range
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
