#!/usr/bin/env python3
"""Fallback-E4B T1 labeling with immutable row retention and I5 guards."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoProcessor

from zagreus_scaled_kd_teacher import (
    LABELS,
    MODEL_ID,
    MODEL_REVISION,
    PERMUTATION_COUNT,
    permutation_orders,
    render_conversation,
)


EXPECTED_INPUT = "b499eefb0dfb5a443a01d1b5ef4dd92501d26f8204f7e83db06884c8f00c6bcd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def semantic_probabilities(display_logits: torch.Tensor, order: Sequence[int]) -> list[float]:
    displayed = torch.softmax(display_logits.float(), dim=-1).cpu().tolist()
    semantic = [0.0] * 4
    for display, semantic_index in enumerate(order):
        semantic[semantic_index] = float(displayed[display])
    return semantic


def normalize_teacher(row: dict[str, Any], teacher: dict[str, Any], source: str) -> dict[str, Any]:
    if teacher.get("model_id") != MODEL_ID or teacher.get("model_revision") != MODEL_REVISION:
        raise RuntimeError(f"Teacher identity mismatch: {row['id']}")
    views = []
    for item in teacher["permutations"]:
        probabilities = [math.exp(float(value)) for value in item["semantic_log_probabilities"]]
        total = sum(probabilities)
        views.append(
            {
                "view": int(item["view"]),
                "semantic_order_by_display_slot": list(item["semantic_order_by_display_slot"]),
                "probabilities": [value / total for value in probabilities],
                "top1_semantic": int(item["top1_semantic"]),
            }
        )
    probabilities = [float(value) for value in teacher["ensemble"]["probabilities"]]
    total = sum(probabilities)
    probabilities = [value / total for value in probabilities]
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "source": source,
        "views": sorted(views, key=lambda item: item["view"]),
        "probabilities": probabilities,
        "top1_semantic": int(max(range(4), key=probabilities.__getitem__)),
    }


def finalize_row(row: dict[str, Any], teacher: dict[str, Any]) -> dict[str, Any]:
    output = json.loads(json.dumps(row, ensure_ascii=False))
    output.pop("teacher_reuse", None)
    output["teacher_supervision"] = {
        "teachers": [teacher],
        "teacher_count": 1,
        "averaged_probabilities": teacher["probabilities"],
        "mean_pairwise_js_divergence_raw": 0.0,
        "agreement_scalar_status": "singleton_fallback_no_pairs",
        "agreement_weight": 1.0,
        "agreement_weight_map": "max(0.25, exp(-4 * mean_pairwise_js_divergence))",
        "rows_filtered_on_agreement": False,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    if sha256(args.input) != EXPECTED_INPUT:
        raise RuntimeError("T1 input hash drift")
    if any(value in str(args.input).casefold() for value in ("5_shots", "harness", "italic.jsonl")):
        raise RuntimeError("Denylisted T1 input path")
    rows = read_jsonl(args.input)
    if len(rows) != 21_000:
        raise RuntimeError("T1 input row count drift")
    for row in rows:
        authorized = bool(row["gold_ce_authorized"])
        if not authorized and row["campaign_gold"] is not None:
            raise RuntimeError(f"Unauthorized campaign gold: {row['id']}")

    output_path = args.output_dir / "t1_labeled.jsonl"
    completed = {row["id"]: row for row in read_jsonl(output_path)} if output_path.exists() else {}
    reused = []
    for row in rows:
        if row["id"] in completed or row.get("teacher_reuse") is None:
            continue
        reused.append(finalize_row(row, normalize_teacher(row, row["teacher_reuse"], "frozen_exact_reuse")))
    if reused:
        append_jsonl(output_path, reused)
        completed.update({row["id"]: row for row in reused})

    pending_rows = [row for row in rows if row["id"] not in completed]
    flat = [
        (row, view, order)
        for row in pending_rows
        for view, order in enumerate(permutation_orders(str(row["id"])))
    ]
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA required")
    resolved = model_info(MODEL_ID, revision=MODEL_REVISION).sha
    if resolved != MODEL_REVISION:
        raise RuntimeError("Fallback teacher revision drift")
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
    label_values = []
    for label in LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Label tokenization changed: {label} -> {ids}")
        label_values.append(ids[0])
    label_ids = torch.tensor(label_values, device=model.device)
    partial: dict[str, list[dict[str, Any]]] = defaultdict(list)
    started = time.perf_counter()
    for offset in range(0, len(flat), args.batch_size):
        batch = flat[offset : offset + args.batch_size]
        prompts = [render_conversation(tokenizer, row, order) for row, _, order in batch]
        encoded = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            outputs = model(**encoded, use_cache=False, return_dict=True, logits_to_keep=1)
            candidates = outputs.logits[:, -1, :].index_select(-1, label_ids)
        finished = []
        for (row, view, order), logits in zip(batch, candidates):
            probabilities = semantic_probabilities(logits, order)
            partial[str(row["id"])].append(
                {
                    "view": view,
                    "semantic_order_by_display_slot": list(order),
                    "probabilities": probabilities,
                    "top1_semantic": int(max(range(4), key=probabilities.__getitem__)),
                }
            )
            if len(partial[str(row["id"])]) == PERMUTATION_COUNT:
                views = sorted(partial.pop(str(row["id"])), key=lambda item: item["view"])
                mean = [sum(item["probabilities"][index] for item in views) / 4 for index in range(4)]
                total = sum(mean)
                mean = [value / total for value in mean]
                teacher = {
                    "model_id": MODEL_ID,
                    "revision": MODEL_REVISION,
                    "source": "new_t1_inference",
                    "views": views,
                    "probabilities": mean,
                    "top1_semantic": int(max(range(4), key=mean.__getitem__)),
                }
                finished.append(finalize_row(row, teacher))
        if finished:
            append_jsonl(output_path, finished)
            completed.update({row["id"]: row for row in finished})
        del encoded, outputs, candidates
        processed = min(offset + len(batch), len(flat))
        print(
            f"T1 views={processed}/{len(flat)} rate={processed/max(time.perf_counter()-started,1e-6):.2f}/s",
            flush=True,
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if len(completed) != len(rows):
        raise RuntimeError(f"GATE-L failed: labeled {len(completed)} of {len(rows)}")
    ordered_ids = [str(row["id"]) for row in rows]
    labeled_ids = [str(row["id"]) for row in read_jsonl(output_path)]
    if set(ordered_ids) != set(labeled_ids) or len(labeled_ids) != len(ordered_ids):
        raise RuntimeError("GATE-L ID conservation failed")
    manifest = {
        "protocol": "zagreus-propositional-v1-gate-l",
        "status": "PASSED",
        "attempted_rows": len(rows),
        "labeled_rows": len(labeled_ids),
        "dropped_rows": 0,
        "reused_exact_teacher_rows": len(reused),
        "new_inference_rows": len(pending_rows),
        "rows_filtered_on_teacher_or_source_agreement": 0,
        "teacher": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "count": 1},
        "agreement": {
            "raw_mean_pairwise_js_divergence": 0.0,
            "status": "singleton_fallback_no_pairs",
            "weight": 1.0,
        },
        "input_sha256": sha256(args.input),
        "labeled_sha256": sha256(output_path),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "new_inference_seconds": time.perf_counter() - started,
        "official_italic_rows_read_or_used": 0,
    }
    write_json(args.output_dir / "GATE_L.json", manifest)
    print("GATE_L=" + json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
