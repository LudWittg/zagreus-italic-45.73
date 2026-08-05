#!/usr/bin/env python3
"""Compute the frozen 36% checkpoint's four-view posterior on P2F replay rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import zagreus_interface_calibration as interface
import zagreus_scaled_kd_train as trainlib


EXPECTED_BASE = "a6111331617aadb607863149362d2727a264dc137f21dcfd974c535fd62f5504"
EXPECTED_TRAIN = "5c45bcf4c4b1fe24a7c30661aaea75cb49252791461a60037e60b05e24ea77df"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_hash(row: dict[str, Any], order: tuple[int, ...]) -> str:
    return hashlib.sha256(trainlib.render_prompt(row, order).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16 CUDA GPU is required")
    if sha256(args.base_model / "model.safetensors") != EXPECTED_BASE:
        raise ValueError("Locked 36% base hash mismatch")
    if sha256(args.train) != EXPECTED_TRAIN:
        raise ValueError("Frozen P2F train hash mismatch")
    rows = trainlib.read_jsonl(args.train)
    replay = sorted(
        (row for row in rows if str(row.get("source_split")) == "unassigned"),
        key=lambda row: str(row["id"]),
    )
    if len(replay) != 850 or any(row.get("p2b_origin") is not None for row in replay):
        raise ValueError(f"Expected exactly 850 replay rows, found {len(replay)}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = interface.CHAT_TEMPLATE
    label_ids = trainlib.answer_token_ids(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).cuda()
    model.config.use_cache = False
    model.eval()
    records = {
        str(row["id"]): {
            "id": str(row["id"]),
            "source_split": "unassigned",
            "permutations": [],
        }
        for row in replay
    }
    with torch.no_grad():
        for offset in range(0, len(replay), args.batch_size):
            batch_rows = replay[offset : offset + args.batch_size]
            orders_by_row = trainlib.p2c_orders(batch_rows)
            for view_index in range(4):
                orders = [orders[view_index] for orders in orders_by_row]
                logits = trainlib.student_semantic_logits(
                    model, tokenizer, batch_rows, orders, label_ids, args.max_length
                )
                probabilities = F.softmax(logits, dim=-1).cpu().tolist()
                for row, order, values in zip(batch_rows, orders, probabilities):
                    records[str(row["id"])]["permutations"].append(
                        {
                            "view": view_index,
                            "semantic_order_by_display_slot": list(order),
                            "semantic_probabilities": values,
                            "prompt_sha256": prompt_hash(row, order),
                        }
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row_id in sorted(records):
            handle.write(json.dumps(records[row_id], ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "protocol": "p2f-frozen-36pct-four-view-replay-anchor-v1",
        "model_weight_sha256": EXPECTED_BASE,
        "train_sha256": EXPECTED_TRAIN,
        "rows": len(records),
        "views_per_row": 4,
        "temperature": 1.0,
        "serialization": "locked plain Risposta template",
        "output_sha256": sha256(args.output),
        "official_italic_rows_read": 0,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("P2F_ANCHOR=" + json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
