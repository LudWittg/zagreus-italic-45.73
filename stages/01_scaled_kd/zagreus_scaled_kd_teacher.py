"""Attach permutation-ensemble Gemma soft labels to clean source MCQs."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoProcessor

from zagreus_language_synthetic_data import prompt_hash, sha256_file


MODEL_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
LABELS = "ABCD"
PERMUTATION_COUNT = 4
SYSTEM_PROMPT = (
    "Rispondi al quesito a scelta multipla. Valuta il contenuto delle opzioni "
    "e restituisci soltanto la lettera corretta, senza spiegazioni."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def permutation_orders(row_id: str, count: int = 4) -> list[tuple[int, ...]]:
    orders = [tuple(range(count))]
    attempt = 0
    while len(orders) < PERMUTATION_COUNT:
        values = list(range(count))
        seed = int(prompt_hash(["scaled-kd-teacher-permutation", row_id, attempt])[:16], 16)
        random.Random(seed).shuffle(values)
        order = tuple(values)
        if order not in orders:
            orders.append(order)
        attempt += 1
        if attempt > 100:
            raise RuntimeError(f"Could not create unique permutations for {row_id}")
    return orders


def question_text(row: dict[str, Any]) -> str:
    question = str(row["question"]).strip()
    if row.get("context"):
        question = f"{str(row['context']).strip()}\n\n{question}"
    return question


def render_prompt(row: dict[str, Any], semantic_order: Sequence[int]) -> str:
    options = "\n".join(
        f"{LABELS[display]}. {row['options'][semantic]}"
        for display, semantic in enumerate(semantic_order)
    )
    return f"Domanda: {question_text(row)}\n\nOpzioni:\n{options}\n\nRisposta:"


def render_conversation(tokenizer: Any, row: dict[str, Any], order: Sequence[int]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_prompt(row, order)},
    ]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def semantic_log_probabilities(
    displayed: Sequence[float], order: Sequence[int]
) -> list[float]:
    semantic = [0.0] * len(order)
    for display, semantic_index in enumerate(order):
        semantic[semantic_index] = float(displayed[display])
    return semantic


def ensemble(permutations: Sequence[dict[str, Any]], gold: int) -> dict[str, Any]:
    probabilities = [
        [math.exp(value) for value in item["semantic_log_probabilities"]]
        for item in permutations
    ]
    mean = [
        sum(row[index] for row in probabilities) / len(probabilities)
        for index in range(4)
    ]
    total = sum(mean)
    mean = [value / total for value in mean]
    top1 = max(range(4), key=mean.__getitem__)
    permutation_top1 = [int(item["top1_semantic"]) for item in permutations]
    eligible = permutation_top1.count(gold) >= 3 and mean[gold] >= 0.35
    return {
        "probabilities": mean,
        "log_probabilities": [math.log(max(value, 1e-12)) for value in mean],
        "top1_semantic": top1,
        "gold_probability": mean[gold],
        "permutation_top1": permutation_top1,
        "permutation_top1_agreement": max(Counter(permutation_top1).values())
        / len(permutation_top1),
        "eligible": eligible,
        "eligibility_policy": "gold-top1-at-least-3-of-4-and-ensemble-pgold-at-least-0.35-v1",
        "training_mode": "gold-ce-plus-soft-kd" if eligible else "gold-ce-only",
    }


def validate_rows(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if "italic" in path.name.casefold():
        raise ValueError(f"Official-looking input path is forbidden: {path}")
    for row in rows:
        if str(row.get("id", "")).casefold().startswith("italic:"):
            raise ValueError(f"Official-looking row is forbidden: {row['id']}")
        if str(row.get("source", "")).casefold() == "italic":
            raise ValueError(f"Official source is forbidden: {row['id']}")
        if len(row["options"]) != 4 or not 0 <= int(row["gold"]) < 4:
            raise ValueError(f"Malformed row: {row['id']}")


def score_split(
    split: str,
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    label_ids: torch.Tensor,
    batch_size: int,
) -> dict[str, Any]:
    raw_path = output_dir / f"{split}_teacher_raw.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if raw_path.exists():
        completed = {row["id"]: row for row in read_jsonl(raw_path)}
    flat = [
        (row, view, order)
        for row in rows
        if row["id"] not in completed
        for view, order in enumerate(permutation_orders(str(row["id"])))
    ]
    scores: dict[str, list[dict[str, Any]]] = defaultdict(list)
    started = time.time()
    with raw_path.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(flat), batch_size):
            batch = flat[offset : offset + batch_size]
            rendered = [render_conversation(tokenizer, row, order) for row, _, order in batch]
            encoded = tokenizer(rendered, padding=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                outputs = model(
                    **encoded,
                    use_cache=False,
                    return_dict=True,
                    logits_to_keep=1,
                )
                candidate = outputs.logits[:, -1, :].float().index_select(-1, label_ids)
                displayed = torch.log_softmax(candidate, dim=-1).cpu().tolist()
            del encoded, outputs, candidate
            for (row, view, order), values in zip(batch, displayed):
                semantic = semantic_log_probabilities(values, order)
                probabilities = [math.exp(value) for value in semantic]
                scores[str(row["id"])].append(
                    {
                        "view": view,
                        "semantic_order_by_display_slot": list(order),
                        "semantic_log_probabilities": semantic,
                        "top1_semantic": max(range(4), key=semantic.__getitem__),
                        "gold_probability": probabilities[int(row["gold"])],
                        "prompt_sha256": prompt_hash(
                            {
                                "system": SYSTEM_PROMPT,
                                "user": render_prompt(row, order),
                            }
                        ),
                    }
                )
            finished_ids = {
                str(row["id"])
                for row, _, _ in batch
                if len(scores[str(row["id"])]) == PERMUTATION_COUNT
            }
            for row_id in sorted(finished_ids):
                source = next(row for row, _, _ in batch if str(row["id"]) == row_id)
                permutations = sorted(scores.pop(row_id), key=lambda value: value["view"])
                payload = json.loads(json.dumps(source, ensure_ascii=False))
                payload["teacher"] = {
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "candidate_space": "semantic-options",
                    "permutation_count": PERMUTATION_COUNT,
                    "permutations": permutations,
                    "ensemble": ensemble(permutations, int(source["gold"])),
                }
                completed[row_id] = payload
                handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            processed = min(offset + len(batch), len(flat))
            rate = processed / max(time.time() - started, 1e-6)
            print(
                f"teacher {split}: views={processed}/{len(flat)} rate={rate:.2f}/s",
                flush=True,
            )

    ordered = [completed[str(row["id"])] for row in rows]
    output_path = output_dir / f"{split}.jsonl"
    write_jsonl(output_path, ordered)
    eligible = [bool(row["teacher"]["ensemble"]["eligible"]) for row in ordered]
    gold_probabilities = [
        float(row["teacher"]["ensemble"]["gold_probability"]) for row in ordered
    ]
    return {
        "rows": len(ordered),
        "eligible": sum(eligible),
        "eligible_rate": sum(eligible) / len(eligible),
        "teacher_ensemble_accuracy": sum(
            int(row["teacher"]["ensemble"]["top1_semantic"]) == int(row["gold"])
            for row in ordered
        )
        / len(ordered),
        "mean_gold_probability": sum(gold_probabilities) / len(gold_probabilities),
        "by_domain": {
            domain: {
                "rows": len(domain_rows),
                "ensemble_accuracy": sum(
                    int(row["teacher"]["ensemble"]["top1_semantic"])
                    == int(row["gold"])
                    for row in domain_rows
                )
                / len(domain_rows),
                "eligible_rate": sum(
                    bool(row["teacher"]["ensemble"]["eligible"])
                    for row in domain_rows
                )
                / len(domain_rows),
            }
            for domain in sorted({str(row["domain"]) for row in ordered})
            for domain_rows in [[row for row in ordered if row["domain"] == domain]]
        },
        "sha256": sha256_file(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    args = parser.parse_args()
    if args.model_id != MODEL_ID or args.model_revision != MODEL_REVISION:
        raise ValueError("The teacher model and revision are locked for this experiment")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16 CUDA GPU is required")

    inputs = {
        split: read_jsonl(args.data_dir / f"{split}.jsonl") for split in ("train", "dev")
    }
    for split, rows in inputs.items():
        validate_rows(rows, args.data_dir / f"{split}.jsonl")

    args.output_dir.mkdir(parents=True, exist_ok=True)
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
            raise ValueError(f"Teacher label is not one token: {label} -> {ids}")
        label_values.append(int(ids[0]))
    label_ids = torch.tensor(label_values, device=model.device)
    torch.cuda.reset_peak_memory_stats()

    summary = {
        "protocol": "zagreus-scaled-sector-kd-teacher-v1",
        "official_italic_read_or_used": False,
        "teacher": {
            "model_id": MODEL_ID,
            "revision": model_info(MODEL_ID, revision=MODEL_REVISION).sha,
            "permutations": PERMUTATION_COUNT,
            "ensemble": "arithmetic-mean-semantic-probabilities",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "splits": {},
        "input_hashes": {
            split: sha256_file(args.data_dir / f"{split}.jsonl") for split in inputs
        },
    }
    for split in ("train", "dev"):
        summary["splits"][split] = score_split(
            split,
            inputs[split],
            args.output_dir,
            model,
            tokenizer,
            label_ids,
            args.batch_size,
        )
        write_json(args.output_dir / "manifest.json", summary)
    summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    write_json(args.output_dir / "manifest.json", summary)
    print("SCALED_KD_TEACHER=" + json.dumps(summary, ensure_ascii=False), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
