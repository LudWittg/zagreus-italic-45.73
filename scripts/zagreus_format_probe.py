#!/usr/bin/env python3
"""Paired ARM-A format probe on dev_early without loading official target rows."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


LABELS = "ABCD"
PLAIN_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% if not loop.last %}\n\n{% endif %}{% endfor %}"
EXPECTED_MODEL = "f9906c629fc7b34048005b9f5f173fab75fd5f81d0d05619073d044be0c71ef8"
EXPECTED_DEV = "3f943adb939e50d53ebf28eeef84244cb0d4fd4dd29f7af97af924a3e1612732"
EXPECTED_CURRENT_SELECTION = 0.39666666666666667
EXPECTED_CURRENT_GENERATION = 0.4083333333333333
GENERATION_SEED = 20260807


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def stable_rng(seed: int, *parts: object) -> random.Random:
    payload = ":".join([str(seed), *(str(part) for part in parts)])
    value = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")
    return random.Random(value)


def non_identity_order(row_id: str, step: int, count: int = 4) -> tuple[int, ...]:
    identity = tuple(range(count))
    attempt = 0
    while True:
        values = list(identity)
        stable_rng(20260801, "kd-view", row_id, step, attempt).shuffle(values)
        order = tuple(values)
        if order != identity:
            return order
        attempt += 1


def question_text(row: dict[str, Any]) -> str:
    question = str(row["question"]).strip()
    if row.get("context"):
        question = f"{str(row['context']).strip()}\n\n{question}"
    return question


def target_prompt(row: dict[str, Any], order: Sequence[int], variable_labels: bool) -> str:
    option_count = len(order)
    options = "\n".join(
        f"{LABELS[display]}) {row['options'][semantic]}" for display, semantic in enumerate(order)
    )
    allowed = LABELS[:option_count] if variable_labels else "ABCD"
    topic = str(row.get("domain") or "lingua_italiana")
    return (
        "Rispondi alla seguente domanda a scelta multipla sull'argomento "
        f"'{topic}'. La tua risposta deve essere nel seguente formato: 'LETTERA' "
        f"(senza virgolette) dove LETTERA è una tra {allowed}. Scrivi solo la lettera "
        "corrispondente alla tua risposta senza spiegazioni.\n\n"
        f"{question_text(row)}\n\n{options}\n\nRisposta:"
    )


def plain_template_render(messages: Sequence[dict[str, str]]) -> str:
    # The frozen Jinja template contains a blank source line inside a block, but
    # Jinja consumes one newline at each block boundary. Both the official
    # Transformers 4.55.2 stack and the 5.13.1 scoring stack therefore render
    # exactly one newline between adjacent message contents.
    return "\n".join(message["content"] for message in messages)


def render(row: dict[str, Any], order: Sequence[int], condition: str, prefix: Sequence[dict[str, str]]) -> str:
    target = target_prompt(row, order, variable_labels=(condition == "official_five_shot"))
    if condition == "current_single_turn":
        return target
    return plain_template_render([*prefix, {"role": "user", "content": target}])


def answer_token_ids(tokenizer: Any) -> torch.Tensor:
    values = []
    for label in LABELS:
        ids = tokenizer(f" {label}", add_special_tokens=False).input_ids
        if len(ids) != 1 or tokenizer.decode(ids).strip() != label:
            raise RuntimeError(f"label-token contract failed for {label}: {ids}")
        values.append(ids[0])
    return torch.tensor(values, device="cuda")


def encode(tokenizer: Any, prompts: Sequence[str], max_length: int) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(prompts), add_special_tokens=False, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    return {key: value.cuda(non_blocking=True) for key, value in encoded.items()}


def macro_accuracy(rows: Sequence[dict[str, Any]]) -> float:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[str(row["domain"])].append(bool(row["correct"]))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


@torch.inference_mode()
def score_four_view(
    model: Any, tokenizer: Any, rows: Sequence[dict[str, Any]], prefix: Sequence[dict[str, str]],
    condition: str, label_ids: torch.Tensor, batch_size: int, max_length: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    all_predictions = []
    views = []
    for view in range(4):
        predictions = []
        total_nll = 0.0
        for offset in range(0, len(rows), batch_size):
            batch_rows = rows[offset:offset + batch_size]
            orders = [
                tuple(range(4)) if view == 0 else non_identity_order(str(row["id"]), 10_000_000 + view)
                for row in batch_rows
            ]
            prompts = [render(row, order, condition, prefix) for row, order in zip(batch_rows, orders)]
            outputs = model(**encode(tokenizer, prompts, max_length), use_cache=False, return_dict=True, logits_to_keep=1)
            display_logits = outputs.logits[:, -1, :].float().index_select(-1, label_ids)
            semantic_logits = torch.empty_like(display_logits)
            for batch_index, order in enumerate(orders):
                for display, semantic in enumerate(order):
                    semantic_logits[batch_index, semantic] = display_logits[batch_index, display]
            log_probs = F.log_softmax(semantic_logits, dim=-1)
            gold = torch.tensor([int(row["gold"]) for row in batch_rows], device="cuda")
            total_nll += float(F.nll_loss(log_probs, gold, reduction="sum"))
            predicted = semantic_logits.argmax(-1).cpu().tolist()
            probabilities = log_probs.exp().cpu().tolist()
            for row, pred, probs in zip(batch_rows, predicted, probabilities):
                predictions.append({
                    "id": str(row["id"]), "domain": str(row["domain"]), "gold": int(row["gold"]),
                    "prediction": int(pred), "correct": int(pred) == int(row["gold"]),
                    "probabilities": probs, "view": view,
                })
        views.append({
            "view": view, "n": len(predictions),
            "accuracy": sum(row["correct"] for row in predictions) / len(predictions),
            "macro_accuracy": macro_accuracy(predictions), "nll": total_nll / len(predictions),
        })
        all_predictions.extend(predictions)

    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_predictions:
        by_id[row["id"]].append(row)
    agreements = []
    ensemble = []
    for row_id, predictions in by_id.items():
        predictions.sort(key=lambda row: row["view"])
        choices = [row["prediction"] for row in predictions]
        agreements.append(np.mean([left == right for left, right in itertools.combinations(choices, 2)]))
        mean_probs = np.mean(np.asarray([row["probabilities"] for row in predictions]), axis=0)
        gold = predictions[0]["gold"]
        pred = int(np.argmax(mean_probs))
        ensemble.append({"id": row_id, "domain": predictions[0]["domain"], "gold": gold, "prediction": pred, "correct": pred == gold})
    return {
        "views": views,
        "selection_score": float(np.mean([view["macro_accuracy"] for view in views])),
        "canonical_accuracy": views[0]["accuracy"],
        "canonical_nll": views[0]["nll"],
        "four_view_agreement": float(np.mean(agreements)),
        "probability_ensemble_accuracy": sum(row["correct"] for row in ensemble) / len(ensemble),
        "probability_ensemble_macro": macro_accuracy(ensemble),
    }, all_predictions


def generation_orders(rows: Sequence[dict[str, Any]]) -> list[tuple[dict[str, Any], tuple[int, ...], str]]:
    ordered = sorted(rows, key=lambda row: hashlib.sha256(f"{GENERATION_SEED}:dev_early:{row['id']}".encode()).hexdigest())
    examples = []
    for index, row in enumerate(ordered):
        gold_semantic = int(row["gold"])
        gold_display = index % 4
        remaining = [value for value in range(4) if value != gold_semantic]
        stable_rng(GENERATION_SEED, "p2b-generation-order", "dev_early", str(row["id"])).shuffle(remaining)
        order: list[int | None] = [None] * 4
        order[gold_display] = gold_semantic
        cursor = 0
        for display in range(4):
            if order[display] is None:
                order[display] = remaining[cursor]
                cursor += 1
        examples.append((row, tuple(int(value) for value in order), LABELS[gold_display]))
    return examples


@torch.inference_mode()
def score_generation(
    model: Any, tokenizer: Any, examples: Sequence[tuple[dict[str, Any], tuple[int, ...], str]],
    prefix: Sequence[dict[str, str]], condition: str, answer_ids: set[int], batch_size: int, max_length: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    output = []
    for offset in range(0, len(examples), batch_size):
        batch_examples = examples[offset:offset + batch_size]
        prompts = [render(row, order, condition, prefix) for row, order, _ in batch_examples]
        batch = encode(tokenizer, prompts, max_length)
        width = batch["input_ids"].shape[1]
        sequences = model.generate(
            **batch, do_sample=False, max_new_tokens=3,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
        for (row, _, gold_label), ids in zip(batch_examples, sequences[:, width:].cpu().tolist()):
            text = tokenizer.decode(ids, skip_special_tokens=True)
            stripped = text.strip()
            match = re.search(r"[A-Z]", text)
            parsed = match.group(0) if match else ""
            output.append({
                "id": str(row["id"]), "domain": str(row["domain"]), "gold": gold_label,
                "output": text, "parsed": parsed, "correct": parsed == gold_label,
                "valid": parsed in LABELS, "exact": bool(re.fullmatch(r"[A-D]", stripped)),
                "immediate_eos": len(ids) >= 2 and ids[0] in answer_ids and ids[1] == tokenizer.eos_token_id,
            })
    counts = Counter(row["parsed"] for row in output if row["parsed"] in LABELS)
    return {
        "n": len(output), "accuracy": sum(row["correct"] for row in output) / len(output),
        "macro_accuracy": macro_accuracy(output),
        "valid_rate": sum(row["valid"] for row in output) / len(output),
        "exact_rate": sum(row["exact"] for row in output) / len(output),
        "immediate_eos_rate": sum(row["immediate_eos"] for row in output) / len(output),
        "predicted_counts": dict(sorted(counts.items())),
    }, output


def paired_summary(current: Sequence[dict[str, Any]], official: Sequence[dict[str, Any]]) -> dict[str, Any]:
    current_by_id = {row["id"]: row for row in current}
    official_by_id = {row["id"]: row for row in official}
    ids = sorted(current_by_id)
    losses = sum(current_by_id[row_id]["correct"] and not official_by_id[row_id]["correct"] for row_id in ids)
    gains = sum(official_by_id[row_id]["correct"] and not current_by_id[row_id]["correct"] for row_id in ids)
    domains = sorted({current_by_id[row_id]["domain"] for row_id in ids})
    return {
        "n": len(ids), "current_correct_official_wrong": losses,
        "current_wrong_official_correct": gains, "net_correct_change": gains - losses,
        "by_domain_accuracy_delta": {
            domain: float(np.mean([
                int(official_by_id[row_id]["correct"]) - int(current_by_id[row_id]["correct"])
                for row_id in ids if current_by_id[row_id]["domain"] == domain
            ])) for domain in domains
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    if transformers.__version__ != "5.13.1" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Transformers 5.13.1 and BF16 CUDA required")
    if sha256(args.model / "model.safetensors") != EXPECTED_MODEL or sha256(args.dev) != EXPECTED_DEV:
        raise RuntimeError("frozen model/dev hash mismatch")
    rows = read_jsonl(args.dev)
    prefix = json.loads(args.prefix.read_text(encoding="utf-8"))
    if len(rows) != 600 or {len(row["options"]) for row in rows} != {4}:
        raise RuntimeError("expected 600 four-option dev rows")
    if len(prefix) != 11 or [row["content"] for row in prefix if row["role"] == "assistant"] != list("ACCBA"):
        raise RuntimeError("official prefix contract drift")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = PLAIN_TEMPLATE
    probe_messages = [*prefix, {"role": "user", "content": target_prompt(rows[0], tuple(range(4)), True)}]
    if tokenizer.apply_chat_template(probe_messages, tokenize=False, add_generation_prompt=True) != plain_template_render(probe_messages):
        raise RuntimeError("plain-template render mismatch")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    label_ids = answer_token_ids(tokenizer)
    answer_ids = set(label_ids.tolist())
    token_lengths = {
        condition: [
            len(tokenizer(render(row, tuple(range(4)), condition, prefix), add_special_tokens=False).input_ids)
            for row in rows
        ] for condition in ("current_single_turn", "official_five_shot")
    }
    if max(token_lengths["official_five_shot"]) > args.max_length:
        raise RuntimeError("official-format probe would truncate")

    conditions = {}
    prediction_outputs = {}
    generation_examples = generation_orders(rows)
    started = time.perf_counter()
    for condition in ("current_single_turn", "official_five_shot"):
        four_view, predictions = score_four_view(
            model, tokenizer, rows, prefix, condition, label_ids, args.batch_size, args.max_length
        )
        generation, generation_rows = score_generation(
            model, tokenizer, generation_examples, prefix, condition, answer_ids,
            args.generation_batch_size, args.max_length,
        )
        conditions[condition] = {"four_view": four_view, "generation": generation}
        prediction_outputs[condition] = {"four_view": predictions, "generation": generation_rows}
        if condition == "current_single_turn":
            if abs(four_view["selection_score"] - EXPECTED_CURRENT_SELECTION) > 1e-12:
                raise RuntimeError("current selection parity failed: " + json.dumps(four_view, sort_keys=True))
            if abs(generation["accuracy"] - EXPECTED_CURRENT_GENERATION) > 1e-12:
                raise RuntimeError("current generation parity failed: " + json.dumps(generation, sort_keys=True))

    current = conditions["current_single_turn"]
    official = conditions["official_five_shot"]
    result = {
        "protocol": "zagreus-format-probe-v1",
        "status": "PASSED",
        "official_eval_rows_loaded": 0,
        "model_weight_sha256": EXPECTED_MODEL,
        "dev_sha256": EXPECTED_DEV,
        "prefix_sha256": sha256(args.prefix),
        "prefix": {"messages": len(prefix), "demonstrations": 5, "demo_gold_letters": "ACCBA", "contains_three_option_demo": True},
        "dev_option_counts": {"4": 600},
        "token_lengths": {
            condition: {"minimum": min(values), "mean": float(np.mean(values)), "maximum": max(values), "truncated": 0}
            for condition, values in token_lengths.items()
        },
        "conditions": conditions,
        "delta_official_minus_current": {
            "four_view_selection": official["four_view"]["selection_score"] - current["four_view"]["selection_score"],
            "canonical_accuracy": official["four_view"]["canonical_accuracy"] - current["four_view"]["canonical_accuracy"],
            "generation_accuracy": official["generation"]["accuracy"] - current["generation"]["accuracy"],
            "four_view_agreement": official["four_view"]["four_view_agreement"] - current["four_view"]["four_view_agreement"],
        },
        "paired_generation": paired_summary(
            prediction_outputs["current_single_turn"]["generation"],
            prediction_outputs["official_five_shot"]["generation"],
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "limitations": [
            "dev_early contains only four-option targets, so this isolates the five-shot conversation prefix but not the benchmark's three-option target-instruction shift.",
            "dev_early is adaptive and covers six sectors; this is an instrument diagnostic, not a promotion measurement."
        ],
    }
    write_json(args.output / "result.json", result)
    for condition, values in prediction_outputs.items():
        write_jsonl(args.output / f"{condition}_generation.jsonl", values["generation"])
        write_jsonl(args.output / f"{condition}_four_view.jsonl", values["four_view"])
    print("FORMAT_PROBE_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
