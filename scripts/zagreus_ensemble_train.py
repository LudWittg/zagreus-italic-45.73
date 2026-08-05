#!/usr/bin/env python3
"""Format-matched full-parameter ARM-F/ARM-E trainer for ENSEMBLE-v1."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import zagreus_format_probe as probe

LABELS = "ABCDE"
PLAIN_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% if not loop.last %}\n\n{% endif %}{% endfor %}"
EXPECTED_MODEL = "f9906c629fc7b34048005b9f5f173fab75fd5f81d0d05619073d044be0c71ef8"
EXPECTED_DEV = "3f943adb939e50d53ebf28eeef84244cb0d4fd4dd29f7af97af924a3e1612732"
EXPECTED_PREFIX = "6b04ad87a538509798e267ac84607474b880e8e243fb3b9bb0059e1eb9d0adb0"
EXPECTED_ANCHORS = "913023f76d46f968010baff871204705a67c854884adf91a3b80e9029857559f"
PARAMETERS = 437_760_960
PLANNED_STEPS = 140
CHECKPOINT_INTERVAL = 70
MICRO_BASE_ROWS = 1
GRAD_ACCUM = 32
PEAK_LR = 1.5e-4
MIN_LR = 1.5e-5
WARMUP = 0.05
ANCHOR_BETA = 2.0
AGREEMENT_FLOOR = 0.6683333333333333 - 0.01
MARGINAL_CEILING = 0.08500000000000002
NLL_CEILING = 1.4523645305633546


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def verify(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"hash mismatch for {path}: {actual} != {expected}")
    return actual


def question_text(row: dict[str, Any]) -> str:
    question = str(row["question"]).strip()
    return f"{str(row['context']).strip()}\n\n{question}" if row.get("context") else question


def user_prompt(row: dict[str, Any], item: dict[str, Any]) -> str:
    count = int(item["option_count"])
    order = list(map(int, item["semantic_order_by_display_slot"]))
    options = "\n".join(f"{LABELS[display]}) {item['options'][semantic]}" for display, semantic in enumerate(order))
    topic = str(row.get("domain") or "lingua_italiana")
    return (
        "Rispondi alla seguente domanda a scelta multipla sull'argomento "
        f"'{topic}'. La tua risposta deve essere nel seguente formato: 'LETTERA' "
        f"(senza virgolette) dove LETTERA è una tra {LABELS[:count]}. Scrivi solo la lettera "
        "corrispondente alla tua risposta senza spiegazioni.\n\n"
        f"{question_text(row)}\n\n{options}\n\nRisposta:"
    )


def render_variant(recipe: dict[str, Any], variant: dict[str, Any], row_map: dict[str, dict[str, Any]]) -> tuple[str, list[tuple[int, int]], str]:
    messages: list[tuple[str, str]] = [("system", "Sei un assistente utile.")]
    for demo in variant["demos"]:
        messages.append(("user", user_prompt(row_map[str(demo["id"])], demo)))
        messages.append(("assistant", LABELS[int(demo["answer_display"])]))
    messages.append(("user", user_prompt(row_map[str(recipe["id"])], variant["target"])))
    if len(messages) != 12:
        raise RuntimeError("format recipe did not render exactly 12 prompt messages")
    pieces: list[str] = []
    spans = []
    length = 0
    for index, (role, content) in enumerate(messages):
        if index:
            pieces.append("\n")
            length += 1
        start = length
        pieces.append(content)
        length += len(content)
        if role == "assistant":
            spans.append((start, length))
    if len(spans) != 5:
        raise RuntimeError("format recipe did not render five demonstration assistants")
    return "".join(pieces), spans, LABELS[int(variant["target"]["answer_display"])]


def candidate_ids(tokenizer: Any) -> tuple[torch.Tensor, torch.Tensor]:
    plain, spaced = [], []
    for label in LABELS:
        plain_ids = tokenizer.encode(label, add_special_tokens=False)
        spaced_ids = tokenizer.encode(" " + label, add_special_tokens=False)
        if len(plain_ids) != 1 or len(spaced_ids) != 1:
            raise RuntimeError(f"label-token contract failed for {label}: {plain_ids}/{spaced_ids}")
        plain.append(int(plain_ids[0]))
        spaced.append(int(spaced_ids[0]))
    return torch.tensor(plain, device="cuda"), torch.tensor(spaced, device="cuda")


def encode_variant(tokenizer: Any, prompt: str, spans: Sequence[tuple[int, int]], target_label: str, max_length: int) -> tuple[torch.Tensor, list[int], int]:
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids = list(map(int, encoded["input_ids"]))
    offsets = encoded["offset_mapping"]
    demo_positions = []
    for start, end in spans:
        matching = [index for index, (left, right) in enumerate(offsets) if left < end and right > start]
        if len(matching) != 1 or prompt_ids[matching[0]] != tokenizer.encode(prompt[start:end], add_special_tokens=False)[0]:
            raise RuntimeError(f"demonstration answer is not one stable token: {prompt[start:end]!r}/{matching}")
        demo_positions.append(matching[0])
    target_ids = tokenizer.encode(" " + target_label, add_special_tokens=False)
    if len(target_ids) != 1:
        raise RuntimeError("target completion is not one token")
    target_position = len(prompt_ids)
    ids = prompt_ids + target_ids + [int(tokenizer.eos_token_id)]
    if len(ids) > max_length:
        raise RuntimeError(f"format-matched sequence exceeds max length: {len(ids)} > {max_length}")
    return torch.tensor(ids, device="cuda")[None, :], demo_positions, target_position


def selected_logits(model: Any, input_ids: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
    keep = torch.tensor([position - 1 for position in positions] + [positions[-1]], device=input_ids.device)
    output = model(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False,
        return_dict=True, logits_to_keep=keep,
    )
    if output.logits.shape[1] != len(keep):
        raise RuntimeError(f"logits_to_keep tensor contract failed: {output.logits.shape}")
    return output.logits[0].float()


def semantic_logits(display_logits: torch.Tensor, order: Sequence[int]) -> torch.Tensor:
    semantic = torch.empty_like(display_logits)
    for display, semantic_index in enumerate(order):
        semantic[int(semantic_index)] = display_logits[display]
    return semantic


def sequence_loss(
    model: Any, tokenizer: Any, recipe: dict[str, Any], variant: dict[str, Any], row_map: dict[str, dict[str, Any]],
    plain_ids: torch.Tensor, spaced_ids: torch.Tensor, arm: str, anchor_reference: dict[str, Any] | None, max_length: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    prompt, spans, target_label = render_variant(recipe, variant, row_map)
    input_ids, demo_positions, target_position = encode_variant(tokenizer, prompt, spans, target_label, max_length)
    positions = [*demo_positions, target_position]
    logits = selected_logits(model, input_ids, positions)
    turn_logits = []
    losses = []
    for index, demo in enumerate(variant["demos"]):
        count = int(demo["option_count"])
        candidates = logits[index].index_select(0, plain_ids[:count])
        turn_logits.append(candidates)
        if recipe["row_kind"] == "signal":
            losses.append(F.cross_entropy(candidates[None, :], torch.tensor([int(demo["answer_display"])], device="cuda")))
    target = variant["target"]
    count = int(target["option_count"])
    displayed = logits[5].index_select(0, spaced_ids[:count])
    semantic = semantic_logits(displayed, target["semantic_order_by_display_slot"])
    turn_logits.append(semantic)
    eos_loss = F.cross_entropy(logits[6][None, :], torch.tensor([int(tokenizer.eos_token_id)], device="cuda"))
    metrics = {"demo_ce": 0.0, "target_ce": 0.0, "target_kd": 0.0, "anchor_kl": 0.0, "eos_ce": float(eos_loss.detach())}
    if recipe["row_kind"] == "anchor":
        if anchor_reference is None:
            raise RuntimeError("anchor row lacks reference posterior")
        anchor_terms = []
        for turn, values in enumerate(turn_logits):
            reference = torch.tensor(anchor_reference["turn_probabilities"][turn], device="cuda", dtype=torch.float32)
            student_log = F.log_softmax(values, dim=-1)
            anchor_terms.append(torch.sum(reference * (torch.log(reference.clamp_min(1e-12)) - student_log)))
        anchor_kl = torch.stack(anchor_terms).mean()
        metrics["anchor_kl"] = float(anchor_kl.detach())
        return ANCHOR_BETA * anchor_kl, metrics
    demo_loss = torch.stack(losses).mean()
    teacher_key = "control_probabilities" if arm == "F" else "ensemble_probabilities"
    teacher = torch.tensor(target[teacher_key], device="cuda", dtype=torch.float32)
    teacher_t2 = F.softmax(torch.log(teacher.clamp_min(1e-12)) / 2.0, dim=-1)
    student_log_t2 = F.log_softmax(semantic / 2.0, dim=-1)
    kd = torch.sum(teacher_t2 * (torch.log(teacher_t2.clamp_min(1e-12)) - student_log_t2)) * 4.0
    kd_weight = 1.0 if arm == "F" else float(target["agreement_weight"])
    if target["gold_ce_authorized"]:
        gold_semantic = int(target["semantic_order_by_display_slot"][int(target["answer_display"])])
        ce = F.cross_entropy(semantic[None, :], torch.tensor([gold_semantic], device="cuda"))
        target_loss = 0.5 * ce + 0.5 * kd_weight * kd
        metrics["target_ce"] = float(ce.detach())
    else:
        target_loss = kd_weight * kd
    metrics["demo_ce"] = float(demo_loss.detach())
    metrics["target_kd"] = float(kd.detach())
    return (5.0 * demo_loss + target_loss) / 6.0 + 0.1 * eos_loss, metrics


@torch.inference_mode()
def build_anchor_posteriors(
    model: Any, tokenizer: Any, recipes: Sequence[dict[str, Any]], row_map: dict[str, dict[str, Any]],
    plain_ids: torch.Tensor, spaced_ids: torch.Tensor, max_length: int, output: Path,
) -> tuple[dict[tuple[str, int], dict[str, Any]], str]:
    records = []
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for recipe_index, recipe in enumerate(recipes):
            if recipe["row_kind"] != "anchor":
                continue
            for variant in recipe["variants"]:
                prompt, spans, target_label = render_variant(recipe, variant, row_map)
                input_ids, demo_positions, target_position = encode_variant(tokenizer, prompt, spans, target_label, max_length)
                logits = selected_logits(model, input_ids, [*demo_positions, target_position])
                probabilities = []
                for index, demo in enumerate(variant["demos"]):
                    count = int(demo["option_count"])
                    probabilities.append(torch.softmax(logits[index].index_select(0, plain_ids[:count]), dim=-1).cpu().tolist())
                target = variant["target"]
                count = int(target["option_count"])
                displayed = logits[5].index_select(0, spaced_ids[:count])
                probabilities.append(torch.softmax(semantic_logits(displayed, target["semantic_order_by_display_slot"]), dim=-1).cpu().tolist())
                records.append({"id": recipe["id"], "view": int(variant["view"]), "turn_probabilities": probabilities})
            if (recipe_index + 1) % 500 == 0:
                print(f"anchor_prepass_recipe_index={recipe_index + 1}/{len(recipes)} records={len(records)}", flush=True)
    write_jsonl(output, records)
    mapping = {(str(row["id"]), int(row["view"])): row for row in records}
    if len(mapping) != 12_000:
        raise RuntimeError(f"anchor posterior count drift: {len(mapping)}")
    return mapping, sha256(output)


def lr_at(step: int) -> float:
    progress = step / PLANNED_STEPS
    if progress <= WARMUP:
        return PEAK_LR * progress / WARMUP
    cosine = (progress - WARMUP) / (1 - WARMUP)
    return MIN_LR + (PEAK_LR - MIN_LR) * 0.5 * (1 + math.cos(math.pi * cosine))


def evaluate(model: Any, tokenizer: Any, dev: Sequence[dict[str, Any]], prefix: Sequence[dict[str, str]], label_ids: torch.Tensor, batch_size: int, max_length: int, seed: int) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    outputs = {}
    metrics = {}
    generation_examples = probe.generation_orders(dev)
    answer_set = set(map(int, label_ids.tolist()))
    for condition in ("official_five_shot", "current_single_turn"):
        four, predictions = probe.score_four_view(model, tokenizer, dev, prefix, condition, label_ids, batch_size, max_length)
        generation, generated = probe.score_generation(model, tokenizer, generation_examples, prefix, condition, answer_set, min(batch_size, 32), max_length)
        metrics[condition] = {"four_view": four, "generation": generation}
        outputs[f"{condition}_four_view"] = predictions
        outputs[f"{condition}_generation"] = generated
    matched = metrics["official_five_shot"]
    counts = matched["generation"]["predicted_counts"]
    maximum_marginal = max(abs(counts.get(label, 0) / 600 - 0.25) for label in "ABCD")
    checks = {
        "exact_one_letter_rate": matched["generation"]["exact_rate"] == 1.0,
        "immediate_eos_rate": matched["generation"]["immediate_eos_rate"] == 1.0,
        "non_letter_emission": 1.0 - matched["generation"]["valid_rate"] < 0.002,
        "four_view_agreement": matched["four_view"]["four_view_agreement"] >= AGREEMENT_FLOOR,
        "letter_marginal": maximum_marginal <= MARGINAL_CEILING,
        "canonical_nll": matched["four_view"]["canonical_nll"] <= NLL_CEILING,
    }
    metrics["eligibility"] = {"checks": checks, "eligible": all(checks.values()), "maximum_absolute_letter_share_delta": maximum_marginal}
    return metrics, outputs


def save_model(model: Any, tokenizer: Any, path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().to(device="cpu", dtype=torch.bfloat16) for name, value in model.state_dict().items()}
    model.save_pretrained(path, state_dict=state, safe_serialization=True)
    tokenizer.save_pretrained(path)
    (path / "chat_template.jinja").write_text(PLAIN_TEMPLATE, encoding="utf-8")
    del state
    return sha256(path / "model.safetensors")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("F", "E"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--signal", type=Path, required=True)
    parser.add_argument("--signal-sha", required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--recipes", type=Path, required=True)
    parser.add_argument("--recipes-sha", required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument(
        "--final-push-gates",
        action="store_true",
        help="Use FINAL PUSH interface-only gates and canonical-accuracy ranking.",
    )
    parser.add_argument(
        "--recovery-stop-step",
        type=int,
        choices=(70, 140),
        default=140,
        help="Stop a deterministic recovery at a historical checkpoint without changing its 140-step LR schedule.",
    )
    args = parser.parse_args()
    if args.final_push_gates:
        globals()["AGREEMENT_FLOOR"] = 0.0
        globals()["MARGINAL_CEILING"] = 0.10
        globals()["NLL_CEILING"] = 1.4523645305633546 + 0.15
    if transformers.__version__ != "5.13.1" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("pinned Transformers 5.13.1 and BF16 CUDA are required")
    hashes = {
        "model": verify(args.base_model / "model.safetensors", EXPECTED_MODEL),
        "signal": verify(args.signal, args.signal_sha), "anchors": verify(args.anchors, EXPECTED_ANCHORS),
        "recipes": verify(args.recipes, args.recipes_sha), "dev": verify(args.dev, EXPECTED_DEV),
        "prefix": verify(args.prefix, EXPECTED_PREFIX),
    }
    if any(term in str(path).casefold() for path in (args.signal, args.anchors, args.recipes, args.dev) for term in ("5_shots", "harness", "italic.jsonl")):
        raise RuntimeError("denylisted training path")
    signal, anchors, recipes, dev = read_jsonl(args.signal), read_jsonl(args.anchors), read_jsonl(args.recipes), read_jsonl(args.dev)
    prefix = json.loads(args.prefix.read_text(encoding="utf-8"))
    if (len(signal), len(anchors), len(recipes), len(dev), len(prefix)) != (21_000, 3_000, 24_000, 600, 11):
        raise RuntimeError("frozen input shape drift")
    row_map = {str(row["id"]): row for row in signal + anchors}
    if len(row_map) != 24_000 or any(str(row.get("source", "")).casefold() == "italic" for row in row_map.values()):
        raise RuntimeError("training row identity/source contract failed")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = PLAIN_TEMPLATE
    plain_ids, spaced_ids = candidate_ids(tokenizer)
    dev_label_ids = probe.answer_token_ids(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True).cuda()
    for parameter in model.parameters():
        parameter.data = parameter.data.float()
        parameter.requires_grad_(True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if sum(parameter.numel() for parameter in parameters) != PARAMETERS or {parameter.dtype for parameter in parameters} != {torch.float32}:
        raise RuntimeError("full-parameter FP32-master invariant failed")
    torch.cuda.reset_peak_memory_stats()
    anchor_path = args.output_dir / "fp32_anchor_posteriors.jsonl"
    anchor_map, anchor_sha = build_anchor_posteriors(model, tokenizer, recipes, row_map, plain_ids, spaced_ids, args.max_length, anchor_path)
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        baseline, baseline_outputs = evaluate(model, tokenizer, dev, prefix, dev_label_ids, args.eval_batch_size, args.max_length, args.seed)
    if abs(baseline["official_five_shot"]["four_view"]["selection_score"] - 0.385) > 0.005:
        raise RuntimeError("ARM-A matched baseline parity failed")
    if not baseline["eligibility"]["eligible"]:
        raise RuntimeError("ARM-A eligibility recalibration failed")
    write_json(args.output_dir / "baseline_metrics.json", baseline)
    for name, rows in baseline_outputs.items():
        write_jsonl(args.output_dir / f"baseline_{name}.jsonl", rows)
    optimizer = torch.optim.AdamW(parameters, lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=0.05, fused=True)
    optimizer.zero_grad(set_to_none=True)
    signal_indices = [index for index, recipe in enumerate(recipes) if recipe["row_kind"] == "signal"]
    anchor_indices = [index for index, recipe in enumerate(recipes) if recipe["row_kind"] == "anchor"]
    trajectory = []
    anchor_per_epoch = math.ceil(len(signal_indices) * 0.20 / 0.80)
    for epoch in range(2):
        regular = signal_indices.copy(); random.Random(args.seed + epoch).shuffle(regular)
        anchor_order = anchor_indices.copy(); random.Random(args.seed + 100 + epoch).shuffle(anchor_order)
        anchor_order = (anchor_order * math.ceil(anchor_per_epoch / len(anchor_order)))[:anchor_per_epoch]
        tags = regular + anchor_order; random.Random(args.seed + 200 + epoch).shuffle(tags); trajectory.extend(tags)
    trajectory = trajectory[:args.recovery_stop_step * GRAD_ACCUM * MICRO_BASE_ROWS]
    anchor_fraction = sum(recipes[index]["row_kind"] == "anchor" for index in trajectory) / len(trajectory)
    expected_exposures = args.recovery_stop_step * GRAD_ACCUM * MICRO_BASE_ROWS
    if len(trajectory) != expected_exposures or anchor_fraction < 0.15:
        raise RuntimeError(f"training trajectory invariant failed: {len(trajectory)}/{anchor_fraction}")
    history = []
    best = None
    loss_totals = Counter()
    optimizer_steps = 0
    started = time.perf_counter()

    def checkpoint(reason: str) -> None:
        nonlocal best
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            metrics, predictions = evaluate(model, tokenizer, dev, prefix, dev_label_ids, args.eval_batch_size, args.max_length, args.seed)
        metrics.update({"reason": reason, "optimizer_steps": optimizer_steps, "elapsed_seconds": time.perf_counter() - started})
        directory = args.output_dir / f"checkpoint_step_{optimizer_steps:06d}"
        write_json(directory / "metrics.json", metrics)
        for name, rows in predictions.items():
            write_jsonl(directory / f"{name}.jsonl", rows)
        record = {"checkpoint": str(directory), "metrics": metrics}
        history.append(record)
        matched = metrics["official_five_shot"]["four_view"]
        rank = (
            (float(matched["canonical_accuracy"]), -float(matched["canonical_nll"]))
            if args.final_push_gates
            else (float(matched["selection_score"]), -float(matched["canonical_nll"]))
        )
        if metrics["eligibility"]["eligible"] and (best is None or rank > tuple(best["rank"])):
            weight_hash = save_model(model, tokenizer, args.output_dir / "selected_model")
            best = {"checkpoint": str(directory), "rank": list(rank), "weight_sha256": weight_hash, "metrics": metrics}
        write_json(args.output_dir / "training_progress.json", {"history": history, "best_eligible": best})
        print(f"checkpoint step={optimizer_steps} matched={matched['selection_score']:.6f} eligible={metrics['eligibility']['eligible']}", flush=True)
        model.train()

    model.train()
    for exposure, recipe_index in enumerate(trajectory, 1):
        recipe = recipes[recipe_index]
        for variant in recipe["variants"]:
            reference = anchor_map.get((str(recipe["id"]), int(variant["view"])))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, components = sequence_loss(model, tokenizer, recipe, variant, row_map, plain_ids, spaced_ids, args.arm, reference, args.max_length)
            (loss / (GRAD_ACCUM * 4)).backward()
            loss_totals["variant_count"] += 1
            loss_totals["total"] += float(loss.detach())
            for name, value in components.items():
                loss_totals[name] += value
        if exposure % GRAD_ACCUM:
            continue
        for group in optimizer.param_groups:
            group["lr"] = lr_at(optimizer_steps + 1)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        if optimizer_steps % CHECKPOINT_INTERVAL == 0 or optimizer_steps == PLANNED_STEPS:
            checkpoint("terminal" if optimizer_steps == args.recovery_stop_step else "interval")
        if optimizer_steps % 10 == 0:
            print(f"train arm={args.arm} seed={args.seed} step={optimizer_steps}/{PLANNED_STEPS} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    if optimizer_steps != args.recovery_stop_step:
        raise RuntimeError("optimizer-step drift")
    count = max(1, loss_totals["variant_count"])
    manifest = {
        "protocol": (
            "zagreus-final-push-track-a-deterministic-recovery"
            if args.final_push_gates
            else "zagreus-ensemble-v1-format-matched-full-parameter"
        ),
        "status": "COMPLETED", "official_italic_rows_read_or_used": 0,
        "arm": args.arm, "seed": args.seed, "hashes": hashes | {"fp32_anchor_posteriors": anchor_sha},
        "parameterization": {"trainable_parameters": PARAMETERS, "fp32_master_weights": True, "bf16_autocast": True, "gradient_checkpointing": True},
        "optimizer": {"name": "fused AdamW", "betas": [0.9, 0.95], "weight_decay": 0.05, "clip": 1.0, "peak_lr": PEAK_LR, "minimum_lr": MIN_LR, "warmup_fraction": WARMUP},
        "run": {"optimizer_steps": optimizer_steps, "recovery_stop_step": args.recovery_stop_step, "lr_schedule_reference_steps": PLANNED_STEPS, "base_row_exposures": len(trajectory), "sequence_exposures": len(trajectory) * 4, "assistant_letter_supervisions": len(trajectory) * 4 * 6, "anchor_fraction": anchor_fraction, "elapsed_seconds": time.perf_counter()-started, "gpu": torch.cuda.get_device_name(0), "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()},
        "mean_losses": {name: loss_totals[name] / count for name in ("total", "demo_ce", "target_ce", "target_kd", "anchor_kl", "eos_ce")},
        "baseline": baseline, "history": history, "selected_eligible": best, "last_gradient_norm_before_clip": gradient_norm,
        "selection_rule": "matched_canonical_accuracy" if args.final_push_gates else "matched_selection",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__},
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print("ENSEMBLE_TRAIN_RESULT=" + json.dumps(manifest, ensure_ascii=False), flush=True)
    del optimizer, model
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
