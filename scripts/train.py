#!/usr/bin/env python3
"""FINAL PUSH Track C: online augmentation, sparse-head batched long training."""

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
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import zagreus_ensemble_t3_corpus as corpus
import zagreus_ensemble_train as legacy
import zagreus_format_probe as probe


EXPECTED_INIT = "209c861e792c812e98d4287ae5c3cd6eaa6161e38342432cd83868786ec697b6"
EXPECTED_SIGNAL = "648937a2eadcbd7a797cfe44fcf905bfb163a952ef1c24341f70484503c76cd5"
EXPECTED_ANCHORS = "913023f76d46f968010baff871204705a67c854884adf91a3b80e9029857559f"
EXPECTED_DEV = "3f943adb939e50d53ebf28eeef84244cb0d4fd4dd29f7af97af924a3e1612732"
EXPECTED_PREFIX = "6b04ad87a538509798e267ac84607474b880e8e243fb3b9bb0059e1eb9d0adb0"
PARAMETERS = 437_760_960
LABELS = "ABCDE"
PLAIN_TEMPLATE = legacy.PLAIN_TEMPLATE
PROFILE_COUNTS = tuple(count for count, amount in corpus.PROFILE_REFERENCE.items() for _ in range(amount))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"hash mismatch for {path}: {actual} != {expected}")
    return actual


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def top1(values: Sequence[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


class OnlineRenderer:
    def __init__(self, signal: list[dict[str, Any]], anchors: list[dict[str, Any]], seed: int):
        self.signal = signal
        self.anchors = anchors
        self.seed = seed
        self.row_map = {str(row["id"]): row for row in signal + anchors}
        self.authorized = [row for row in signal if row.get("gold_ce_authorized") and row.get("campaign_gold") is not None]
        if len(self.authorized) != 9713:
            raise RuntimeError(f"authorized demonstration-pool drift: {len(self.authorized)}")
        self.target_display = Counter()
        self.demo_display = Counter()

    def option_count(self, rng: random.Random) -> int:
        return int(PROFILE_COUNTS[rng.randrange(len(PROFILE_COUNTS))])

    @staticmethod
    def next_display(counters: Counter, count: int) -> int:
        display = int(counters[count] % count)
        counters[count] += 1
        return display

    def transformed(self, row: dict[str, Any], count: int, protected: set[int], salt: str):
        return corpus.transformed_item(row, count, protected, self.authorized, salt)

    def example(self, row: dict[str, Any], row_kind: str, epoch: int, serial: int) -> tuple[dict[str, Any], dict[str, Any]]:
        salt = f"online:{self.seed}:{epoch}:{serial}:{row['id']}"
        rng = random.Random(corpus.stable_int(salt))
        view = rng.randrange(4)
        count = self.option_count(rng)
        if row_kind == "signal":
            control = corpus.view_probabilities(row, view, True)
            authorized_gold = int(row["campaign_gold"]) if row.get("gold_ce_authorized") else None
            answer_base = authorized_gold if authorized_gold is not None else top1(control)
            protected = {top1(control)}
            if authorized_gold is not None:
                protected.add(authorized_gold)
        else:
            authorized_gold = int(row["campaign_gold"])
            answer_base = authorized_gold
            protected = {answer_base}
        options, base_indices, mapping = self.transformed(row, count, protected, salt + ":target")
        answer_semantic = mapping[answer_base]
        desired = self.next_display(self.target_display, count)
        order = corpus.placed_order(count, answer_semantic, desired, salt + ":target-order")
        target: dict[str, Any] = {
            "option_count": count,
            "options": options,
            "base_semantic_indices": base_indices,
            "semantic_order_by_display_slot": list(order),
            "answer_display": desired,
            "gold_ce_authorized": authorized_gold is not None,
        }
        if row_kind == "signal":
            target["control_probabilities"] = corpus.transformed_probabilities(control, base_indices)
        demos = []
        used_groups = {str(row["group_id"])}
        start = rng.randrange(len(self.authorized))
        cursor = 0
        while len(demos) < 5:
            demo = self.authorized[(start + cursor) % len(self.authorized)]
            cursor += 1
            group = str(demo["group_id"])
            if group in used_groups:
                continue
            used_groups.add(group)
            demo_count = self.option_count(rng)
            demo_gold = int(demo["campaign_gold"])
            demo_options, demo_base, demo_mapping = self.transformed(
                demo, demo_count, {demo_gold}, salt + f":demo:{len(demos)}"
            )
            demo_answer = demo_mapping[demo_gold]
            demo_display = self.next_display(self.demo_display, demo_count)
            demo_order = corpus.placed_order(
                demo_count, demo_answer, demo_display, salt + f":demo-order:{len(demos)}"
            )
            demos.append({
                "id": str(demo["id"]),
                "option_count": demo_count,
                "options": demo_options,
                "base_semantic_indices": demo_base,
                "semantic_order_by_display_slot": list(demo_order),
                "answer_display": demo_display,
            })
        recipe = {"id": str(row["id"]), "group_id": str(row["group_id"]), "row_kind": row_kind}
        variant = {"view": view, "prompt_messages": 12, "demos": demos, "target": target}
        return recipe, variant


def encode_example(
    tokenizer: Any, renderer: OnlineRenderer, recipe: dict[str, Any], variant: dict[str, Any], max_length: int
) -> dict[str, Any]:
    prompt, spans, target_label = legacy.render_variant(recipe, variant, renderer.row_map)
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids = list(map(int, encoded["input_ids"]))
    offsets = encoded["offset_mapping"]
    demo_positions = []
    for start, end in spans:
        matching = [index for index, (left, right) in enumerate(offsets) if left < end and right > start]
        if len(matching) != 1:
            raise RuntimeError("demonstration answer tokenization drift")
        demo_positions.append(matching[0])
    target_ids = tokenizer.encode(" " + target_label, add_special_tokens=False)
    if len(target_ids) != 1:
        raise RuntimeError("target completion tokenization drift")
    target_position = len(prompt_ids)
    ids = prompt_ids + target_ids + [int(tokenizer.eos_token_id)]
    if len(ids) > max_length:
        raise RuntimeError(f"online sequence exceeds max length: {len(ids)}")
    return {
        "ids": ids,
        "positions": [*demo_positions, target_position],
        "recipe": recipe,
        "variant": variant,
    }


def collate(examples: Sequence[dict[str, Any]], pad_id: int, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = max(len(example["ids"]) for example in examples)
    input_ids = torch.full((len(examples), maximum), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    gather = torch.empty((len(examples), 7), dtype=torch.long, device=device)
    for index, example in enumerate(examples):
        ids = torch.tensor(example["ids"], dtype=torch.long, device=device)
        pad = maximum - len(ids)
        input_ids[index, pad:] = ids
        attention[index, pad:] = 1
        positions = example["positions"]
        gather[index] = torch.tensor([pad + position - 1 for position in positions] + [pad + positions[-1]], device=device)
    return input_ids, attention, gather


def sparse_logits(model: Any, input_ids: torch.Tensor, attention: torch.Tensor, gather: torch.Tensor) -> torch.Tensor:
    hidden = model.model(input_ids=input_ids, attention_mask=attention, use_cache=False, return_dict=True).last_hidden_state
    selected = hidden.gather(1, gather[:, :, None].expand(-1, -1, hidden.shape[-1]))
    return model.lm_head(selected).float()


def semantic_logits(display_logits: torch.Tensor, order: Sequence[int]) -> torch.Tensor:
    semantic = torch.empty_like(display_logits)
    for display, semantic_index in enumerate(order):
        semantic[int(semantic_index)] = display_logits[display]
    return semantic


def batch_loss(
    model: Any,
    reference: Any,
    examples: Sequence[dict[str, Any]],
    tokenizer: Any,
    plain_ids: torch.Tensor,
    spaced_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float], int]:
    input_ids, attention, gather = collate(examples, int(tokenizer.pad_token_id))
    logits = sparse_logits(model, input_ids, attention, gather)
    anchor_indices = [index for index, example in enumerate(examples) if example["recipe"]["row_kind"] == "anchor"]
    reference_logits = None
    if anchor_indices:
        subset = torch.tensor(anchor_indices, device="cuda")
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            reference_logits = sparse_logits(
                reference,
                input_ids.index_select(0, subset),
                attention.index_select(0, subset),
                gather.index_select(0, subset),
            )
    anchor_cursor = 0
    losses = []
    totals = Counter()
    for batch_index, example in enumerate(examples):
        variant = example["variant"]
        turns = []
        demo_losses = []
        for turn, demo in enumerate(variant["demos"]):
            count = int(demo["option_count"])
            values = logits[batch_index, turn].index_select(0, plain_ids[:count])
            turns.append(values)
            demo_losses.append(F.cross_entropy(
                values[None, :], torch.tensor([int(demo["answer_display"])], device="cuda")
            ))
        target = variant["target"]
        count = int(target["option_count"])
        displayed = logits[batch_index, 5].index_select(0, spaced_ids[:count])
        semantic = semantic_logits(displayed, target["semantic_order_by_display_slot"])
        turns.append(semantic)
        eos = F.cross_entropy(
            logits[batch_index, 6][None, :], torch.tensor([int(tokenizer.eos_token_id)], device="cuda")
        )
        if example["recipe"]["row_kind"] == "anchor":
            assert reference_logits is not None
            ref = reference_logits[anchor_cursor]
            anchor_cursor += 1
            terms = []
            for turn, item in enumerate(variant["demos"]):
                turn_count = int(item["option_count"])
                ref_values = ref[turn].index_select(0, plain_ids[:turn_count])
                ref_prob = F.softmax(ref_values, dim=-1)
                terms.append(torch.sum(ref_prob * (F.log_softmax(ref_values, dim=-1) - F.log_softmax(turns[turn], dim=-1))))
            ref_displayed = ref[5].index_select(0, spaced_ids[:count])
            ref_semantic = semantic_logits(ref_displayed, target["semantic_order_by_display_slot"])
            ref_prob = F.softmax(ref_semantic, dim=-1)
            terms.append(torch.sum(ref_prob * (F.log_softmax(ref_semantic, dim=-1) - F.log_softmax(semantic, dim=-1))))
            anchor_kl = torch.stack(terms).mean()
            item_loss = 2.0 * anchor_kl
            totals["anchor_kl"] += float(anchor_kl.detach())
        else:
            demo_ce = torch.stack(demo_losses).mean()
            teacher = torch.tensor(target["control_probabilities"], device="cuda", dtype=torch.float32)
            teacher_t2 = F.softmax(torch.log(teacher.clamp_min(1e-12)) / 2.0, dim=-1)
            kd = torch.sum(teacher_t2 * (torch.log(teacher_t2.clamp_min(1e-12)) - F.log_softmax(semantic / 2.0, dim=-1))) * 4.0
            if target["gold_ce_authorized"]:
                gold_semantic = int(target["semantic_order_by_display_slot"][int(target["answer_display"])])
                ce = F.cross_entropy(semantic[None, :], torch.tensor([gold_semantic], device="cuda"))
                target_loss = 0.5 * ce + 0.5 * kd
                totals["target_ce"] += float(ce.detach())
            else:
                target_loss = kd
            item_loss = (5.0 * demo_ce + target_loss) / 6.0 + 0.1 * eos
            totals["demo_ce"] += float(demo_ce.detach())
            totals["target_kd"] += float(kd.detach())
        totals["eos_ce"] += float(eos.detach())
        losses.append(item_loss)
    return torch.stack(losses).mean(), dict(totals), int(attention.sum().item())


def lr_at(step: int, total_steps: int) -> float:
    progress = step / total_steps
    if progress <= 0.05:
        return 1.5e-4 * progress / 0.05
    cosine = (progress - 0.05) / 0.95
    return 1.5e-5 + (1.5e-4 - 1.5e-5) * 0.5 * (1.0 + math.cos(math.pi * cosine))


def epoch_tags(signal_count: int, anchor_count: int, seed: int, epoch: int) -> list[tuple[str, int]]:
    anchor_exposures = math.ceil(signal_count * 0.20 / 0.80)
    tags = [("signal", index) for index in range(signal_count)]
    anchor_order = list(range(anchor_count))
    random.Random(seed + 1000 + epoch).shuffle(anchor_order)
    anchor_order = (anchor_order * math.ceil(anchor_exposures / anchor_count))[:anchor_exposures]
    tags.extend(("anchor", index) for index in anchor_order)
    random.Random(seed + epoch).shuffle(tags)
    return tags


def batches_for_epoch(
    renderer: OnlineRenderer,
    tokenizer: Any,
    epoch: int,
    batch_size: int,
    max_length: int,
    pool_multiple: int,
) -> Iterable[list[dict[str, Any]]]:
    tags = epoch_tags(len(renderer.signal), len(renderer.anchors), renderer.seed, epoch)
    serial = epoch * len(tags)
    pool_size = batch_size * pool_multiple
    for start in range(0, len(tags), pool_size):
        pool = []
        for offset, (kind, index) in enumerate(tags[start:start + pool_size]):
            row = renderer.signal[index] if kind == "signal" else renderer.anchors[index]
            recipe, variant = renderer.example(row, kind, epoch, serial + start + offset)
            pool.append(encode_example(tokenizer, renderer, recipe, variant, max_length))
        pool.sort(key=lambda example: len(example["ids"]))
        mini_batches = [pool[index:index + batch_size] for index in range(0, len(pool), batch_size)]
        random.Random(renderer.seed + 5000 + epoch + start).shuffle(mini_batches)
        yield from mini_batches


def token_budget_microbatches(examples: Sequence[dict[str, Any]], max_padded_tokens: int) -> list[list[dict[str, Any]]]:
    """Split one effective batch without changing its optimizer-step weight."""
    result = []
    cursor = 0
    while cursor < len(examples):
        size = len(examples) - cursor
        while size > 1:
            maximum = max(len(example["ids"]) for example in examples[cursor:cursor + size])
            if maximum * size <= max_padded_tokens:
                break
            size -= 1
        result.append(list(examples[cursor:cursor + size]))
        cursor += size
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--signal", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--dev", type=Path)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--pool-multiple", type=int, default=8)
    parser.add_argument("--max-padded-tokens", type=int, default=16000)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--skip-eval-smoke", action="store_true")
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Train and save the terminal checkpoint without loading selection data.",
    )
    args = parser.parse_args()
    if transformers.__version__ != "5.13.1" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("pinned Transformers 5.13.1 and BF16 CUDA required")
    hashes: dict[str, str] = {
        "init": verify(args.base_model / "model.safetensors", EXPECTED_INIT),
        "signal": verify(args.signal, EXPECTED_SIGNAL),
        "anchors": verify(args.anchors, EXPECTED_ANCHORS),
    }
    if args.skip_eval:
        hashes.update({"dev": "not_loaded", "prefix": "not_loaded"})
    else:
        if args.dev is None or args.prefix is None:
            raise RuntimeError("--dev and --prefix are required unless --skip-eval is set")
        hashes.update({"dev": verify(args.dev, EXPECTED_DEV), "prefix": verify(args.prefix, EXPECTED_PREFIX)})
    screened_paths = [args.signal, args.anchors]
    if args.dev is not None:
        screened_paths.append(args.dev)
    if any(term in str(path).casefold() for path in screened_paths for term in ("5_shots", "harness", "italic.jsonl")):
        raise RuntimeError("denylisted training path")
    signal = legacy.read_jsonl(args.signal)
    anchors = legacy.read_jsonl(args.anchors)
    dev = [] if args.skip_eval else legacy.read_jsonl(args.dev)
    prefix = [] if args.skip_eval else json.loads(args.prefix.read_text())
    if (len(signal), len(anchors)) != (21000, 3000):
        raise RuntimeError("input shape drift")
    if not args.skip_eval and (len(dev), len(prefix)) != (600, 11):
        raise RuntimeError("selection input shape drift")
    if any(str(row.get("source", "")).casefold() == "italic" for row in signal + anchors):
        raise RuntimeError("official row reached Track C")
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = PLAIN_TEMPLATE
    plain_ids, spaced_ids = legacy.candidate_ids(tokenizer)
    dev_label_ids = None if args.skip_eval else probe.answer_token_ids(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True).cuda()
    for parameter in model.parameters():
        parameter.data = parameter.data.float(); parameter.requires_grad_(True)
    model.config.use_cache = False
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if sum(parameter.numel() for parameter in parameters) != PARAMETERS or {parameter.dtype for parameter in parameters} != {torch.float32}:
        raise RuntimeError("FP32-master invariant failed")
    reference = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True).cuda().eval()
    reference.config.use_cache = False
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    renderer = OnlineRenderer(signal, anchors, args.seed)
    steps_per_epoch = math.ceil((len(signal) + math.ceil(len(signal) * 0.20 / 0.80)) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    if args.smoke_steps:
        total_steps = min(total_steps, args.smoke_steps)
    legacy.AGREEMENT_FLOOR = 0.0
    legacy.MARGINAL_CEILING = 0.10
    legacy.NLL_CEILING = 1.4523645305633546 + 0.15
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_eval:
        baseline = {}
    elif args.skip_eval_smoke:
        if not args.smoke_steps:
            raise RuntimeError("--skip-eval-smoke is permitted only with --smoke-steps")
        baseline = {}
    else:
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            baseline, baseline_outputs = legacy.evaluate(model, tokenizer, dev, prefix, dev_label_ids, args.eval_batch_size, args.max_length, args.seed)
        write_json(args.output_dir / "baseline_metrics.json", baseline)
        for name, rows in baseline_outputs.items():
            legacy.write_jsonl(args.output_dir / f"baseline_{name}.jsonl", rows)
    optimizer = torch.optim.AdamW(parameters, lr=1.5e-4, betas=(0.9, 0.95), weight_decay=0.05, fused=True)
    optimizer.zero_grad(set_to_none=True)
    history = []
    best = None
    terminal = None
    losses = Counter()
    tokens = 0
    step = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def checkpoint(reason: str) -> None:
        nonlocal best, terminal
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            metrics, predictions = legacy.evaluate(model, tokenizer, dev, prefix, dev_label_ids, args.eval_batch_size, args.max_length, args.seed)
        metrics.update({"optimizer_steps": step, "reason": reason, "elapsed_seconds": time.perf_counter() - started})
        directory = args.output_dir / f"checkpoint_step_{step:06d}"
        if reason == "terminal":
            terminal_weight_hash = legacy.save_model(model, tokenizer, args.output_dir / "terminal_model")
            metrics["terminal_weight_sha256"] = terminal_weight_hash
            terminal = {
                "checkpoint": str(directory),
                "model_path": str(args.output_dir / "terminal_model"),
                "weight_sha256": terminal_weight_hash,
            }
        write_json(directory / "metrics.json", metrics)
        for name, rows in predictions.items():
            legacy.write_jsonl(directory / f"{name}.jsonl", rows)
        canonical = float(metrics["official_five_shot"]["four_view"]["canonical_accuracy"])
        record = {"checkpoint": str(directory), "metrics": metrics}
        history.append(record)
        if metrics["eligibility"]["eligible"] and (best is None or canonical > best["canonical_accuracy"]):
            weight_hash = legacy.save_model(model, tokenizer, args.output_dir / "selected_model")
            best = {"checkpoint": str(directory), "canonical_accuracy": canonical, "weight_sha256": weight_hash, "metrics": metrics}
        write_json(args.output_dir / "training_progress.json", {"history": history, "best_eligible": best})
        print(f"checkpoint step={step} canonical={canonical:.6f} eligible={metrics['eligibility']['eligible']}", flush=True)
        model.train()

    model.train()
    stop = False
    for epoch in range(args.epochs):
        for batch in batches_for_epoch(renderer, tokenizer, epoch, args.batch_size, args.max_length, args.pool_multiple):
            batch_started = time.perf_counter()
            components = Counter()
            batch_tokens = 0
            batch_loss_value = 0.0
            for microbatch in token_budget_microbatches(batch, args.max_padded_tokens):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, micro_components, micro_tokens = batch_loss(
                        model, reference, microbatch, tokenizer, plain_ids, spaced_ids
                    )
                weight = len(microbatch) / len(batch)
                (loss * weight).backward()
                batch_loss_value += float(loss.detach()) * weight
                batch_tokens += micro_tokens
                for name, value in micro_components.items():
                    components[name] += value
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step + 1, total_steps)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            step += 1; tokens += batch_tokens
            losses["total"] += batch_loss_value; losses["batches"] += 1
            for name, value in components.items():
                losses[name] += value
            if step % 25 == 0:
                elapsed = time.perf_counter() - started
                print(f"train seed={args.seed} epoch={epoch} step={step}/{total_steps} tok_s={tokens/elapsed:.1f} batch_s={time.perf_counter()-batch_started:.3f}", flush=True)
            if not args.skip_eval and not args.skip_eval_smoke and (step % args.checkpoint_interval == 0 or step == total_steps):
                checkpoint("terminal" if step == total_steps else "interval")
            if step >= total_steps:
                stop = True; break
        if stop:
            break
    if step != total_steps:
        raise RuntimeError(f"optimizer-step drift: {step} != {total_steps}")
    if args.skip_eval:
        terminal_weight_hash = legacy.save_model(model, tokenizer, args.output_dir / "terminal_model")
        terminal = {
            "checkpoint": "terminal_without_selection_eval",
            "model_path": str(args.output_dir / "terminal_model"),
            "weight_sha256": terminal_weight_hash,
        }
    count = max(1, int(losses["batches"]))
    manifest = {
        "protocol": "zagreus-final-push-track-c-online-augmentation",
        "status": "COMPLETED",
        "official_italic_rows_read_or_used": 0,
        "seed": args.seed,
        "hashes": hashes,
        "selection_rule": "terminal_checkpoint" if args.skip_eval else "matched_instrument_canonical_accuracy_only",
        "gates": {"interface_only": True, "nll_divergence_ceiling": 1.6023645305633545},
        "augmentation": {"online_demo_sampling": True, "online_option_permutation": True, "training_pool_demos_only": True, "all_six_turns_supervised": True, "dynamic_padding": True, "length_bucket_pool_multiple": args.pool_multiple, "effective_batch_size": args.batch_size, "microbatch_max_padded_tokens": args.max_padded_tokens},
        "optimizer": {"name": "fused AdamW", "peak_lr": 1.5e-4, "minimum_lr": 1.5e-5, "warmup_fraction": 0.05, "cosine_total_steps": total_steps, "weight_decay": 0.05, "clip": 1.0},
        "run": {"epochs": args.epochs, "batch_size": args.batch_size, "optimizer_steps": step, "tokens": tokens, "tokens_per_second": tokens / (time.perf_counter() - started), "anchor_fraction": 0.20, "elapsed_seconds": time.perf_counter() - started, "gpu": torch.cuda.get_device_name(0), "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()},
        "parameterization": {"trainable_parameters": PARAMETERS, "fp32_master_weights": True, "bf16_autocast": True, "gradient_checkpointing": False, "sparse_lm_head_after_hidden_gather": True},
        "mean_batch_losses": {name: losses[name] / count for name in ("total", "demo_ce", "target_ce", "target_kd", "anchor_kl", "eos_ce")},
        "baseline": baseline,
        "history": history,
        "selected_eligible": best,
        "terminal_model": terminal,
        "last_gradient_norm_before_clip": gradient_norm,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__},
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print("FINAL_PUSH_TRAIN_RESULT=" + json.dumps(manifest), flush=True)
    del optimizer, reference, model
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
