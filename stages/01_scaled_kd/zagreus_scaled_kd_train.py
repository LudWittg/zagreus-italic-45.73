"""Time-budgeted all-layer rsLoRA training with permutation-ensemble soft KD."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import itertools
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import zagreus_interface_calibration as interface


MODEL_ID = "mii-llm/zagreus-0.4B-ita"
EXPECTED_BASE_WEIGHT_SHA256 = (
    "edb113db34469879f54db04ec7a31a0040ef1258690539f70577277ec3ba0ff9"
)
LABELS = "ABCD"
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
EXPECTED_TRAINABLE_PARAMETERS = 34_734_080
P2C_BOOTSTRAP_SEED = 20260803
P2C_BOOTSTRAP_SAMPLES = 10_000
P2C_BASELINE_FOUR_VIEW_MACRO = 0.34625
P2C_BASELINE_CANONICAL_NLL = 1.9136
P2C_BASELINE_AGREEMENT = 0.628
P2C_AGREEMENT_TOLERANCE = 0.02
P2F_ANCHOR_BETA = 2.0
SYSTEM_PROMPT = (
    "Rispondi alla domanda a scelta multipla scrivendo soltanto la lettera "
    "dell'opzione corretta, senza spiegazioni."
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


def sha256_file(path: Path) -> str:
    return interface.sha256_file(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_clean(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if "italic" in path.name.casefold():
        raise ValueError(f"Official-looking data path is forbidden: {path}")
    for row in rows:
        if str(row.get("id", "")).casefold().startswith("italic:"):
            raise ValueError(f"Official-looking row is forbidden: {row['id']}")
        if str(row.get("source", "")).casefold() == "italic":
            raise ValueError(f"Official source is forbidden: {row['id']}")
        ensemble = row.get("teacher", {}).get("ensemble", {})
        probabilities = ensemble.get("probabilities")
        if not isinstance(probabilities, list) or len(probabilities) != 4:
            raise ValueError(f"Missing teacher probabilities: {row['id']}")
        if abs(sum(map(float, probabilities)) - 1.0) > 1e-5:
            raise ValueError(f"Teacher probabilities are not normalized: {row['id']}")
        permutations = row.get("teacher", {}).get("permutations")
        if not isinstance(permutations, list) or len(permutations) != 4:
            raise ValueError(f"Missing four teacher permutation views: {row['id']}")
        for expected_view, view in enumerate(permutations):
            if int(view.get("view", -1)) != expected_view:
                raise ValueError(f"Teacher view order mismatch: {row['id']}")
            order = view.get("semantic_order_by_display_slot")
            log_probabilities = view.get("semantic_log_probabilities")
            if sorted(map(int, order or [])) != list(range(4)):
                raise ValueError(f"Invalid teacher display order: {row['id']} view {expected_view}")
            if not isinstance(log_probabilities, list) or len(log_probabilities) != 4:
                raise ValueError(f"Missing teacher view posterior: {row['id']} view {expected_view}")


def question_text(row: dict[str, Any]) -> str:
    question = str(row["question"]).strip()
    if row.get("context"):
        question = f"{str(row['context']).strip()}\n\n{question}"
    return question


def render_prompt(row: dict[str, Any], order: Sequence[int]) -> str:
    options = "\n".join(
        f"{LABELS[display]}) {row['options'][semantic]}"
        for display, semantic in enumerate(order)
    )
    topic = str(row.get("domain") or "lingua_italiana")
    return (
        "Rispondi alla seguente domanda a scelta multipla sull'argomento "
        f"'{topic}'. La tua risposta deve essere nel seguente formato: 'LETTERA' "
        "(senza virgolette) dove LETTERA è una tra ABCD. Scrivi solo la lettera "
        "corrispondente alla tua risposta senza spiegazioni.\n\n"
        f"{question_text(row)}\n\n{options}\n\nRisposta:"
    )


def non_identity_order(row_id: str, step: int, count: int = 4) -> tuple[int, ...]:
    identity = tuple(range(count))
    attempt = 0
    while True:
        values = list(identity)
        payload = interface.stable_rng(20260801, "kd-view", row_id, step, attempt)
        payload.shuffle(values)
        order = tuple(values)
        if order != identity:
            return order
        attempt += 1


def answer_token_ids(tokenizer: Any, device: torch.device | str | None = None) -> torch.Tensor:
    values = []
    for label in LABELS:
        ids = tokenizer(f" {label}", add_special_tokens=False).input_ids
        if len(ids) != 1 or tokenizer.decode(ids).strip() != label:
            raise ValueError(f"Student label is not one leading-space token: {label} -> {ids}")
        values.append(int(ids[0]))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(values, device=device)


def p2b_generation_examples(
    rows: Sequence[dict[str, Any]], count: int, seed: int, split: str
) -> list[dict[str, Any]]:
    """Build balanced four-option format probes from the actual P2B dev rows."""
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{split}:{row['id']}".encode("utf-8")
        ).hexdigest(),
    )[:count]
    examples = []
    for index, row in enumerate(ordered):
        gold_semantic = int(row["gold"])
        gold_display = index % 4
        remaining = [value for value in range(4) if value != gold_semantic]
        rng = interface.stable_rng(seed, "p2b-generation-order", split, str(row["id"]))
        rng.shuffle(remaining)
        order: list[int | None] = [None] * 4
        order[gold_display] = gold_semantic
        cursor = 0
        for display in range(4):
            if order[display] is None:
                order[display] = remaining[cursor]
                cursor += 1
        semantic_order = tuple(int(value) for value in order)
        examples.append(
            {
                "id": f"p2b-format:{split}:{row['id']}",
                "source_id": str(row["id"]),
                "group_id": row.get("group_id"),
                "domain": row.get("domain"),
                "allowed": LABELS,
                "gold_label": LABELS[gold_display],
                "shot_count": 0,
                "prompt": render_prompt(row, semantic_order),
            }
        )
    return examples


def encode_prompts(tokenizer: Any, prompts: Sequence[str], max_length: int) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(prompts),
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.cuda(non_blocking=True) for key, value in encoded.items()}


def display_to_semantic(display_logits: torch.Tensor, orders: Sequence[Sequence[int]]) -> torch.Tensor:
    result = torch.empty_like(display_logits)
    for batch_index, order in enumerate(orders):
        for display, semantic in enumerate(order):
            result[batch_index, semantic] = display_logits[batch_index, display]
    return result


def student_semantic_logits(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    orders: Sequence[Sequence[int]],
    label_ids: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    prompts = [render_prompt(row, order) for row, order in zip(rows, orders)]
    batch = encode_prompts(tokenizer, prompts, max_length)
    outputs = model(
        **batch,
        use_cache=False,
        return_dict=True,
        logits_to_keep=1,
    )
    displayed = outputs.logits[:, -1, :].float().index_select(-1, label_ids)
    return display_to_semantic(displayed, orders)


def js_per_row(first_logits: torch.Tensor, second_logits: torch.Tensor) -> torch.Tensor:
    first_log = F.log_softmax(first_logits, dim=-1)
    second_log = F.log_softmax(second_logits, dim=-1)
    first = first_log.exp()
    second = second_log.exp()
    mixture = 0.5 * (first + second)
    mixture_log = torch.log(mixture.clamp_min(1e-12))
    return 0.5 * (
        torch.sum(first * (first_log - mixture_log), dim=-1)
        + torch.sum(second * (second_log - mixture_log), dim=-1)
    )


def p2c_lambda(optimizer_step: int, arm: str) -> float:
    """Return the prospectively frozen P2C consistency weight for a 1-based step."""
    if arm in {"4v", "anchor"} or optimizer_step <= 100:
        return 0.0
    if optimizer_step >= 140:
        return 0.5
    return 0.5 * (optimizer_step - 100) / 40.0


def p2c_orders(rows: Sequence[dict[str, Any]]) -> list[list[tuple[int, ...]]]:
    """Use the exact four orders whose pinned E4B posteriors are preserved."""
    return [
        [
            tuple(map(int, view["semantic_order_by_display_slot"]))
            for view in row["teacher"]["permutations"]
        ]
        for row in rows
    ]


def is_anchor_replay(row: dict[str, Any]) -> bool:
    """Prefer the explicit FINAL-v1 marker while preserving legacy P2F data."""
    if "anchor_replay" in row:
        return bool(row["anchor_replay"])
    return str(row.get("source_split")) == "unassigned"


def p2c_loss(
    view_logits: Sequence[torch.Tensor],
    rows: Sequence[dict[str, Any]],
    temperature: float,
    arm: str,
    optimizer_step: int,
    anchor_beta: float = P2F_ANCHOR_BETA,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Four-view CE/KD, optionally with ensemble KD or the frozen P2F anchor."""
    if len(view_logits) != 4:
        raise ValueError(f"P2C requires exactly four student views, found {len(view_logits)}")
    device = view_logits[0].device
    gold = torch.tensor([int(row["gold"]) for row in rows], device=device)
    eligible = torch.tensor(
        [bool(row["teacher"]["ensemble"]["eligible"]) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    gold_by_view = []
    teacher_kl_by_view = []
    replay_anchor_kl_by_view = []
    student_probabilities_t = []
    student_log_probabilities_t = []
    for view_index, logits in enumerate(view_logits):
        gold_by_view.append(F.cross_entropy(logits, gold, reduction="none"))
        teacher_log = torch.tensor(
            [
                row["teacher"]["permutations"][view_index]["semantic_log_probabilities"]
                for row in rows
            ],
            dtype=torch.float32,
            device=device,
        )
        teacher_t = F.softmax(teacher_log / temperature, dim=-1)
        student_log_t = F.log_softmax(logits / temperature, dim=-1)
        student_log_probabilities_t.append(student_log_t)
        student_probabilities_t.append(student_log_t.exp())
        teacher_kl_by_view.append(
            torch.sum(
                teacher_t
                * (torch.log(teacher_t.clamp_min(1e-12)) - student_log_t),
                dim=-1,
            )
            * (temperature**2)
        )
        if arm == "anchor":
            anchor = torch.tensor(
                [
                    (
                        row["p2f_anchor"]["permutations"][view_index][
                            "semantic_probabilities"
                        ]
                        if is_anchor_replay(row)
                        else [0.25, 0.25, 0.25, 0.25]
                    )
                    for row in rows
                ],
                dtype=torch.float32,
                device=device,
            )
            student_log_t1 = F.log_softmax(logits, dim=-1)
            replay_anchor_kl_by_view.append(
                torch.sum(
                    anchor
                    * (torch.log(anchor.clamp_min(1e-12)) - student_log_t1),
                    dim=-1,
                )
            )
    gold_values = torch.stack(gold_by_view).mean(0)
    teacher_kl = torch.stack(teacher_kl_by_view).mean(0)
    eligible_loss = 0.40 * gold_values + 0.50 * teacher_kl
    fallback_loss = 0.90 * gold_values
    base_per_row = eligible * eligible_loss + (1.0 - eligible) * fallback_loss

    replay_anchor_kl = torch.zeros_like(gold_values)
    replay = torch.zeros_like(gold_values)
    if arm == "anchor":
        replay = torch.tensor(
            [float(is_anchor_replay(row)) for row in rows],
            dtype=torch.float32,
            device=device,
        )
        replay_anchor_kl = torch.stack(replay_anchor_kl_by_view).mean(0)
        replay_loss = gold_values + anchor_beta * replay_anchor_kl
        base_per_row = replay * replay_loss + (1.0 - replay) * base_per_row

    ensemble_t = torch.stack(student_probabilities_t).mean(0).detach()
    ensemble_log_t = torch.log(ensemble_t.clamp_min(1e-12))
    consistency_by_view = [
        torch.sum(ensemble_t * (ensemble_log_t - student_log_t), dim=-1)
        * (temperature**2)
        for student_log_t in student_log_probabilities_t
    ]
    consistency = torch.stack(consistency_by_view).mean(0)
    consistency_weight = p2c_lambda(optimizer_step, arm)
    per_row = (1.0 - consistency_weight) * base_per_row + consistency_weight * consistency
    return per_row.mean(), {
        "total": float(per_row.mean().detach()),
        "base_total": float(base_per_row.mean().detach()),
        "gold_ce": float(gold_values.mean().detach()),
        "teacher_kl_t2": float(teacher_kl.mean().detach()),
        "ensemble_kl_t2": float(consistency.mean().detach()),
        "replay_anchor_kl_t1": float(
            (replay_anchor_kl * replay).sum().detach() / replay.sum().clamp_min(1.0)
        ),
        "replay_rate": float(replay.mean().detach()),
        "consistency_weight": consistency_weight,
        "eligible_rate": float(eligible.mean().detach()),
    }


def kd_loss(
    canonical: torch.Tensor,
    permuted: torch.Tensor,
    rows: Sequence[dict[str, Any]],
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = canonical.device
    gold = torch.tensor([int(row["gold"]) for row in rows], device=device)
    teacher = torch.tensor(
        [row["teacher"]["ensemble"]["probabilities"] for row in rows],
        dtype=torch.float32,
        device=device,
    )
    eligible = torch.tensor(
        [bool(row["teacher"]["ensemble"]["eligible"]) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    gold_values = F.cross_entropy(canonical, gold, reduction="none")
    teacher_t = F.softmax(torch.log(teacher.clamp_min(1e-12)) / temperature, dim=-1)
    student_log_t = F.log_softmax(canonical / temperature, dim=-1)
    teacher_kl = torch.sum(
        teacher_t * (torch.log(teacher_t.clamp_min(1e-12)) - student_log_t), dim=-1
    ) * (temperature**2)
    consistency = js_per_row(canonical, permuted)
    eligible_loss = 0.40 * gold_values + 0.50 * teacher_kl + 0.10 * consistency
    fallback_loss = 0.90 * gold_values + 0.10 * consistency
    per_row = eligible * eligible_loss + (1.0 - eligible) * fallback_loss
    return per_row.mean(), {
        "total": float(per_row.mean().detach()),
        "gold_ce": float(gold_values.mean().detach()),
        "teacher_kl_t2": float(teacher_kl.mean().detach()),
        "consistency_js": float(consistency.mean().detach()),
        "eligible_rate": float(eligible.mean().detach()),
    }


def macro_accuracy(rows: Sequence[dict[str, Any]]) -> float:
    by_domain: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["domain"])].append(bool(row["correct"]))
    return float(np.mean([np.mean(values) for values in by_domain.values()]))


@torch.no_grad()
def evaluate_view(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    label_ids: torch.Tensor,
    batch_size: int,
    max_length: int,
    view: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    predictions = []
    total_nll = 0.0
    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset : offset + batch_size]
        orders = [
            tuple(range(4))
            if view == 0
            else non_identity_order(str(row["id"]), 10_000_000 + view)
            for row in batch_rows
        ]
        logits = student_semantic_logits(
            model, tokenizer, batch_rows, orders, label_ids, max_length
        )
        log_probs = F.log_softmax(logits, dim=-1)
        predicted = logits.argmax(-1).cpu().tolist()
        gold = torch.tensor([int(row["gold"]) for row in batch_rows], device="cuda")
        total_nll += float(F.nll_loss(log_probs, gold, reduction="sum"))
        probabilities = log_probs.exp().cpu().tolist()
        for row, pred, probs in zip(batch_rows, predicted, probabilities):
            predictions.append(
                {
                    "id": row["id"],
                    "domain": row["domain"],
                    "gold": int(row["gold"]),
                    "prediction": int(pred),
                    "correct": int(pred) == int(row["gold"]),
                    "probabilities": probs,
                    "view": view,
                }
            )
    accuracy = sum(row["correct"] for row in predictions) / len(predictions)
    by_domain = {
        domain: {
            "n": len(domain_rows),
            "accuracy": sum(row["correct"] for row in domain_rows) / len(domain_rows),
        }
        for domain in sorted({str(row["domain"]) for row in predictions})
        for domain_rows in [[row for row in predictions if row["domain"] == domain]]
    }
    return {
        "n": len(predictions),
        "accuracy": accuracy,
        "macro_accuracy": macro_accuracy(predictions),
        "nll": total_nll / len(predictions),
        "by_domain": by_domain,
    }, predictions


@torch.no_grad()
def evaluate_model(
    model: Any,
    tokenizer: Any,
    dev_rows: Sequence[dict[str, Any]],
    label_ids: torch.Tensor,
    batch_size: int,
    max_length: int,
    generation_examples: Sequence[dict[str, Any]],
    generation_batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    views = []
    all_predictions = []
    for view in range(4):
        metrics, predictions = evaluate_view(
            model,
            tokenizer,
            dev_rows,
            label_ids,
            batch_size,
            max_length,
            view,
        )
        views.append(metrics)
        all_predictions.extend(predictions)
    interface_ids = {label: int(value) for label, value in zip(LABELS, label_ids.tolist())}
    generation, generation_rows = interface.generation_metrics(
        model,
        tokenizer,
        generation_examples,
        interface_ids,
        generation_batch_size,
        max_length,
    )
    by_id: dict[str, list[int]] = defaultdict(list)
    prediction_rows_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_predictions:
        by_id[str(row["id"])].append(int(row["prediction"]))
        prediction_rows_by_id[str(row["id"])].append(row)
    pair_agreements = []
    for row_id, values in by_id.items():
        if len(values) != 4:
            raise RuntimeError(f"Expected four prediction views for {row_id}, found {len(values)}")
        pairs = list(itertools.combinations(values, 2))
        pair_agreements.append(sum(left == right for left, right in pairs) / len(pairs))
    generated_counts = generation["predicted_counts"]
    generated_n = int(generation["n"])
    letter_shares = {
        label: int(generated_counts.get(label, 0)) / generated_n for label in LABELS
    }
    diagnostics = {
        "letter_shares": letter_shares,
        "maximum_absolute_letter_share_delta_from_uniform": max(
            abs(value - 0.25) for value in letter_shares.values()
        ),
        "non_letter_emission_rate": 1.0 - float(generation["valid_rate"]),
        "four_view_pairwise_prediction_agreement": float(np.mean(pair_agreements)),
        "definition": (
            "letter shares divide by all generated rows; agreement is the mean "
            "per-row fraction of the six semantic top-1 view pairs that agree"
        ),
    }
    diagnostics["green"] = (
        diagnostics["maximum_absolute_letter_share_delta_from_uniform"] <= 0.03
        and diagnostics["non_letter_emission_rate"] <= 0.002
        and diagnostics["four_view_pairwise_prediction_agreement"] > 0.85
    )
    diagnostics["thresholds"] = {
        "maximum_absolute_letter_share_delta_from_uniform": 0.03,
        "non_letter_emission_rate": 0.002,
        "four_view_pairwise_prediction_agreement_strictly_greater_than": 0.85,
    }
    source_by_id = {str(row["id"]): row for row in dev_rows}
    ensemble_rows = []
    stable_wrong = 0
    longest_option_choices = 0
    total_view_choices = 0
    ensemble_nll = 0.0
    for row_id, predictions in prediction_rows_by_id.items():
        if len(predictions) != 4:
            raise RuntimeError(f"Expected four probability views for {row_id}")
        predictions = sorted(predictions, key=lambda row: int(row["view"]))
        mean_probabilities = np.mean(
            np.asarray([row["probabilities"] for row in predictions], dtype=np.float64),
            axis=0,
        )
        gold = int(predictions[0]["gold"])
        predicted = int(np.argmax(mean_probabilities))
        ensemble_nll -= math.log(max(float(mean_probabilities[gold]), 1e-12))
        ensemble_rows.append(
            {
                "id": row_id,
                "domain": predictions[0]["domain"],
                "gold": gold,
                "prediction": predicted,
                "correct": predicted == gold,
            }
        )
        semantic_predictions = [int(row["prediction"]) for row in predictions]
        stable_wrong += int(len(set(semantic_predictions)) == 1 and semantic_predictions[0] != gold)
        option_lengths = [len(str(value).strip()) for value in source_by_id[row_id]["options"]]
        maximum_length = max(option_lengths)
        longest = {index for index, value in enumerate(option_lengths) if value == maximum_length}
        longest_option_choices += sum(value in longest for value in semantic_predictions)
        total_view_choices += len(semantic_predictions)
    p2c_diagnostics = {
        "probability_averaged_ensemble": {
            "accuracy": sum(bool(row["correct"]) for row in ensemble_rows) / len(ensemble_rows),
            "macro_accuracy": macro_accuracy(ensemble_rows),
            "nll": ensemble_nll / len(ensemble_rows),
        },
        "stable_wrong_rate": stable_wrong / len(ensemble_rows),
        "longest_option_choice_rate": longest_option_choices / total_view_choices,
        "longest_option_definition": (
            "semantic option with maximum stripped Unicode-code-point length; all tied maxima count"
        ),
    }
    return {
        "canonical": views[0],
        "permutations": views[1:],
        "selection_score": float(np.mean([row["macro_accuracy"] for row in views])),
        "mean_accuracy": float(np.mean([row["accuracy"] for row in views])),
        "generation": generation,
        "p2b_diagnostics": diagnostics,
        "p2c_diagnostics": p2c_diagnostics,
        "evaluation_seconds": time.perf_counter() - started,
    }, all_predictions + [{"generation": row} for row in generation_rows]


def question_view_scores(predictions: Sequence[dict[str, Any]]) -> dict[str, float]:
    by_id: dict[str, list[float]] = defaultdict(list)
    for row in predictions:
        if "id" in row:
            by_id[str(row["id"])].append(float(bool(row["correct"])))
    result = {}
    for row_id, values in by_id.items():
        if len(values) != 4:
            raise RuntimeError(f"Expected four views for bootstrap row {row_id}, found {len(values)}")
        result[row_id] = float(np.mean(values))
    return result


def p2c_paired_macro_bootstrap(
    candidate_predictions: Sequence[dict[str, Any]],
    baseline_predictions: Sequence[dict[str, Any]],
    dev_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Question-clustered, domain-stratified paired bootstrap for four-view macro."""
    candidate = question_view_scores(candidate_predictions)
    baseline = question_view_scores(baseline_predictions)
    domains: dict[str, list[str]] = defaultdict(list)
    for row in dev_rows:
        row_id = str(row["id"])
        if row_id not in candidate or row_id not in baseline:
            raise RuntimeError(f"Missing bootstrap prediction for {row_id}")
        domains[str(row["domain"])].append(row_id)
    observed_by_domain = {
        domain: float(np.mean([candidate[row_id] - baseline[row_id] for row_id in row_ids]))
        for domain, row_ids in domains.items()
    }
    observed = float(np.mean(list(observed_by_domain.values())))
    rng = np.random.default_rng(P2C_BOOTSTRAP_SEED)
    samples = np.empty(P2C_BOOTSTRAP_SAMPLES, dtype=np.float64)
    for sample_index in range(P2C_BOOTSTRAP_SAMPLES):
        domain_values = []
        for row_ids in domains.values():
            indices = rng.integers(0, len(row_ids), size=len(row_ids))
            domain_values.append(
                np.mean(
                    [candidate[row_ids[index]] - baseline[row_ids[index]] for index in indices]
                )
            )
        samples[sample_index] = np.mean(domain_values)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "estimate": observed,
        "ci_95": [float(lower), float(upper)],
        "by_domain": observed_by_domain,
        "bootstrap_seed": P2C_BOOTSTRAP_SEED,
        "bootstrap_samples": P2C_BOOTSTRAP_SAMPLES,
        "cluster": "question (all four views move together)",
        "stratification": "domain",
    }


def learning_rate(elapsed: float, budget: float, maximum: float, minimum: float) -> float:
    progress = min(1.0, max(0.0, elapsed / budget))
    if progress < 0.05:
        return maximum * max(0.01, progress / 0.05)
    cosine = (progress - 0.05) / 0.95
    return minimum + (maximum - minimum) * 0.5 * (1.0 + math.cos(math.pi * cosine))


def learning_rate_by_step(step: int, total_steps: int, maximum: float, minimum: float) -> float:
    progress = min(1.0, max(0.0, step / max(1, total_steps)))
    if progress < 0.05:
        return maximum * max(0.01, progress / 0.05)
    cosine = (progress - 0.05) / 0.95
    return minimum + (maximum - minimum) * 0.5 * (1.0 + math.cos(math.pi * cosine))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--budget-seconds", type=float, default=10_800.0)
    parser.add_argument("--checkpoint-seconds", type=float, default=1_800.0)
    parser.add_argument("--checkpoint-steps", type=int, default=0)
    parser.add_argument("--finalize-reserve-seconds", type=float, default=240.0)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--generation-dev-examples", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--early-stopping-evals", type=int, default=0)
    parser.add_argument("--minimum-nll-improvement", type=float, default=0.005)
    parser.add_argument("--selection-mode", choices=("legacy", "p2b_nll", "p2c"), default="legacy")
    parser.add_argument("--require-p2b-diagnostics", action="store_true")
    parser.add_argument("--p2c-arm", choices=("4v", "ens", "anchor"))
    parser.add_argument("--p2f-anchor", type=Path)
    parser.add_argument("--expected-p2f-anchor-sha256")
    parser.add_argument("--p2f-beta", type=float, default=P2F_ANCHOR_BETA)
    parser.add_argument("--p2f-evaluation-steps", default="")
    parser.add_argument("--expected-anchor-rows", type=int, default=0)
    parser.add_argument("--frozen-recipe", choices=("p2f",))
    parser.add_argument(
        "--expected-base-weight-sha256",
        default=EXPECTED_BASE_WEIGHT_SHA256,
    )
    parser.add_argument(
        "--starting-model-kind",
        default="merged-two-hour-rslora",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16 CUDA GPU is required")
    total_vram = torch.cuda.get_device_properties(0).total_memory
    if total_vram < 30 * 1024**3:
        raise RuntimeError(f"At least 30 GiB VRAM is required, found {total_vram / 1024**3:.1f}")
    model_weight = args.base_model / "model.safetensors"
    actual_base_weight_sha256 = sha256_file(model_weight)
    if actual_base_weight_sha256 != args.expected_base_weight_sha256:
        raise ValueError(f"Unexpected base-model weight hash: {actual_base_weight_sha256}")

    train_rows = read_jsonl(args.train)
    dev_rows = read_jsonl(args.dev)
    validate_clean(train_rows, args.train)
    validate_clean(dev_rows, args.dev)
    train_groups = {str(row["group_id"]) for row in train_rows}
    dev_groups = {str(row["group_id"]) for row in dev_rows}
    if train_groups & dev_groups:
        raise ValueError("Train/dev source groups overlap")

    p2f_evaluation_steps = {
        int(value) for value in args.p2f_evaluation_steps.split(",") if value.strip()
    }
    if args.p2c_arm == "anchor":
        if args.p2f_anchor is None or not args.expected_p2f_anchor_sha256:
            raise ValueError("P2F anchor mode requires a hash-pinned --p2f-anchor")
        actual_anchor_sha256 = sha256_file(args.p2f_anchor)
        if actual_anchor_sha256 != args.expected_p2f_anchor_sha256:
            raise ValueError(
                f"P2F anchor hash mismatch: {actual_anchor_sha256} != "
                f"{args.expected_p2f_anchor_sha256}"
            )
        anchor_rows = read_jsonl(args.p2f_anchor)
        anchors = {str(row["id"]): row for row in anchor_rows}
        replay_ids = {
            str(row["id"]) for row in train_rows if is_anchor_replay(row)
        }
        expected_anchor_rows = 850 if args.frozen_recipe == "p2f" else args.expected_anchor_rows
        if expected_anchor_rows <= 0:
            raise ValueError("Anchor mode requires positive --expected-anchor-rows")
        if (
            len(anchor_rows) != expected_anchor_rows
            or len(anchors) != expected_anchor_rows
            or set(anchors) != replay_ids
        ):
            raise ValueError(
                f"P2F anchor coverage mismatch: rows={len(anchor_rows)}, "
                f"unique={len(anchors)}, replay={len(replay_ids)}, "
                f"expected={expected_anchor_rows}"
            )
        for row in train_rows:
            row_id = str(row["id"])
            is_replay = row_id in replay_ids
            if is_replay:
                anchor = anchors[row_id]
                views = anchor.get("permutations")
                expected_orders = [list(order) for order in p2c_orders([row])[0]]
                if not isinstance(views, list) or len(views) != 4:
                    raise ValueError(f"Malformed P2F anchor views: {row_id}")
                if [view.get("semantic_order_by_display_slot") for view in views] != expected_orders:
                    raise ValueError(f"P2F anchor permutation mismatch: {row_id}")
                for view in views:
                    probabilities = list(map(float, view.get("semantic_probabilities") or []))
                    if len(probabilities) != 4 or abs(sum(probabilities) - 1.0) > 1e-5:
                        raise ValueError(f"Malformed P2F anchor probabilities: {row_id}")
                row["p2f_anchor"] = {"permutations": views}
            elif "p2f_anchor" in row:
                raise ValueError(f"Exam row unexpectedly contains a P2F anchor: {row_id}")
        if args.p2f_beta <= 0:
            raise ValueError("Anchor beta must be positive")
        if not p2f_evaluation_steps:
            raise ValueError("Anchor mode requires explicit evaluation steps")
        if args.frozen_recipe == "p2f":
            if args.p2f_beta != P2F_ANCHOR_BETA:
                raise ValueError(f"P2F beta is frozen to {P2F_ANCHOR_BETA}")
            if p2f_evaluation_steps != {100, 214}:
                raise ValueError("P2F evaluations are frozen to optimizer steps 100 and 214")
    elif args.p2f_anchor is not None or args.expected_p2f_anchor_sha256 is not None:
        raise ValueError("P2F anchor arguments are valid only for --p2c-arm anchor")
    elif args.expected_anchor_rows or args.frozen_recipe is not None:
        raise ValueError("Anchor-row and frozen-recipe arguments require --p2c-arm anchor")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = interface.CHAT_TEMPLATE
    label_ids = answer_token_ids(tokenizer)

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda()
    base.config.use_cache = False
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0.05,
        target_modules=list(TARGET_MODULES),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=True,
    )
    model = get_peft_model(base, lora_config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    modules = [
        name
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and "default" in module.lora_A
    ]
    if args.rank == 64 and trainable != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(f"Expected {EXPECTED_TRAINABLE_PARAMETERS} trainable parameters, found {trainable}")
    if len(modules) != 224:
        raise RuntimeError(f"Expected 224 LoRA modules, found {len(modules)}")

    generation_examples = (
        p2b_generation_examples(
            dev_rows,
            min(args.generation_dev_examples, len(dev_rows)),
            args.seed + 1,
            "exam_dev_early",
        )
        if args.selection_mode in {"p2b_nll", "p2c"}
        else interface.build_examples(
            dev_rows,
            min(args.generation_dev_examples, len(dev_rows)),
            args.seed + 1,
            "scaled_kd_dev",
        )
    )
    baseline_metrics, baseline_predictions = evaluate_model(
        model,
        tokenizer,
        dev_rows,
        label_ids,
        args.eval_batch_size,
        args.max_length,
        generation_examples,
        args.generation_batch_size,
    )
    write_json(args.output_dir / "baseline_metrics.json", baseline_metrics)
    write_jsonl(args.output_dir / "baseline_predictions.jsonl", baseline_predictions)

    if args.selection_mode in {"p2b_nll", "p2c"}:
        if args.max_epochs <= 0 or args.checkpoint_steps <= 0:
            raise ValueError("Exam modes require positive --max-epochs and --checkpoint-steps")
        if args.require_p2b_diagnostics and args.generation_dev_examples != len(dev_rows):
            raise ValueError("Exam diagnostics require generation over the full dev split")
        if args.require_p2b_diagnostics and not baseline_metrics["p2b_diagnostics"]["green"]:
            print("P2B baseline diagnostics are not all green; trained checkpoints must still pass", flush=True)
    if args.selection_mode == "p2c":
        if args.p2c_arm is None:
            raise ValueError("P2C mode requires --p2c-arm")
        if args.frozen_recipe == "p2f":
            if args.max_epochs != 2 or args.max_optimizer_steps != 214 or args.checkpoint_steps != 100:
                raise ValueError("P2F is frozen to 2 epochs, 214 optimizer steps, checkpoints every 100")
        elif args.p2c_arm == "anchor":
            if args.max_epochs != 2 or args.max_optimizer_steps != 250 or args.checkpoint_steps != 70:
                raise ValueError(
                    "FINAL-v1 anchor arm is frozen to 2 epochs, 250 optimizer steps, "
                    "checkpoints every 70 steps"
                )
            if p2f_evaluation_steps != {70, 140, 210, 250}:
                raise ValueError("FINAL-v1 evaluations are frozen to steps 70, 140, 210, and 250")
        elif args.max_epochs != 2 or args.max_optimizer_steps != 214 or args.checkpoint_steps != 100:
            raise ValueError("P2C controls are frozen to 2 epochs, 214 steps, checkpoints every 100")
        if args.generation_dev_examples != len(dev_rows):
            raise ValueError("P2C requires generation over the full frozen dev split")
        if args.p2c_arm != "anchor" and p2f_evaluation_steps:
            raise ValueError("P2F evaluation steps are valid only for the anchor arm")
    elif args.p2c_arm is not None:
        raise ValueError("--p2c-arm is only valid with --selection-mode p2c")

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.selection_mode in {"p2b_nll", "p2c"} and {parameter.dtype for parameter in parameters} != {torch.float32}:
        raise RuntimeError("Exam training requires FP32 adapter master weights")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=0.01,
        fused=True,
    )
    started = time.perf_counter()
    optimizer_steps = 0
    seen_examples = 0
    accumulated = 0
    epoch = 0
    cursor = 0
    order = list(range(len(train_rows)))
    random.Random(args.seed).shuffle(order)
    next_checkpoint = args.checkpoint_seconds
    history = []
    best = None
    evaluated_steps: set[int] = set()
    best_dev_nll = float(baseline_metrics["canonical"]["nll"])
    stale_evaluations = 0
    stop_reason = "unknown"
    losses: Counter[str] = Counter()
    optimization_seconds = 0.0
    optimizer.zero_grad(set_to_none=True)

    micro_batches_per_epoch = math.ceil(len(train_rows) / args.micro_batch_size)
    planned_steps = (
        math.ceil(micro_batches_per_epoch / args.gradient_accumulation) * args.max_epochs
        if args.max_epochs > 0
        else args.max_optimizer_steps
    )

    def elapsed() -> float:
        return time.perf_counter() - started

    def checkpoint(reason: str) -> None:
        nonlocal best, best_dev_nll, stale_evaluations
        model.eval()
        metrics, predictions = evaluate_model(
            model,
            tokenizer,
            dev_rows,
            label_ids,
            args.eval_batch_size,
            args.max_length,
            generation_examples,
            args.generation_batch_size,
        )
        metrics.update(
            {
                "reason": reason,
                "optimizer_steps": optimizer_steps,
                "seen_examples": seen_examples,
                "training_elapsed_seconds": elapsed(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "mean_losses": {
                    name: losses[name] / max(1, losses["micro_batches"])
                    for name in (
                        (
                            "total",
                            "base_total",
                            "gold_ce",
                            "teacher_kl_t2",
                            "ensemble_kl_t2",
                            "consistency_weight",
                            "eligible_rate",
                            "replay_anchor_kl_t1",
                            "replay_rate",
                        )
                        if args.selection_mode == "p2c"
                        else (
                            "total",
                            "gold_ce",
                            "teacher_kl_t2",
                            "consistency_js",
                            "eligible_rate",
                        )
                    )
                },
            }
        )
        generation = metrics["generation"]
        eos_floor = max(
            0.98,
            baseline_metrics["generation"]["immediate_eos_rate"] - 0.005,
        )
        legacy_eligible = (
            generation["exact_rate"] >= 0.99
            and generation["immediate_eos_rate"] >= eos_floor
            and metrics["canonical"]["accuracy"]
            >= baseline_metrics["canonical"]["accuracy"] - 0.005
        )
        if args.selection_mode == "p2c":
            gain = p2c_paired_macro_bootstrap(predictions, baseline_predictions, dev_rows)
            metrics["p2c_gain_vs_baseline"] = gain
            diagnostics = metrics["p2b_diagnostics"]
            agreement_floor = max(
                P2C_BASELINE_AGREEMENT - P2C_AGREEMENT_TOLERANCE,
                float(metrics["selection_score"]),
            )
            checks = {
                "exact_one_letter_output": float(generation["exact_rate"]) == 1.0,
                "non_letter_emission": float(diagnostics["non_letter_emission_rate"]) <= 0.002,
                "letter_marginal": float(
                    diagnostics["maximum_absolute_letter_share_delta_from_uniform"]
                ) <= 0.05,
                "agreement": float(diagnostics["four_view_pairwise_prediction_agreement"])
                >= agreement_floor,
                "four_view_macro_gain": float(gain["ci_95"][0]) > 0.0,
                "canonical_nll": float(metrics["canonical"]["nll"])
                <= P2C_BASELINE_CANONICAL_NLL,
            }
            metrics["eligible"] = all(checks.values())
            metrics["eligibility_checks"] = checks
            metrics["eligibility_thresholds"] = {
                "exact_one_letter_output": 1.0,
                "non_letter_emission_rate_at_most": 0.002,
                "letter_marginal_max_deviation_at_most": 0.05,
                "agreement_at_least": agreement_floor,
                "four_view_macro_gain_ci_lower_strictly_greater_than": 0.0,
                "canonical_nll_at_most": P2C_BASELINE_CANONICAL_NLL,
                "policy": "p2c-v2-frozen",
            }
        else:
            metrics["eligible"] = (
                bool(metrics["p2b_diagnostics"]["green"])
                if args.require_p2b_diagnostics
                else legacy_eligible
            )
            metrics["eligibility_thresholds"] = {
                "exact_rate": 0.99,
                "immediate_eos_rate": eos_floor,
                "canonical_accuracy": baseline_metrics["canonical"]["accuracy"] - 0.005,
                "policy": "absolute-output-floor-plus-relative-base-non-regression-v1",
            }
        current_nll = float(metrics["canonical"]["nll"])
        if best_dev_nll - current_nll >= args.minimum_nll_improvement:
            best_dev_nll = current_nll
            stale_evaluations = 0
            metrics["nll_improved_by_required_margin"] = True
        else:
            stale_evaluations += 1
            metrics["nll_improved_by_required_margin"] = False
        metrics["nll_early_stopping"] = {
            "best_nll_after_evaluation": best_dev_nll,
            "minimum_improvement": args.minimum_nll_improvement,
            "consecutive_evaluations_without_required_improvement": stale_evaluations,
        }
        checkpoint_dir = args.output_dir / f"checkpoint_step_{optimizer_steps:06d}"
        adapter_dir = checkpoint_dir / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        write_json(checkpoint_dir / "metrics.json", metrics)
        write_jsonl(checkpoint_dir / "dev_predictions.jsonl", predictions)
        record = {
            "checkpoint": str(checkpoint_dir),
            "adapter": str(adapter_dir),
            "metrics": metrics,
        }
        history.append(record)
        eligible = [row for row in history if row["metrics"]["eligible"]]
        if args.selection_mode == "p2b_nll":
            best = min(
                eligible,
                key=lambda row: (
                    row["metrics"]["canonical"]["nll"],
                    -row["metrics"]["selection_score"],
                    row["metrics"]["optimizer_steps"],
                ),
            ) if eligible else None
        elif args.selection_mode == "p2c":
            best = max(
                eligible,
                key=lambda row: (
                    row["metrics"]["selection_score"],
                    -row["metrics"]["canonical"]["nll"],
                    -row["metrics"]["optimizer_steps"],
                ),
            ) if eligible else None
        else:
            pool = eligible or history
            best = max(
                pool,
                key=lambda row: (
                    row["metrics"]["selection_score"],
                    row["metrics"]["canonical"]["accuracy"],
                    -row["metrics"]["canonical"]["nll"],
                    -row["metrics"]["training_elapsed_seconds"],
                ),
            )
        write_json(args.output_dir / "training_progress.json", {"history": history, "best": best})
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "optimizer_steps": optimizer_steps,
                "seen_examples": seen_examples,
                "epoch": epoch,
                "cursor": cursor,
                "order": order,
                "history": history,
                "best": best,
            },
            args.output_dir / "resume_state.pt",
        )
        print(
            f"checkpoint step={optimizer_steps} score={metrics['selection_score']:.4f} "
            f"canonical={metrics['canonical']['accuracy']:.4f} "
            f"exact={generation['exact_rate']:.4f} eligible={metrics['eligible']}",
            flush=True,
        )
        evaluated_steps.add(optimizer_steps)
        model.train()

    def optimizer_step(partial_micro_batches: int | None = None) -> None:
        nonlocal optimizer_steps, accumulated
        if partial_micro_batches is not None and partial_micro_batches < args.gradient_accumulation:
            correction = args.gradient_accumulation / partial_micro_batches
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        if args.selection_mode in {"p2b_nll", "p2c"} and planned_steps > 0:
            lr = learning_rate_by_step(
                optimizer_steps + 1, planned_steps, args.learning_rate, args.minimum_learning_rate
            )
        else:
            lr = learning_rate(
                elapsed(), args.budget_seconds, args.learning_rate, args.minimum_learning_rate
            )
        for group in optimizer.param_groups:
            group["lr"] = lr
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        accumulated = 0

    while True:
        if args.smoke_steps and optimizer_steps >= args.smoke_steps:
            stop_reason = "smoke_steps"
            break
        if not args.smoke_steps and elapsed() >= args.budget_seconds - args.finalize_reserve_seconds:
            stop_reason = "wall_time_reserve"
            break
        if args.max_optimizer_steps and optimizer_steps >= args.max_optimizer_steps:
            stop_reason = "max_optimizer_steps"
            break
        if args.early_stopping_evals and stale_evaluations >= args.early_stopping_evals:
            stop_reason = "nll_early_stopping"
            break
        if cursor >= len(order):
            if args.selection_mode in {"p2b_nll", "p2c"} and accumulated:
                optimizer_step(accumulated)
                if (
                    args.checkpoint_steps
                    and optimizer_steps % args.checkpoint_steps == 0
                    and (args.p2c_arm != "anchor" or optimizer_steps in p2f_evaluation_steps)
                ):
                    checkpoint("step_interval")
            epoch += 1
            if args.max_epochs and epoch >= args.max_epochs:
                stop_reason = "max_epochs"
                break
            cursor = 0
            order = list(range(len(train_rows)))
            random.Random(args.seed + epoch).shuffle(order)
        indices = order[cursor : cursor + args.micro_batch_size]
        cursor += len(indices)
        rows = [train_rows[index] for index in indices]
        optimization_started = time.perf_counter()
        if args.selection_mode == "p2c":
            row_orders = p2c_orders(rows)
            views = [
                student_semantic_logits(
                    model,
                    tokenizer,
                    rows,
                    [orders[view_index] for orders in row_orders],
                    label_ids,
                    args.max_length,
                )
                for view_index in range(4)
            ]
            loss, components = p2c_loss(
                views,
                rows,
                args.temperature,
                str(args.p2c_arm),
                optimizer_steps + 1,
                args.p2f_beta,
            )
        else:
            canonical_orders = [tuple(range(4)) for _ in rows]
            permutation_orders = [non_identity_order(str(row["id"]), optimizer_steps) for row in rows]
            canonical = student_semantic_logits(
                model, tokenizer, rows, canonical_orders, label_ids, args.max_length
            )
            permuted = student_semantic_logits(
                model, tokenizer, rows, permutation_orders, label_ids, args.max_length
            )
            loss, components = kd_loss(canonical, permuted, rows, args.temperature)
        (loss / args.gradient_accumulation).backward()
        optimization_seconds += time.perf_counter() - optimization_started
        accumulated += 1
        seen_examples += len(rows)
        losses["micro_batches"] += 1
        for name, value in components.items():
            losses[name] += value
        if accumulated < args.gradient_accumulation:
            continue
        optimizer_step()
        if (
            args.checkpoint_steps
            and optimizer_steps % args.checkpoint_steps == 0
            and (args.p2c_arm != "anchor" or optimizer_steps in p2f_evaluation_steps)
        ):
            checkpoint("step_interval")
        elif not args.checkpoint_steps and elapsed() >= next_checkpoint:
            checkpoint("interval")
            next_checkpoint += args.checkpoint_seconds
        if optimizer_steps % 50 == 0:
            print(
                f"progress step={optimizer_steps} seen={seen_examples} epoch={epoch} "
                f"elapsed={elapsed():.1f}s lr={optimizer.param_groups[0]['lr']:.8f}",
                flush=True,
            )

    if accumulated:
        optimizer.zero_grad(set_to_none=True)
    if optimizer_steps not in evaluated_steps:
        checkpoint("smoke" if args.smoke_steps else stop_reason)
    selected_adapter = Path(best["adapter"]) if best is not None else None
    if args.selection_mode not in {"p2b_nll", "p2c"} and selected_adapter is None:
        raise RuntimeError("No checkpoint was selected")
    if args.selection_mode in {"p2b_nll", "p2c"}:
        state_dtypes = {
            value.dtype
            for state in optimizer.state.values()
            for value in state.values()
            if torch.is_tensor(value) and value.is_floating_point() and value.numel() > 1
        }
        if state_dtypes != {torch.float32}:
            raise RuntimeError(f"P2B optimizer master-state dtype mismatch: {state_dtypes}")
    manifest = {
        "protocol": (
            (
                "zagreus-exam-p2f-anchored-four-view-rslora-v1"
                if args.p2c_arm == "anchor"
                else "zagreus-exam-p2c-four-view-rslora-v2"
            )
            if args.selection_mode == "p2c"
            else (
                "zagreus-exam-p2b-soft-kd-rslora-v1"
                if args.selection_mode == "p2b_nll"
                else "zagreus-scaled-sector-soft-kd-rslora-v1"
            )
        ),
        "official_italic_read_or_used_for_training_or_selection": False,
        "starting_model": {
            "kind": args.starting_model_kind,
            "path": str(args.base_model),
            "model_weight_sha256": actual_base_weight_sha256,
        },
        "model_id": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "data": {
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "train_sha256": sha256_file(args.train),
            "dev_sha256": sha256_file(args.dev),
            "train_dev_group_overlap": False,
            "selection": "fresh-external-source-dev-only",
        },
        "objective": (
            {
                "arm": args.p2c_arm,
                "views_per_row": 4,
                "teacher_target": "matching preserved per-view E4B posterior",
                "eligible_base": {"gold_ce": 0.40, "teacher_kl": 0.50},
                "teacher_ineligible_base": {"gold_ce": 0.90},
                "ensemble_distillation": (
                    "KL(stopgrad(mean(softmax(student_view_logits/T)))) || softmax(student_view_logits/T) * T^2"
                    if args.p2c_arm == "ens"
                    else None
                ),
                "p2f_replay_anchor": (
                    {
                        "rows": len([row for row in train_rows if is_anchor_replay(row)]),
                        "loss": "gold CE + beta * KL(P_36 || P_student)",
                        "temperature": 1.0,
                        "beta": args.p2f_beta,
                        "anchor_sha256": args.expected_p2f_anchor_sha256,
                        "e4b_kd_on_replay": False,
                        "replay_marker": "anchor_replay with legacy source_split fallback",
                        "frozen_recipe": args.frozen_recipe,
                    }
                    if args.p2c_arm == "anchor"
                    else None
                ),
                "normalization": "(1-lambda)*L_base + lambda*L_cons",
                "lambda_schedule": (
                    {"start_step": 100, "end_step": 140, "maximum": 0.5}
                    if args.p2c_arm == "ens"
                    else {"constant": 0.0}
                ),
                "temperature": args.temperature,
            }
            if args.selection_mode == "p2c"
            else {
                "eligible_rows": {
                    "gold_ce": 0.40,
                    "teacher_kl": 0.50,
                    "permutation_consistency_js": 0.10,
                },
                "teacher_ineligible_rows": {
                    "gold_ce": 0.90,
                    "permutation_consistency_js": 0.10,
                },
                "temperature": args.temperature,
            }
        ),
        "adapter": {
            "type": "all-layer-rslora",
            "rank": args.rank,
            "alpha": args.rank,
            "dropout": 0.05,
            "target_modules": list(TARGET_MODULES),
            "lora_modules": len(modules),
            "trainable_parameters": trainable,
        },
        "config": vars(args) | {
            "base_model": str(args.base_model),
            "train": str(args.train),
            "dev": str(args.dev),
            "output_dir": str(args.output_dir),
            "p2f_anchor": str(args.p2f_anchor) if args.p2f_anchor is not None else None,
        },
        "baseline": baseline_metrics,
        "training_seconds": elapsed(),
        "optimization_seconds_excluding_evaluation": optimization_seconds,
        "stop_reason": stop_reason,
        "optimizer_steps": optimizer_steps,
        "seen_examples": seen_examples,
        "epochs_completed": seen_examples / max(1, len(train_rows)),
        "history": history,
        "selected": best,
        "selected_adapter_sha256": (
            sha256_file(selected_adapter / "adapter_model.safetensors")
            if selected_adapter is not None
            else None
        ),
        "fp32_adapter_master_weights": all(parameter.dtype == torch.float32 for parameter in parameters),
        "fp32_optimizer_master_state": (
            all(
                value.dtype == torch.float32
                for state in optimizer.state.values()
                for value in state.values()
                if torch.is_tensor(value) and value.is_floating_point() and value.numel() > 1
            )
            if args.selection_mode in {"p2b_nll", "p2c"}
            else None
        ),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print("SCALED_KD_TRAINING=" + json.dumps(manifest, ensure_ascii=False), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
