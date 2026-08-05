#!/usr/bin/env python3
"""Build the audited four-view, five-shot ENSEMBLE-v1 training recipes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from zagreus_ensemble_t2_common import read_jsonl, sha256

EXPECTED_ANCHORS = "913023f76d46f968010baff871204705a67c854884adf91a3b80e9029857559f"
PROFILE_REFERENCE = {3: 2464, 4: 7466, 5: 68}  # ITALIC public aggregate, excluding two 2-option rows.
TARGET_COUNTS = {3: 5915, 4: 17922, 5: 163}  # exact largest-remainder allocation over 24,000 rows.


def stable_int(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def top1(values: Sequence[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def view_probabilities(row: dict[str, Any], view: int, control: bool) -> list[float]:
    supervision = row["control_teacher_supervision"] if control else row["teacher_supervision"]
    if control:
        return list(map(float, supervision["teachers"][0]["views"][view]["probabilities"]))
    teachers = supervision["teachers"]
    return [sum(float(teacher["views"][view]["probabilities"][option]) for teacher in teachers) / len(teachers) for option in range(4)]


def transformed_probabilities(values: Sequence[float], base_indices: Sequence[int | None]) -> list[float]:
    transformed = [1e-8 if index is None else float(values[index]) for index in base_indices]
    total = sum(transformed)
    return [value / total for value in transformed]


def transformed_item(
    row: dict[str, Any], option_count: int, protected: set[int], donor_pool: Sequence[dict[str, Any]], salt: str,
) -> tuple[list[str], list[int | None], dict[int, int]]:
    base_indices: list[int | None] = list(range(4))
    options = list(map(str, row["options"]))
    if option_count == 3:
        candidates = [index for index in range(4) if index not in protected]
        if not candidates:
            raise RuntimeError(f"No legal three-option drop: {row['id']}")
        drop = candidates[stable_int("drop", salt, row["id"]) % len(candidates)]
        base_indices = [index for index in range(4) if index != drop]
        options = [options[index] for index in base_indices]
    elif option_count == 5:
        normalized = {" ".join(value.casefold().split()) for value in options}
        start = stable_int("donor", salt, row["id"]) % len(donor_pool)
        for offset in range(len(donor_pool)):
            donor = donor_pool[(start + offset) % len(donor_pool)]
            if donor["group_id"] == row["group_id"]:
                continue
            donor_gold = int(donor["campaign_gold"])
            candidates = [index for index in range(4) if index != donor_gold]
            candidate = str(donor["options"][candidates[stable_int("donor-option", salt, donor["id"]) % 3]])
            if " ".join(candidate.casefold().split()) not in normalized:
                options.append(candidate)
                base_indices.append(None)
                break
        if len(options) != 5:
            raise RuntimeError(f"No legal five-option distractor: {row['id']}")
    elif option_count != 4:
        raise ValueError(option_count)
    mapping = {base: transformed for transformed, base in enumerate(base_indices) if base is not None}
    return options, base_indices, mapping


def placed_order(option_count: int, answer_semantic: int, desired_display: int, salt: str, prior: set[tuple[int, ...]] | None = None) -> tuple[int, ...]:
    for attempt in range(100):
        remaining = [index for index in range(option_count) if index != answer_semantic]
        random.Random(stable_int("order", salt, attempt)).shuffle(remaining)
        order: list[int | None] = [None] * option_count
        order[desired_display] = answer_semantic
        cursor = 0
        for display in range(option_count):
            if order[display] is None:
                order[display] = remaining[cursor]
                cursor += 1
        result = tuple(int(value) for value in order)
        if prior is None or result not in prior:
            return result
    raise RuntimeError(f"Could not create unique order: {salt}")


def option_count_cycle() -> list[int]:
    cycle = []
    for count, amount in PROFILE_REFERENCE.items():
        cycle.extend([count] * amount)
    random.Random(20260804).shuffle(cycle)
    return cycle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.anchors) != EXPECTED_ANCHORS:
        raise RuntimeError("T3 anchor hash contract failed")
    signal = read_jsonl(args.signal)
    anchors = read_jsonl(args.anchors)
    if len(signal) != 21_000 or len(anchors) != 3_000:
        raise RuntimeError("T3 row-count contract failed")
    for row in signal + anchors:
        if str(row.get("source", "")).casefold() == "italic" or str(row.get("id", "")).casefold().startswith("italic:"):
            raise RuntimeError("Official row reached T3")
    authorized = [row for row in signal if row.get("gold_ce_authorized") and row.get("campaign_gold") is not None]
    if len(authorized) < 5:
        raise RuntimeError("Insufficient training-pool demonstrations")
    donor_pool = authorized
    all_targets = [(row, "signal") for row in signal] + [(row, "anchor") for row in anchors]
    ranked = sorted(range(len(all_targets)), key=lambda index: stable_int("target-count", all_targets[index][0]["id"]))
    target_count_by_index = {}
    cursor = 0
    for count in (3, 4, 5):
        for index in ranked[cursor:cursor + TARGET_COUNTS[count]]:
            target_count_by_index[index] = count
        cursor += TARGET_COUNTS[count]
    if cursor != 24_000:
        raise RuntimeError("T3 option-count allocation drift")
    count_cycle = option_count_cycle()
    demo_count_cursor = 0
    display_counters = {"target": Counter(), "demo": Counter()}
    display_histograms = {"target": defaultdict(Counter), "demo": defaultdict(Counter)}
    output_path = args.output_dir / "format_recipes.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for target_index, (row, row_kind) in enumerate(all_targets):
            option_count = target_count_by_index[target_index]
            if row_kind == "signal":
                control_mean = list(map(float, row["control_teacher_supervision"]["averaged_probabilities"]))
                ensemble_mean = list(map(float, row["teacher_supervision"]["averaged_probabilities"]))
                authorized_gold = int(row["campaign_gold"]) if row.get("gold_ce_authorized") else None
                answer_base = authorized_gold if authorized_gold is not None else top1(ensemble_mean)
                protected = {top1(control_mean), top1(ensemble_mean)}
                if authorized_gold is not None:
                    protected.add(authorized_gold)
            else:
                answer_base = int(row["campaign_gold"])
                authorized_gold = answer_base
                protected = {answer_base}
            target_options, target_base_indices, target_mapping = transformed_item(
                row, option_count, protected, donor_pool, f"target:{target_index}"
            )
            answer_semantic = target_mapping[answer_base]
            used_target_orders: set[tuple[int, ...]] = set()
            variants = []
            for view in range(4):
                desired = display_counters["target"][option_count] % option_count
                display_counters["target"][option_count] += 1
                target_order = placed_order(option_count, answer_semantic, desired, f"target:{row['id']}:{view}", used_target_orders)
                used_target_orders.add(target_order)
                display_histograms["target"][option_count][desired] += 1
                demos = []
                used_groups = {str(row["group_id"])}
                start = stable_int("demos", row["id"], view) % len(authorized)
                probe = 0
                while len(demos) < 5:
                    demo = authorized[(start + probe) % len(authorized)]
                    probe += 1
                    if str(demo["group_id"]) in used_groups:
                        continue
                    used_groups.add(str(demo["group_id"]))
                    demo_count = count_cycle[demo_count_cursor % len(count_cycle)]
                    demo_count_cursor += 1
                    demo_gold = int(demo["campaign_gold"])
                    demo_options, demo_base_indices, demo_mapping = transformed_item(
                        demo, demo_count, {demo_gold}, donor_pool, f"demo:{row['id']}:{view}:{len(demos)}"
                    )
                    demo_answer = demo_mapping[demo_gold]
                    demo_display = display_counters["demo"][demo_count] % demo_count
                    display_counters["demo"][demo_count] += 1
                    demo_order = placed_order(demo_count, demo_answer, demo_display, f"demo-order:{row['id']}:{view}:{len(demos)}")
                    display_histograms["demo"][demo_count][demo_display] += 1
                    demos.append({
                        "id": str(demo["id"]), "option_count": demo_count, "options": demo_options,
                        "base_semantic_indices": demo_base_indices,
                        "semantic_order_by_display_slot": list(demo_order), "answer_display": demo_display,
                    })
                target = {
                    "option_count": option_count, "options": target_options,
                    "base_semantic_indices": target_base_indices,
                    "semantic_order_by_display_slot": list(target_order), "answer_display": desired,
                    "gold_ce_authorized": authorized_gold is not None,
                }
                if row_kind == "signal":
                    target["control_probabilities"] = transformed_probabilities(view_probabilities(row, view, True), target_base_indices)
                    target["ensemble_probabilities"] = transformed_probabilities(view_probabilities(row, view, False), target_base_indices)
                    target["agreement_weight"] = float(row["teacher_supervision"]["agreement_weight"])
                variants.append({"view": view, "prompt_messages": 12, "target_completion": "space-plus-letter-plus-eos", "demos": demos, "target": target})
            recipe = {"id": str(row["id"]), "group_id": str(row["group_id"]), "row_kind": row_kind, "variants": variants}
            handle.write(json.dumps(recipe, ensure_ascii=False, separators=(",", ":")) + "\n")
            if (target_index + 1) % 1000 == 0:
                print(f"T3 recipes={target_index + 1}/24000", flush=True)
    for section in ("target", "demo"):
        for count, histogram in display_histograms[section].items():
            values = [histogram[index] for index in range(count)]
            if max(values) - min(values) > 1:
                raise RuntimeError(f"Displayed-letter imbalance: {section}/{count}: {values}")
    manifest = {
        "protocol": "zagreus-ensemble-v1-t3-format-recipes",
        "official_italic_read_or_used": False,
        "inputs": {"signal_sha256": sha256(args.signal), "anchors_sha256": EXPECTED_ANCHORS},
        "unique_signal_rows": len(signal), "unique_anchor_rows": len(anchors), "recipe_rows": 24_000,
        "variants_per_row": 4, "prompt_messages_per_variant": 12, "target_is_completion_not_thirteenth_message": True,
        "supervised_assistant_turns_per_variant": 6,
        "target_option_counts": TARGET_COUNTS,
        "profile_reference": {"3": 2464, "4": 7466, "5": 68, "excluded_2_option_rows": 2},
        "displayed_answer_histograms": {
            section: {str(count): dict(sorted(hist.items())) for count, hist in sorted(values.items())}
            for section, values in display_histograms.items()
        },
        "demonstration_pool": {"source": "training signal only", "authorized_gold_rows": len(authorized), "fixtures_used": False},
        "recipes_sha256": sha256(output_path),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print("T3_RESULT=" + json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
