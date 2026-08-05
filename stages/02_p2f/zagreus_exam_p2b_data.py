#!/usr/bin/env python3
"""Freeze and finalize the RUNBOOK-v2 P2B exam + replay dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import zagreus_exam_bank_v1 as bank


PROTOCOL = "zagreus-exam-p2b-data-v1"
SEED = 20260802
REPLAY_ROWS = 850
EXPECTED_TEACHER_ID = "google/gemma-4-E4B-it"
EXPECTED_TEACHER_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def stable_key(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{SEED}:{namespace}:{value}".encode()).hexdigest()


def teacher_input_row(row: Mapping[str, Any], split: str) -> dict[str, Any]:
    result = json.loads(json.dumps(row, ensure_ascii=False))
    result.setdefault("context", "")
    result["source_split"] = split
    result["p2b_origin"] = "official_answer_keyed_exam_bank"
    return result


def validate_exam_rows(rows: list[dict[str, Any]], expected: int, label: str) -> None:
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} {label} rows, found {len(rows)}")
    for row in rows:
        if len(row.get("options", [])) != 4 or not 0 <= int(row.get("gold", -1)) < 4:
            raise RuntimeError(f"Malformed {label} row: {row.get('id')}")
        if not str(row.get("key_provenance", "")).casefold().startswith("official "):
            raise RuntimeError(f"Non-official gold in {label}: {row.get('id')}")


def prepare(args: argparse.Namespace) -> None:
    gate0 = read_json(args.gate0)
    routing = read_json(args.routing)
    if gate0.get("status") != "PASSED" or not gate0.get("gpu_spend_authorized"):
        raise RuntimeError("Gate 0 is not passed")
    routes = routing["metrics"]["routes"]
    if any(value["injection_gate"] for value in routes.values()):
        raise RuntimeError("P2B-only preparation expected no injection route")
    if routes.get("syntax", {}).get("route") != "rlvr_candidate":
        raise RuntimeError("Corrected Gate-1 routing artifact is not the expected version")

    exam_train = read_jsonl(args.exam_train)
    dev_early = read_jsonl(args.dev_early)
    validate_exam_rows(exam_train, 2550, "exam_train")
    validate_exam_rows(dev_early, 600, "dev_early")
    other_exam = [read_jsonl(path) for path in args.other_exam_splits]
    all_exam = exam_train + dev_early + [row for split_rows in other_exam for row in split_rows]
    exam_groups = {str(row["group_id"]) for row in all_exam}
    exam_hashes = {bank.normalized_question_hash(row["question"]) for row in all_exam}

    replay = read_jsonl(args.replay_teacher_train)
    eligible_replay = []
    exclusions = Counter()
    for row in replay:
        teacher = row.get("teacher", {})
        if teacher.get("model_id") != EXPECTED_TEACHER_ID or teacher.get("model_revision") != EXPECTED_TEACHER_REVISION:
            exclusions["teacher_mismatch"] += 1
            continue
        if int(teacher.get("permutation_count", -1)) != 4:
            exclusions["teacher_view_mismatch"] += 1
            continue
        if str(row["group_id"]) in exam_groups:
            exclusions["exam_group_overlap"] += 1
            continue
        if bank.normalized_question_hash(row["question"]) in exam_hashes:
            exclusions["exam_question_overlap"] += 1
            continue
        eligible_replay.append(row)
    selected_replay = sorted(
        eligible_replay, key=lambda row: stable_key("replay", str(row["id"]))
    )[:REPLAY_ROWS]
    if len(selected_replay) != REPLAY_ROWS:
        raise RuntimeError(f"Only {len(selected_replay)} eligible replay rows")
    if len({str(row["id"]) for row in selected_replay}) != REPLAY_ROWS:
        raise RuntimeError("Duplicate replay IDs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir = args.output_dir / "teacher_input"
    teacher_smoke_dir = args.output_dir / "teacher_smoke_input"
    teacher_train = [teacher_input_row(row, "exam_train") for row in exam_train]
    teacher_dev = [teacher_input_row(row, "dev_early") for row in dev_early]
    train_path = teacher_dir / "train.jsonl"
    dev_path = teacher_dir / "dev.jsonl"
    replay_path = args.output_dir / "replay_selected.jsonl"
    smoke_train_path = teacher_smoke_dir / "train.jsonl"
    smoke_dev_path = teacher_smoke_dir / "dev.jsonl"
    write_jsonl(train_path, teacher_train)
    write_jsonl(dev_path, teacher_dev)
    write_jsonl(replay_path, selected_replay)
    smoke_train = [
        row
        for domain in sorted({str(row["domain"]) for row in teacher_train})
        for row in sorted(
            [item for item in teacher_train if str(item["domain"]) == domain],
            key=lambda item: stable_key("teacher-smoke-train", str(item["id"])),
        )[:2]
    ]
    smoke_dev = [
        row
        for domain in sorted({str(row["domain"]) for row in teacher_dev})
        for row in sorted(
            [item for item in teacher_dev if str(item["domain"]) == domain],
            key=lambda item: stable_key("teacher-smoke-dev", str(item["id"])),
        )[:2]
    ]
    write_jsonl(smoke_train_path, smoke_train)
    write_jsonl(smoke_dev_path, smoke_dev)
    manifest = {
        "protocol": PROTOCOL,
        "stage": "teacher_input_frozen",
        "official_italic_read_or_used": False,
        "frozen_before_teacher_inference": True,
        "seed": SEED,
        "rows": {"exam_train": len(teacher_train), "dev_early": len(teacher_dev), "replay": len(selected_replay)},
        "replay": {
            "selection": "sha256-order-after-exam-overlap-removal",
            "target_fraction_final_train": 0.25,
            "eligible_before_selection": len(eligible_replay),
            "exclusions": dict(exclusions),
        },
        "domains": {
            "exam_train": dict(Counter(str(row["domain"]) for row in teacher_train)),
            "dev_early": dict(Counter(str(row["domain"]) for row in teacher_dev)),
            "replay": dict(Counter(str(row["domain"]) for row in selected_replay)),
        },
        "input_hashes": {
            "gate0": bank.sha256_file(args.gate0),
            "routing": bank.sha256_file(args.routing),
            "exam_train": bank.sha256_file(args.exam_train),
            "dev_early": bank.sha256_file(args.dev_early),
            "replay_teacher_train": bank.sha256_file(args.replay_teacher_train),
        },
        "output_hashes": {
            "teacher_train": bank.sha256_file(train_path),
            "teacher_dev": bank.sha256_file(dev_path),
            "replay_selected": bank.sha256_file(replay_path),
            "teacher_smoke_train": bank.sha256_file(smoke_train_path),
            "teacher_smoke_dev": bank.sha256_file(smoke_dev_path),
        },
    }
    write_json(args.output_dir / "prepare_manifest.json", manifest)
    print("ZAGREUS_EXAM_P2B_PREPARE=" + json.dumps(manifest, ensure_ascii=False), flush=True)


def validate_teacher_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        teacher = row.get("teacher", {})
        ensemble = teacher.get("ensemble", {})
        if teacher.get("model_id") != EXPECTED_TEACHER_ID or teacher.get("model_revision") != EXPECTED_TEACHER_REVISION:
            raise RuntimeError(f"Teacher mismatch: {row.get('id')}")
        if int(teacher.get("permutation_count", -1)) != 4:
            raise RuntimeError(f"Teacher view mismatch: {row.get('id')}")
        probabilities = ensemble.get("probabilities", [])
        if len(probabilities) != 4 or abs(sum(map(float, probabilities)) - 1.0) > 1e-5:
            raise RuntimeError(f"Malformed teacher posterior: {row.get('id')}")


def finalize(args: argparse.Namespace) -> None:
    prepared = read_json(args.prepare_manifest)
    if prepared.get("protocol") != PROTOCOL or not prepared.get("frozen_before_teacher_inference"):
        raise RuntimeError("P2B teacher inputs were not frozen")
    expected = prepared["output_hashes"]
    if bank.sha256_file(args.teacher_input_train) != expected["teacher_train"]:
        raise RuntimeError("Teacher train input changed")
    if bank.sha256_file(args.teacher_input_dev) != expected["teacher_dev"]:
        raise RuntimeError("Teacher dev input changed")
    if bank.sha256_file(args.replay_selected) != expected["replay_selected"]:
        raise RuntimeError("Replay selection changed")

    exam_train = read_jsonl(args.teacher_train)
    dev = read_jsonl(args.teacher_dev)
    replay = read_jsonl(args.replay_selected)
    validate_teacher_rows(exam_train)
    validate_teacher_rows(dev)
    validate_teacher_rows(replay)
    if len(exam_train) != 2550 or len(dev) != 600 or len(replay) != REPLAY_ROWS:
        raise RuntimeError("Unexpected P2B row counts")
    if {str(row["id"]) for row in exam_train} != {
        str(row["id"]) for row in read_jsonl(args.teacher_input_train)
    }:
        raise RuntimeError("Teacher output changed exam-train IDs")
    if {str(row["id"]) for row in dev} != {
        str(row["id"]) for row in read_jsonl(args.teacher_input_dev)
    }:
        raise RuntimeError("Teacher output changed dev IDs")

    train = sorted(exam_train + replay, key=lambda row: stable_key("final-train", str(row["id"])))
    train_ids = {str(row["id"]) for row in train}
    dev_ids = {str(row["id"]) for row in dev}
    train_groups = {str(row["group_id"]) for row in train}
    dev_groups = {str(row["group_id"]) for row in dev}
    train_hashes = {bank.normalized_question_hash(row["question"]) for row in train}
    dev_hashes = {bank.normalized_question_hash(row["question"]) for row in dev}
    if train_ids & dev_ids or train_groups & dev_groups or train_hashes & dev_hashes:
        raise RuntimeError("P2B train/dev overlap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    dev_path = args.output_dir / "dev.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(dev_path, dev)
    manifest = {
        "protocol": PROTOCOL,
        "stage": "final_train_frozen",
        "official_italic_read_or_used": False,
        "gold_policy": "official-answer-key-for-exam; deterministic-source-gold-for-replay; teacher-never-relabels",
        "teacher": {"model_id": EXPECTED_TEACHER_ID, "revision": EXPECTED_TEACHER_REVISION, "permutations": 4},
        "rows": {"train": len(train), "exam_train": len(exam_train), "replay": len(replay), "dev": len(dev)},
        "replay_fraction": len(replay) / len(train),
        "teacher_eligible": {
            "train": sum(bool(row["teacher"]["ensemble"]["eligible"]) for row in train),
            "dev": sum(bool(row["teacher"]["ensemble"]["eligible"]) for row in dev),
        },
        "domains": {
            "train": dict(Counter(str(row["domain"]) for row in train)),
            "dev": dict(Counter(str(row["domain"]) for row in dev)),
        },
        "overlaps": {"id": 0, "group": 0, "normalized_question": 0},
        "input_hashes": {
            "prepare_manifest": bank.sha256_file(args.prepare_manifest),
            "teacher_manifest": bank.sha256_file(args.teacher_manifest),
            "teacher_train": bank.sha256_file(args.teacher_train),
            "teacher_dev": bank.sha256_file(args.teacher_dev),
            "replay_selected": bank.sha256_file(args.replay_selected),
        },
        "output_hashes": {"train": bank.sha256_file(train_path), "dev": bank.sha256_file(dev_path)},
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print("ZAGREUS_EXAM_P2B_FINALIZE=" + json.dumps(manifest, ensure_ascii=False), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    prepare_cmd = commands.add_parser("prepare")
    prepare_cmd.add_argument("--gate0", type=Path, default=Path("results/zagreus_exam_bank_v1/evidence/GATE_0.json"))
    prepare_cmd.add_argument("--routing", type=Path, default=Path("results/zagreus_exam_bank_v1/phase1_runpod/full/routing_analysis_v1_1.json"))
    prepare_cmd.add_argument("--exam-train", type=Path, default=Path("results/zagreus_exam_bank_v1/private_derived/splits/exam_train.jsonl"))
    prepare_cmd.add_argument("--dev-early", type=Path, default=Path("results/zagreus_exam_bank_v1/private_derived/splits/dev_early.jsonl"))
    prepare_cmd.add_argument("--other-exam-splits", type=Path, nargs="+", default=[Path("results/zagreus_exam_bank_v1/private_derived/splits/exam_diag.jsonl"), Path("results/zagreus_exam_bank_v1/private_derived/splits/final_holdout.jsonl")])
    prepare_cmd.add_argument("--replay-teacher-train", type=Path, default=Path("results/zagreus_exam_bank_v1/p2b_prior_teacher/train.jsonl"))
    prepare_cmd.add_argument("--output-dir", type=Path, default=Path("results/zagreus_exam_bank_v1/p2b_data"))
    prepare_cmd.set_defaults(function=prepare)

    finalize_cmd = commands.add_parser("finalize")
    finalize_cmd.add_argument("--prepare-manifest", type=Path, required=True)
    finalize_cmd.add_argument("--teacher-input-train", type=Path, required=True)
    finalize_cmd.add_argument("--teacher-input-dev", type=Path, required=True)
    finalize_cmd.add_argument("--teacher-manifest", type=Path, required=True)
    finalize_cmd.add_argument("--teacher-train", type=Path, required=True)
    finalize_cmd.add_argument("--teacher-dev", type=Path, required=True)
    finalize_cmd.add_argument("--replay-selected", type=Path, required=True)
    finalize_cmd.add_argument("--output-dir", type=Path, required=True)
    finalize_cmd.set_defaults(function=finalize)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
