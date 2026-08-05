#!/usr/bin/env python3
"""Build the fresh, zero-human FINAL-v1 certification instrument.

The command is intentionally staged.  ``inventory`` performs only frozen-hash
and disjointness work, ``decontaminate`` performs the official-ITALIC screens,
and ``allocate`` is the only stage that writes the evaluation surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

import zagreus_exam_bank_v1 as exam
from zagreus_scaled_kd_data import question_signature


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "results" / "zagreus_final_v1" / "instrument"
READOUT_ROOT = (
    ROOT / "results" / "zagreus_readout_v2_stage0" / "zagreus_readout_v2" / "data"
)
OFFICIAL_PATH = ROOT / "results" / "italic_canonical.jsonl"
EMBEDDING_MODEL_PATH = (
    ROOT
    / "results"
    / "zagreus_v4"
    / "model_cache"
    / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
    / "snapshots"
    / exam.EMBEDDING_REVISION
)

ALLOCATION_SEED = "20260803-final-instrument-v1"
MAX_ROWS = 12_600
MIN_ROWS = 10_000
MAX_REMOVAL_RATE = 0.05

OFFICIAL_COUNTS = {
    "art_history": 980,
    "civic_education": 973,
    "geography": 979,
    "history": 978,
    "lexicon": 979,
    "literature": 984,
    "morphology": 140,
    "orthography": 971,
    "synonyms_and_antonyms": 971,
    "syntax": 973,
    "tourism": 980,
}

POOL_FILES = {
    "teacher_calibration.jsonl": "bebb4780a84979e8a737a49a8f97cdb4b16b5574656eec183d83fd7795a3d84a",
    "lr_calibration.jsonl": "324e411cd390933aaf9f0c9ff7dcc4e1372e95edc79bad5a121bc9decff44ccb",
    "proxy_dev.jsonl": "a5cda7bb678c67e160b8b30d28c6e93fca8245717f569546bf63284527e73425",
    "final_holdout.jsonl": "3a4171ebb764a7f5447bd88f5f0680c58c0d67b2d821fdb722f7077a30441993",
    "pilot_train.jsonl": "0fafdd170cc83d2ec5341fcc02e3f6a1209135485bf71f862a4abf594e412db5",
}

EXCLUSION_FILES = {
    ROOT / "results" / "zagreus_exam_bank_v1" / "p4" / "locked_proxy.jsonl":
        "38dc439226c31cd13b1b42dc3587e9deee8d665d8b9fe4a320d9f431989dffec",
    ROOT / "results" / "zagreus_scaled_kd_v1" / "artifact_download" / "evidence" / "data" / "train.jsonl":
        "915478983c2662c91da306b9ed3bc115528375d67dadfb270a5ae43d60f37760",
    ROOT / "results" / "zagreus_scaled_kd_v1" / "artifact_download" / "evidence" / "data" / "dev.jsonl":
        "ef32a7a0332577b6d7967b2a0e0ceb21556cadd210328a3c5dd6bd3d3790739c",
    READOUT_ROOT / "geometry_probe.jsonl":
        "972b3b7659f347804253f107da9cd745276d2d0a623926d7bf925e7d43a6b61c",
    READOUT_ROOT / "geometry_eval.jsonl":
        "89df64472eff61d273e573a6f62acce3236e2a6c46e07297a47ba1248cee8620",
}

EXPECTED_OFFICIAL_SHA256 = "ca877846f19a6d781ba151382e4f43b10efad1a7e375b5e0c50b047a3917e0af"
PROJECTED_KEYS = (
    "id", "source", "group_id", "context", "question", "options", "gold", "domain"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def verify_hash(path: Path, expected: str) -> str:
    actual = exam.sha256_file(path)
    if actual != expected:
        raise ValueError(f"Frozen hash mismatch for {path}: {actual} != {expected}")
    return actual


def validate_row(row: Mapping[str, Any]) -> None:
    required = {"id", "group_id", "question", "options", "gold", "domain"}
    missing = required - set(row)
    if missing:
        raise ValueError(f"Row {row.get('id')} is missing {sorted(missing)}")
    options = list(row["options"])
    if len(options) != 4:
        raise ValueError(f"Row {row['id']} does not have four options")
    normalized = [exam.normalize_text(value) for value in options]
    if any(not value for value in normalized) or len(set(normalized)) != 4:
        raise ValueError(f"Row {row['id']} has empty or duplicate normalized options")
    if not 0 <= int(row["gold"]) < 4:
        raise ValueError(f"Row {row['id']} has an invalid gold index")


def load_exclusions() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    files = []
    for path, expected in EXCLUSION_FILES.items():
        verify_hash(path, expected)
        current = read_jsonl(path)
        rows.extend(current)
        files.append({"path": str(path), "sha256": expected, "rows": len(current)})
    return {
        "ids": {str(row["id"]) for row in rows if "id" in row},
        "groups": {str(row["group_id"]) for row in rows if row.get("group_id")},
        "signatures": {question_signature(row) for row in rows},
        "files": files,
    }


def weighted_quotas(total: int) -> dict[str, int]:
    weight_sum = sum(OFFICIAL_COUNTS.values())
    raw = {domain: total * weight / weight_sum for domain, weight in OFFICIAL_COUNTS.items()}
    quotas = {domain: math.floor(value) for domain, value in raw.items()}
    remaining = total - sum(quotas.values())
    order = sorted(
        OFFICIAL_COUNTS,
        key=lambda domain: (
            -(raw[domain] - quotas[domain]),
            hashlib.sha256(f"{ALLOCATION_SEED}:remainder:{domain}".encode()).hexdigest(),
        ),
    )
    for domain in order[:remaining]:
        quotas[domain] += 1
    if sum(quotas.values()) != total:
        raise AssertionError("Weighted quota arithmetic failed")
    return quotas


def maximum_feasible_n(availability: Mapping[str, int]) -> tuple[int, dict[str, int]]:
    for total in range(MAX_ROWS, 99, -100):
        quotas = weighted_quotas(total)
        if all(int(availability.get(domain, 0)) >= quota for domain, quota in quotas.items()):
            return total, quotas
    return 0, {}


def inventory(output_dir: Path = OUTPUT_ROOT) -> dict[str, Any]:
    exclusions = load_exclusions()
    source_rows: list[dict[str, Any]] = []
    sources = []
    invalid_source_rows: Counter[str] = Counter()
    for name, expected in POOL_FILES.items():
        path = READOUT_ROOT / name
        verify_hash(path, expected)
        rows = read_jsonl(path)
        for row in rows:
            try:
                validate_row(row)
            except ValueError as error:
                invalid_source_rows[str(error).split(" has ")[-1]] += 1
                continue
            source_rows.append(row)
        sources.append({
            "path": str(path),
            "sha256": expected,
            "rows": len(rows),
            "structurally_valid_rows": sum(
                1 for row in rows
                if len(list(row.get("options", ()))) == 4
                and len({exam.normalize_text(value) for value in row.get("options", ())}) == 4
                and all(exam.normalize_text(value) for value in row.get("options", ()))
                and 0 <= int(row.get("gold", -1)) < 4
            ),
        })

    rejection_reasons: Counter[str] = Counter()
    fresh: list[dict[str, Any]] = []
    for row in source_rows:
        if str(row["domain"]) == "current_events":
            rejection_reasons["current_events_excluded"] += 1
            continue
        reasons = []
        if str(row["id"]) in exclusions["ids"]:
            reasons.append("id")
        if str(row["group_id"]) in exclusions["groups"]:
            reasons.append("group_id")
        if question_signature(row) in exclusions["signatures"]:
            reasons.append("question_signature")
        if reasons:
            for reason in reasons:
                rejection_reasons[f"exclusion:{reason}"] += 1
            continue
        fresh.append(dict(row))

    ids = [str(row["id"]) for row in fresh]
    groups = [str(row["group_id"]) for row in fresh]
    signatures = [question_signature(row) for row in fresh]
    if len(ids) != len(set(ids)) or len(groups) != len(set(groups)) or len(signatures) != len(set(signatures)):
        raise ValueError("Fresh inventory is not unique by ID, group, and question signature")
    availability = Counter(str(row["domain"]) for row in fresh)
    prospective_n, prospective_quotas = maximum_feasible_n(availability)

    inventory_path = output_dir / "private" / "fresh_inventory.jsonl"
    write_jsonl(inventory_path, fresh)
    manifest = {
        "protocol": "zagreus-final-v1-fresh-inventory",
        "status": "INVENTORIED_AWAITING_ITALIC_SCREEN",
        "source_files": sources,
        "exclusion_files": exclusions["files"],
        "counts": {
            "source_rows": sum(int(source["rows"]) for source in sources),
            "structurally_valid_source_rows": len(source_rows),
            "fresh_rows": len(fresh),
            "fresh_by_domain": dict(sorted(availability.items())),
            "rejections": dict(sorted(rejection_reasons.items())),
            "invalid_source_rows": dict(sorted(invalid_source_rows.items())),
            "unique_ids": len(set(ids)),
            "unique_groups": len(set(groups)),
            "unique_question_signatures": len(set(signatures)),
        },
        "prospective_allocation_before_italic_screen": {
            "rows": prospective_n,
            "quotas": prospective_quotas,
            "minimum_paid_campaign_rows": MIN_ROWS,
            "maximum_rows": MAX_ROWS,
        },
        "output": {"path": str(inventory_path), "sha256": exam.sha256_file(inventory_path)},
        "official_italic_used_for_model_evaluation": False,
    }
    write_json(output_dir / "inventory_manifest.json", manifest)
    return manifest


def lexical_screen_explicit(
    rows: Sequence[Mapping[str, Any]], official: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    official_hashes = {str(row["question_hash"]) for row in official}
    official_shingles = [exam.word_shingles(str(row["question"])) for row in official]
    official_words = [frozenset(exam.content_words(str(row["question"]))) for row in official]
    shingle_inverted = exam.build_inverted(official_shingles)
    word_inverted = exam.build_inverted(official_words)
    retained, rejected = [], []
    for original in rows:
        row = dict(original)
        triggers: list[dict[str, Any]] = []
        question_hash = exam.normalized_question_hash(row["question"])
        if question_hash in official_hashes:
            triggers.append({"kind": "official_normalized_exact", "score": 1.0})

        candidate_shingles = exam.word_shingles(str(row["question"]))
        candidates: set[int] = set()
        for token in candidate_shingles:
            candidates.update(shingle_inverted.get(token, ()))
        max_jaccard, jaccard_index = 0.0, None
        for index in candidates:
            score = exam.jaccard(candidate_shingles, official_shingles[index])
            if score > max_jaccard:
                max_jaccard, jaccard_index = score, index
        if max_jaccard >= 0.70 and jaccard_index is not None:
            triggers.append({
                "kind": "word_3_shingle_jaccard",
                "score": max_jaccard,
                "matched_official_question_hash": official[jaccard_index]["question_hash"],
            })

        candidate_words = frozenset(exam.content_words(str(row["question"])))
        max_containment, containment_index = 0.0, None
        if len(candidate_words) >= 8:
            candidates = set()
            for token in candidate_words:
                candidates.update(word_inverted.get(token, ()))
            for index in candidates:
                if min(len(candidate_words), len(official_words[index])) < 8:
                    continue
                score = exam.containment(candidate_words, official_words[index])
                if score > max_containment:
                    max_containment, containment_index = score, index
            if max_containment >= 0.80 and containment_index is not None:
                triggers.append({
                    "kind": "content_word_containment",
                    "score": max_containment,
                    "matched_official_question_hash": official[containment_index]["question_hash"],
                })
        row["final_instrument_lexical_screen"] = {
            "normalized_question_hash": question_hash,
            "max_official_shingle_jaccard": max_jaccard,
            "max_official_content_word_containment": max_containment,
            "triggers": triggers,
        }
        (rejected if triggers else retained).append(row)
    return retained, rejected


def sentence_encoder(model_path: Path) -> Callable[[Sequence[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer

    if model_path.name != exam.EMBEDDING_REVISION:
        raise ValueError(f"Expected embedding revision {exam.EMBEDDING_REVISION}")
    model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)

    def encode(values: Sequence[str]) -> np.ndarray:
        return model.encode(
            list(values),
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)

    return encode


def embedding_screen_explicit(
    rows: Sequence[Mapping[str, Any]],
    official: Sequence[Mapping[str, Any]],
    encode: Callable[[Sequence[str]], np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    official_pairs = [exam.semantic_texts(row) for row in official]
    candidate_pairs = [exam.semantic_texts(row) for row in rows]
    official_q = encode([value[0] for value in official_pairs])
    official_full = encode([value[1] for value in official_pairs])
    candidate_q = encode([value[0] for value in candidate_pairs])
    candidate_full = encode([value[1] for value in candidate_pairs])
    q_scores, q_indices = exam.nearest_cosine(candidate_q, official_q, 256)
    full_scores, full_indices = exam.nearest_cosine(candidate_full, official_full, 256)

    retained, rejected = [], []
    for position, original in enumerate(rows):
        row = dict(original)
        use_full = bool(full_scores[position] > q_scores[position])
        score = float(full_scores[position] if use_full else q_scores[position])
        official_index = int(full_indices[position] if use_full else q_indices[position])
        row["final_instrument_embedding_screen"] = {
            "embedding_model": exam.EMBEDDING_MODEL,
            "embedding_revision": exam.EMBEDDING_REVISION,
            "max_cosine": score,
            "matched_representation": "question_plus_sorted_options" if use_full else "question",
            "matched_official_question_hash": official[official_index]["question_hash"],
            "threshold": exam.EMBEDDING_THRESHOLD,
            "triggered": score >= exam.EMBEDDING_THRESHOLD,
        }
        (rejected if score >= exam.EMBEDDING_THRESHOLD else retained).append(row)
    cache = {
        "q_scores": q_scores,
        "q_indices": q_indices,
        "full_scores": full_scores,
        "full_indices": full_indices,
    }
    return retained, rejected, cache


def decontaminate(
    output_dir: Path = OUTPUT_ROOT,
    official_path: Path = OFFICIAL_PATH,
    model_path: Path = EMBEDDING_MODEL_PATH,
    encode: Callable[[Sequence[str]], np.ndarray] | None = None,
) -> dict[str, Any]:
    inventory_manifest = read_json(output_dir / "inventory_manifest.json")
    inventory_path = Path(inventory_manifest["output"]["path"])
    verify_hash(inventory_path, inventory_manifest["output"]["sha256"])
    verify_hash(official_path, EXPECTED_OFFICIAL_SHA256)
    rows = read_jsonl(inventory_path)
    official = exam.load_official_rows(official_path)

    lexical_retained, lexical_rejected = lexical_screen_explicit(rows, official)
    encode = encode or sentence_encoder(model_path)
    retained, embedding_rejected, cache = embedding_screen_explicit(
        lexical_retained, official, encode
    )
    rejected = lexical_rejected + embedding_rejected
    removal_rate = len(rejected) / max(1, len(rows))

    # Mechanical tail: rederive every lexical classification and re-evaluate
    # the pinned embedding maxima.  No reviewer or model changes a row.
    lexical_tail_retained, lexical_tail_rejected = lexical_screen_explicit(retained, official)
    embedding_tail_rejected = [
        row for row in lexical_tail_retained
        if float(row["final_instrument_embedding_screen"]["max_cosine"])
        >= exam.EMBEDDING_THRESHOLD
    ]
    tail_removals = len(lexical_tail_rejected) + len(embedding_tail_rejected)

    private = output_dir / "private"
    retained_path = private / "decontaminated.jsonl"
    rejected_path = private / "decontamination_rejections.jsonl"
    score_path = private / "embedding_scores.npz"
    write_jsonl(retained_path, lexical_tail_retained)
    write_jsonl(rejected_path, rejected)
    np.savez_compressed(score_path, **cache)
    status = (
        "STOP_REINVENTORY_REMOVAL_ABOVE_5PCT"
        if removal_rate > MAX_REMOVAL_RATE
        else "PASSED_MECHANICAL_ZERO_REMOVAL_TAIL"
        if tail_removals == 0
        else "FAILED_TAIL_RECHECK"
    )
    manifest = {
        "protocol": "zagreus-final-v1-explicit-italic-decontamination",
        "status": status,
        "input": {"path": str(inventory_path), "sha256": exam.sha256_file(inventory_path)},
        "official": {"path": str(official_path), "sha256": EXPECTED_OFFICIAL_SHA256, "rows": len(official)},
        "thresholds": {
            "normalized_exact": 1.0,
            "word_3_shingle_jaccard": 0.70,
            "content_word_containment": 0.80,
            "content_word_minimum": 8,
            "embedding_cosine": exam.EMBEDDING_THRESHOLD,
            "maximum_first_pass_removal_rate": MAX_REMOVAL_RATE,
        },
        "counts": {
            "input": len(rows),
            "lexical_rejected": len(lexical_rejected),
            "embedding_rejected": len(embedding_rejected),
            "total_rejected": len(rejected),
            "retained": len(retained),
            "removal_rate": removal_rate,
            "mechanical_tail_removals": tail_removals,
            "retained_by_domain": dict(sorted(Counter(row["domain"] for row in retained).items())),
        },
        "maximum_retained": {
            "shingle_jaccard": max(
                (row["final_instrument_lexical_screen"]["max_official_shingle_jaccard"] for row in retained),
                default=0.0,
            ),
            "content_word_containment": max(
                (row["final_instrument_lexical_screen"]["max_official_content_word_containment"] for row in retained),
                default=0.0,
            ),
            "embedding_cosine": max(
                (row["final_instrument_embedding_screen"]["max_cosine"] for row in retained),
                default=0.0,
            ),
        },
        "review_mode": "zero_human_zero_model_adjudication_mechanical_thresholds_only",
        "outputs": {
            "retained": {"path": str(retained_path), "sha256": exam.sha256_file(retained_path)},
            "rejected": {"path": str(rejected_path), "sha256": exam.sha256_file(rejected_path)},
            "embedding_scores": {"path": str(score_path), "sha256": exam.sha256_file(score_path)},
        },
    }
    write_json(output_dir / "decontamination_manifest.json", manifest)
    return manifest


def stable_selection_key(row: Mapping[str, Any], purpose: str) -> str:
    return hashlib.sha256(
        f"{ALLOCATION_SEED}:{purpose}:{row['domain']}:{row['group_id']}:{row['id']}".encode()
    ).hexdigest()


def select_weighted(rows: Sequence[dict[str, Any]], total: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    quotas = weighted_quotas(total)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["domain"])].append(row)
    selected: list[dict[str, Any]] = []
    for domain, quota in quotas.items():
        candidates = sorted(by_domain[domain], key=lambda row: stable_selection_key(row, "quota"))
        if len(candidates) < quota:
            raise ValueError(f"Insufficient {domain} rows: {len(candidates)} < {quota}")
        selected.extend(candidates[:quota])
    selected.sort(key=lambda row: stable_selection_key(row, "final"))
    return selected, quotas


def exclusion_overlap_counts(
    rows: Sequence[dict[str, Any]], exclusions: Mapping[str, set[Any]]
) -> dict[str, int]:
    return {
        "id": sum(str(row["id"]) in exclusions["ids"] for row in rows),
        "group_id": sum(str(row["group_id"]) in exclusions["groups"] for row in rows),
        "question_signature": sum(
            question_signature(row) in exclusions["signatures"] for row in rows
        ),
    }


def projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in PROJECTED_KEYS if key in row}


def allocate(output_dir: Path = OUTPUT_ROOT) -> dict[str, Any]:
    decontamination = read_json(output_dir / "decontamination_manifest.json")
    if decontamination.get("status") != "PASSED_MECHANICAL_ZERO_REMOVAL_TAIL":
        availability = {
            str(domain): int(count)
            for domain, count in decontamination.get("counts", {}).get(
                "retained_by_domain", {}
            ).items()
        }
        maximum_n, quotas = maximum_feasible_n(availability)
        gate = {
            "protocol": "zagreus-final-v1-gate-f0",
            "passed": False,
            "reason": decontamination.get("status"),
            "first_pass_removal_rate": decontamination.get("counts", {}).get(
                "removal_rate"
            ),
            "maximum_feasible_weighted_rows_after_screen": maximum_n,
            "maximum_feasible_quotas": quotas,
            "minimum_paid_campaign_rows": MIN_ROWS,
            "gpu_spend_authorized": False,
            "official_italic_used_for_model_evaluation": False,
        }
        write_json(output_dir / "GATE_F0.json", gate)
        raise ValueError(f"Decontamination gate is not passed: {decontamination.get('status')}")
    retained_path = Path(decontamination["outputs"]["retained"]["path"])
    verify_hash(retained_path, decontamination["outputs"]["retained"]["sha256"])
    rows = read_jsonl(retained_path)
    availability = Counter(str(row["domain"]) for row in rows)
    total, quotas = maximum_feasible_n(availability)
    if total < MIN_ROWS:
        raise ValueError(f"Fresh instrument below paid-campaign floor: {total} < {MIN_ROWS}")

    selected, realized_quotas = select_weighted(rows, total)
    if realized_quotas != quotas:
        raise AssertionError("Allocation quota derivation changed within one run")

    exclusions = load_exclusions()
    official = exam.load_official_rows(OFFICIAL_PATH)
    official_signatures = {question_signature(row) for row in official}
    ids = [str(row["id"]) for row in selected]
    groups = [str(row["group_id"]) for row in selected]
    signatures = [question_signature(row) for row in selected]
    overlaps = exclusion_overlap_counts(selected, exclusions)
    checks = {
        "rows_at_least_10000": len(selected) >= MIN_ROWS,
        "rows_equal_computed_n": len(selected) == total,
        "all_sector_quotas_met": Counter(row["domain"] for row in selected) == Counter(quotas),
        "unique_ids": len(ids) == len(set(ids)),
        "unique_groups": len(groups) == len(set(groups)),
        "unique_question_signatures": len(signatures) == len(set(signatures)),
        "excluded_id_overlap_zero": overlaps["id"] == 0,
        "excluded_group_overlap_zero": overlaps["group_id"] == 0,
        "excluded_signature_overlap_zero": overlaps["question_signature"] == 0,
        "official_signature_overlap_zero": not (set(signatures) & official_signatures),
        "no_italic_ids": not any(value.casefold().startswith("italic:") for value in ids),
        "mechanical_tail_zero_removal": decontamination["counts"]["mechanical_tail_removals"] == 0,
        "no_retained_embedding_trigger": not any(
            bool(row["final_instrument_embedding_screen"]["triggered"]) for row in selected
        ),
    }
    for row in selected:
        validate_row(row)
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise ValueError(f"GATE-F0 allocation checks failed: {failed}")

    instrument_path = output_dir / "certification_instrument.jsonl"
    write_jsonl(instrument_path, [projection(row) for row in selected])
    instrument_sha = exam.sha256_file(instrument_path)
    manifest = {
        "protocol": "zagreus-final-v1-fresh-certification-instrument",
        "allocation_seed": ALLOCATION_SEED,
        "rows": len(selected),
        "quotas": quotas,
        "availability_after_decontamination": dict(sorted(availability.items())),
        "official_weights": OFFICIAL_COUNTS,
        "data": {"path": str(instrument_path), "sha256": instrument_sha},
        "decontamination_manifest_sha256": exam.sha256_file(
            output_dir / "decontamination_manifest.json"
        ),
        "checks": checks,
        "baseline_measured": False,
        "candidate_scored": False,
        "official_italic_used_for_model_evaluation": False,
    }
    write_json(output_dir / "instrument_manifest.json", manifest)
    round_trip = read_jsonl(instrument_path)
    gate = {
        "protocol": "zagreus-final-v1-gate-f0",
        "passed": all(checks.values()) and len(round_trip) == len(selected),
        "rows": len(selected),
        "instrument_sha256": instrument_sha,
        "manifest_sha256": exam.sha256_file(output_dir / "instrument_manifest.json"),
        "checks": checks | {"manifest_row_count_round_trip": len(round_trip) == len(selected)},
        "gpu_spend_authorized": False,
        "reason": "local gate only; machine budget authorization is a separate prospective step",
    }
    write_json(output_dir / "GATE_F0.json", gate)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("inventory", "decontaminate", "allocate", "all"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--official", type=Path, default=OFFICIAL_PATH)
    parser.add_argument("--embedding-model", type=Path, default=EMBEDDING_MODEL_PATH)
    args = parser.parse_args()
    if args.stage in {"inventory", "all"}:
        print(json.dumps(inventory(args.output_dir), ensure_ascii=False), flush=True)
    if args.stage in {"decontaminate", "all"}:
        print(
            json.dumps(
                decontaminate(args.output_dir, args.official, args.embedding_model),
                ensure_ascii=False,
            ),
            flush=True,
        )
    if args.stage in {"allocate", "all"}:
        print(json.dumps(allocate(args.output_dir), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
