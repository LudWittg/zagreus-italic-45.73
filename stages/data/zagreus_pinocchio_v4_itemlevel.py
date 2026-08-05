#!/usr/bin/env python3
"""Rebuild Pinocchio culture/general rows with item-level ITALIC screening.

Both formerly family-excluded source splits are ingested.  Family names never
decide contamination: each structurally valid, ITALIC-relevant item is screened
by the locked recursive-normalization, lexical, containment, and MiniLM cosine
rules.  Pinocchio's source answer is retained only as a descriptive reference;
under I5 these rows are KD-only unless stronger row-level key provenance is
added prospectively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow.parquet as pq

import zagreus_exam_bank_v1 as exam
import zagreus_final_instrument as screens
import zagreus_pinocchio_v3_data as v3


PROTOCOL = "zagreus-pinocchio-item-level-v4"
SOURCE_REVISION = "83c6a063c7dd8872e52b4d10504ddb87ed24f0e5"
SOURCE_LICENSE = "apache-2.0"
EXPECTED_PARQUET_SHA256 = {
    "cultura": "848a914377c5311fb0bf143e1d6e2804adf335cdf70d91841d837f6e469bf2d4",
    "generale": "1ca474c025c4d1a815642fdc326b2d914220a625dca6a8cef9ec78462f0b2e6a",
}
EXPECTED_OFFICIAL_SHA256 = screens.EXPECTED_OFFICIAL_SHA256
SEED = "20260804-pinocchio-item-level-v4"


DOMAIN_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "art_history",
        (
            "storia_dell_arte",
            "storia_delle_arti",
            "arte_e_",
            "_arte",
            "artist",
            "architett",
            "archeolog",
            "restauro",
            "beni_cultural",
            "muse",
        ),
    ),
    ("literature", ("letteratur", "letterari", "poesia", "analisi_del_testo")),
    ("tourism", ("turis", "guide_turistic")),
    ("geography", ("geograf",)),
    ("history", ("storia", "storico")),
    (
        "civic_education",
        (
            "educazione_civica",
            "educazionecivica",
            "cittadin",
            "costituz",
            "unione_europe",
            "pubblica_amministrazione",
            "enti_locali",
            "ordinamento",
            "politiche_comunitarie",
        ),
    ),
    (
        "synonyms_and_antonyms",
        ("sinonim", "contrari", "capacita_verbale", "comprensione_verbale"),
    ),
    ("orthography", ("ortograf", "otograf")),
    ("syntax", ("sintass", "completamento_frasi")),
    ("morphology", ("morfolog", "coniug", "verbi")),
    (
        "lexicon",
        (
            "lessico",
            "vocabol",
            "significato",
            "lingua_italiana",
            "linguaitaliana",
            "italiano",
            "grammatica",
        ),
    ),
)


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


def stable_key(row: Mapping[str, Any], purpose: str) -> str:
    return hashlib.sha256(
        f"{SEED}:{purpose}:{row['source_split']}:{row['id']}".encode("utf-8")
    ).hexdigest()


def ascii_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def map_domain(category: Any, macro: Any) -> str | None:
    value = f"{ascii_key(category)} {ascii_key(macro)}"
    if "attual" in value:
        return None
    for domain, fragments in DOMAIN_RULES:
        if any(fragment in value for fragment in fragments):
            return domain
    return None


def source_url(family: str) -> str:
    return (
        "https://huggingface.co/datasets/efederici/pinocchio/resolve/"
        f"{SOURCE_REVISION}/text/{family}-00000-of-00001.parquet"
    )


def inventory(paths: Mapping[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counters: Counter[str] = Counter()
    by_family: Counter[str] = Counter()
    by_domain: Counter[str] = Counter()
    source_rows: list[dict[str, Any]] = []
    files = []
    for family in ("cultura", "generale"):
        path = paths[family]
        actual = exam.sha256_file(path)
        expected = EXPECTED_PARQUET_SHA256[family]
        if actual != expected:
            raise ValueError(f"Parquet hash mismatch for {family}: {actual} != {expected}")
        table = pq.read_table(path)
        files.append(
            {
                "family": family,
                "path": str(path),
                "url": source_url(family),
                "rows": table.num_rows,
                "sha256": actual,
            }
        )
        for index, raw in enumerate(table.to_pylist()):
            counters["source_rows"] += 1
            category = raw.get("category")
            macro = raw.get("macro")
            domain = map_domain(category, macro)
            if domain is None:
                counters[
                    "excluded_current_events"
                    if "attual" in f"{ascii_key(category)} {ascii_key(macro)}"
                    else "excluded_not_italic_mapped"
                ] += 1
                continue
            parsed = v3.four_option_record(raw, family, index)
            if parsed is None:
                counters["excluded_structural_or_option_count"] += 1
                continue
            source_answer = int(parsed.pop("gold"))
            parsed["domain"] = domain
            parsed["source"] = "efederici/pinocchio"
            parsed["source_split"] = family
            parsed["license"] = SOURCE_LICENSE
            parsed["source_revision"] = SOURCE_REVISION
            parsed["source_answer_index"] = source_answer
            parsed["label_contract"] = {
                "campaign_gold": False,
                "source_answer_role": "difficulty_reference_only",
                "training_role": "teacher_kd_only_without_filtering",
                "gold_ce_authorized": False,
                "reason": "I5 requires official keys or deterministic source derivation",
            }
            parsed["normalized_question_hash"] = exam.normalized_question_hash(
                parsed["question"]
            )
            source_rows.append(parsed)
            counters["structurally_valid_italic_mapped"] += 1
            by_family[family] += 1
            by_domain[domain] += 1

    unique: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for row in sorted(source_rows, key=lambda value: stable_key(value, "dedup")):
        signature = str(row["normalized_question_hash"])
        if signature in seen_questions:
            counters["excluded_global_normalized_question_duplicate"] += 1
            continue
        seen_questions.add(signature)
        unique.append(row)
    unique.sort(key=lambda value: stable_key(value, "output"))
    manifest = {
        "protocol": PROTOCOL,
        "stage": "item_inventory_before_italic_screen",
        "source": {
            "dataset_id": "efederici/pinocchio",
            "revision": SOURCE_REVISION,
            "license": SOURCE_LICENSE,
            "families_ingested_without_family_level_contamination_exclusion": [
                "cultura",
                "generale",
            ],
            "files": files,
        },
        "mapping": {
            "target_domains": [domain for domain, _ in DOMAIN_RULES],
            "rules": {domain: list(fragments) for domain, fragments in DOMAIN_RULES},
            "current_events_excluded": True,
        },
        "counts": {
            **dict(sorted(counters.items())),
            "unique_before_italic_screen": len(unique),
            "by_family_before_italic_screen": dict(sorted(by_family.items())),
            "by_domain_before_italic_screen": dict(sorted(by_domain.items())),
        },
        "i5": {
            "source_answers_preserved_as_descriptive_references": True,
            "source_answers_used_as_campaign_gold": False,
            "teacher_filtering_authorized": False,
            "gold_ce_authorized": False,
            "future_training_contract": "label every selected row with a prospectively pinned teacher; KD-only; no agreement filtering",
        },
    }
    return unique, manifest


def descriptive_sample(rows: Sequence[dict[str, Any]], per_domain: int = 100) -> list[dict[str, Any]]:
    by_domain: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["domain"])].append(row)
    selected = []
    for domain in sorted(by_domain):
        ordered = sorted(by_domain[domain], key=lambda row: stable_key(row, "reference"))
        selected.extend(ordered[:per_domain])
    return sorted(selected, key=lambda row: (str(row["domain"]), stable_key(row, "sample")))


def build(
    paths: Mapping[str, Path],
    official_path: Path,
    embedding_model_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if exam.sha256_file(official_path) != EXPECTED_OFFICIAL_SHA256:
        raise ValueError("Official canonical hash changed")
    rows, inventory_manifest = inventory(paths)
    private = output_dir / "private"
    inventory_path = private / "inventory.jsonl"
    write_jsonl(inventory_path, rows)
    inventory_manifest["output"] = {
        "path": str(inventory_path),
        "sha256": exam.sha256_file(inventory_path),
    }
    write_json(output_dir / "inventory_manifest.json", inventory_manifest)

    official = exam.load_official_rows(official_path)
    lexical_retained, lexical_rejected = screens.lexical_screen_explicit(rows, official)
    encoder = screens.sentence_encoder(embedding_model_path)
    retained, embedding_rejected, cache = screens.embedding_screen_explicit(
        lexical_retained, official, encoder
    )
    tail_retained, tail_lexical_rejected = screens.lexical_screen_explicit(
        retained, official
    )
    tail_embedding_rejected = [
        row
        for row in tail_retained
        if float(row["final_instrument_embedding_screen"]["max_cosine"])
        >= exam.EMBEDDING_THRESHOLD
    ]
    if tail_lexical_rejected or tail_embedding_rejected:
        raise RuntimeError("Mechanical zero-removal tail failed")
    retained = sorted(tail_retained, key=lambda row: stable_key(row, "retained"))
    rejected = sorted(
        lexical_rejected + embedding_rejected,
        key=lambda row: stable_key(row, "rejected"),
    )
    retained_path = output_dir / "pinocchio_itemlevel_kd_pool.jsonl"
    rejected_path = private / "decontamination_rejections.jsonl"
    sample_path = output_dir / "difficulty_reference_sample.jsonl"
    write_jsonl(retained_path, retained)
    write_jsonl(rejected_path, rejected)
    sample = descriptive_sample(retained)
    write_jsonl(sample_path, sample)

    import numpy as np

    score_path = private / "embedding_scores.npz"
    np.savez_compressed(score_path, **cache)
    lexical_trigger_counts: Counter[str] = Counter()
    for row in lexical_rejected:
        for trigger in row["final_instrument_lexical_screen"]["triggers"]:
            lexical_trigger_counts[str(trigger["kind"])] += 1
    embedding_surface_counts = Counter(
        str(row["final_instrument_embedding_screen"]["matched_representation"])
        for row in embedding_rejected
    )
    by_domain = Counter(str(row["domain"]) for row in retained)
    by_family = Counter(str(row["source_split"]) for row in retained)
    exact_collisions = lexical_trigger_counts["official_normalized_exact"]
    manifest = {
        "protocol": PROTOCOL,
        "status": "COMPLETE_ITEM_LEVEL_SCREEN_ZERO_RETAINED_TRIGGERS",
        "official_italic_use": {
            "purpose": "decontamination_only",
            "model_evaluation": False,
            "checkpoint_selection": False,
            "rows": len(official),
            "sha256": EXPECTED_OFFICIAL_SHA256,
            "official_text_written_to_outputs": False,
        },
        "screen": {
            "family_level_exclusion": False,
            "recursive_html_normalization": True,
            "normalized_exact": 1.0,
            "word_3_shingle_jaccard": 0.70,
            "content_word_containment": 0.80,
            "content_word_minimum": 8,
            "embedding_model": exam.EMBEDDING_MODEL,
            "embedding_revision": exam.EMBEDDING_REVISION,
            "embedding_cosine": exam.EMBEDDING_THRESHOLD,
            "embedding_surfaces": ["question", "question_plus_sorted_options"],
            "review_mode": "zero_human_zero_model_adjudication",
        },
        "counts": {
            "input_after_structural_mapping_and_dedup": len(rows),
            "lexical_rejected": len(lexical_rejected),
            "lexical_trigger_counts": dict(sorted(lexical_trigger_counts.items())),
            "official_normalized_exact_collisions": exact_collisions,
            "embedding_rejected": len(embedding_rejected),
            "embedding_rejection_surfaces": dict(sorted(embedding_surface_counts.items())),
            "retained": len(retained),
            "retained_by_family": dict(sorted(by_family.items())),
            "retained_by_domain": dict(sorted(by_domain.items())),
            "mechanical_tail_removals": 0,
            "difficulty_reference_rows": len(sample),
        },
        "maximum_retained": {
            "shingle_jaccard": max(
                (
                    row["final_instrument_lexical_screen"][
                        "max_official_shingle_jaccard"
                    ]
                    for row in retained
                ),
                default=0.0,
            ),
            "content_word_containment": max(
                (
                    row["final_instrument_lexical_screen"][
                        "max_official_content_word_containment"
                    ]
                    for row in retained
                ),
                default=0.0,
            ),
            "embedding_cosine": max(
                (
                    row["final_instrument_embedding_screen"]["max_cosine"]
                    for row in retained
                ),
                default=0.0,
            ),
        },
        "i5": inventory_manifest["i5"],
        "outputs": {
            "kd_pool": {
                "path": str(retained_path),
                "rows": len(retained),
                "sha256": exam.sha256_file(retained_path),
                "contains_campaign_gold": False,
            },
            "difficulty_reference_sample": {
                "path": str(sample_path),
                "rows": len(sample),
                "sha256": exam.sha256_file(sample_path),
                "selection_authority": False,
            },
            "rejections": {
                "path": str(rejected_path),
                "rows": len(rejected),
                "sha256": exam.sha256_file(rejected_path),
            },
            "embedding_scores": {
                "path": str(score_path),
                "sha256": exam.sha256_file(score_path),
            },
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cultura", type=Path, required=True)
    parser.add_argument("--generale", type=Path, required=True)
    parser.add_argument("--official", type=Path, default=screens.OFFICIAL_PATH)
    parser.add_argument(
        "--embedding-model", type=Path, default=screens.EMBEDDING_MODEL_PATH
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(
        {"cultura": args.cultura, "generale": args.generale},
        args.official,
        args.embedding_model,
        args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
