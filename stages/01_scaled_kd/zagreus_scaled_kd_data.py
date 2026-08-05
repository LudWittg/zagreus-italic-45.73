"""Build the clean, sector-targeted source dataset for scaled Zagreus KD.

This module deliberately has no ITALIC input.  It creates a fresh, source-group
disjoint train/dev split from Kaikki Italian Wiktionary, Universal Dependencies
Italian ISDT, and provenance-bearing Italian Wikipedia candidates.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import zagreus_language_synthetic_data as language


SEED = 20260801
TRAIN_QUOTAS = {
    "lexicon": 800,
    "morphology": 800,
    "orthography": 1600,
    "synonyms_and_antonyms": 700,
    "syntax": 1600,
    "art_history": 800,
    "civic_education": 800,
    "current_events": 70,
    "geography": 800,
    "history": 800,
    "literature": 800,
    "tourism": 800,
}
DEV_QUOTAS = {
    "lexicon": 100,
    "morphology": 100,
    "orthography": 200,
    "synonyms_and_antonyms": 100,
    "syntax": 200,
    "art_history": 100,
    "civic_education": 100,
    "current_events": 8,
    "geography": 100,
    "history": 100,
    "literature": 100,
    "tourism": 100,
}
LANGUAGE_DOMAINS = {
    "lexicon",
    "morphology",
    "orthography",
    "synonyms_and_antonyms",
    "syntax",
}
GENERAL_DOMAINS = set(TRAIN_QUOTAS) - LANGUAGE_DOMAINS
LICENSES = {
    "kaikki": "CC-BY-SA-3.0-and-GFDL",
    "ud_italian_isdt": "CC-BY-NC-SA-3.0",
    "wikipedia_it": "CC-BY-SA-4.0",
}


def stable_hash(*parts: object) -> str:
    return language.prompt_hash([SEED, *parts])


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def agreement_facts(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Create UD-grounded agreement/clitic completion items.

    Alternatives are attested surface forms of the same lemma and part of
    speech.  The original treebank sentence supplies the gold form.
    """

    sentences = list(language.iter_conllu(paths))
    forms: dict[tuple[str, str], set[str]] = defaultdict(set)
    for sentence in sentences:
        for token in sentence["tokens"]:
            form = language.clean_word(token["form"])
            lemma = language.clean_word(token["lemma"])
            if form and lemma and token["upos"] in {"ADJ", "AUX", "DET", "PRON", "VERB"}:
                forms[(language.normalize_text(lemma), token["upos"])].add(form)

    facts = []
    for sentence in sentences:
        text = str(sentence["text"])
        if not 20 <= len(text) <= 240:
            continue
        group_id = f"ud-sentence:{sentence['sent_id']}"
        for token in sentence["tokens"]:
            form = language.clean_word(token["form"])
            lemma = language.clean_word(token["lemma"])
            if not form or not lemma:
                continue
            key = (language.normalize_text(lemma), token["upos"])
            alternatives = sorted(
                (
                    value
                    for value in forms.get(key, set())
                    if language.normalize_text(value) != language.normalize_text(form)
                ),
                key=lambda value: stable_hash("agreement-option", key, value),
            )
            if len(alternatives) < 3:
                continue
            pattern = re.compile(rf"(?<!\w){re.escape(token['form'])}(?!\w)")
            masked, replacements = pattern.subn("____", text, count=1)
            if replacements != 1:
                continue
            facts.append(
                {
                    "category": "syntax",
                    "source": "ud_italian_isdt",
                    "source_id": (
                        f"ud:{sentence['sent_id']}:token:{token['id']}:agreement"
                    ),
                    "group_id": group_id,
                    "context": "",
                    "answer": form,
                    "distractor_pool": alternatives[:12],
                    "fields": {
                        "token": form,
                        "lemma": lemma,
                        "upos": token["upos"],
                        "masked_sentence": masked,
                        "task_family": (
                            "clitic_completion"
                            if token["upos"] == "PRON"
                            else "agreement_completion"
                        ),
                    },
                }
            )
    return facts


def language_question(fact: dict[str, Any]) -> str:
    category = fact["category"]
    fields = fact["fields"]
    if category == "orthography":
        templates = (
            "Quale grafia è corretta secondo l'ortografia italiana standard?",
            "Indica la forma scritta correttamente, distinguendola dalle grafie simili.",
            "Quale delle seguenti forme non contiene errori ortografici?",
        )
    elif category == "syntax" and "masked_sentence" in fields:
        templates = (
            "Quale forma completa correttamente la frase rispettando concordanza e struttura sintattica?\n«{masked_sentence}»",
            "Completa la frase con la forma grammaticalmente appropriata.\n«{masked_sentence}»",
        )
    else:
        templates = language.TEMPLATES[category]
    template = templates[int(stable_hash("template", fact["source_id"])[:8], 16) % len(templates)]
    return template.format(**fields)


def render_language_fact(fact: dict[str, Any]) -> dict[str, Any]:
    unique_distractors: dict[str, str] = {}
    for value in fact["distractor_pool"]:
        normalized = language.normalize_text(value)
        if normalized != language.normalize_text(fact["answer"]):
            unique_distractors.setdefault(normalized, str(value))
    distractors = sorted(
        unique_distractors.values(),
        key=lambda value: stable_hash("distractor", fact["source_id"], value),
    )[:3]
    if len(distractors) != 3:
        raise ValueError(f"Insufficient distractors for {fact['source_id']}")
    options = [str(fact["answer"]), *distractors]
    random.Random(int(stable_hash("option-order", fact["source_id"])[:16], 16)).shuffle(options)
    return {
        "id": f"scaled-kd:{fact['category']}:{stable_hash(fact['source_id'])[:20]}",
        "source": f"synthetic_{fact['category']}",
        "source_split": "unassigned",
        "group_id": str(fact["group_id"]),
        "context": str(fact.get("context") or ""),
        "question": language_question(fact),
        "options": options,
        "gold": options.index(str(fact["answer"])),
        "domain": fact["category"],
        "native_italian": True,
        "license": LICENSES[fact["source"]],
        "metadata": {
            "source": fact["source"],
            "source_id": fact["source_id"],
            "source_fields": fact["fields"],
            "answer": fact["answer"],
            "task_family": fact["fields"].get("task_family", fact["category"]),
            "validation": {
                "source_backed": True,
                "unique_options": True,
                "official_benchmark_read": False,
            },
        },
    }


def render_general_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    domain = str(candidate["topic"])
    if domain not in GENERAL_DOMAINS:
        raise ValueError(f"Unexpected general domain: {domain}")
    masked = re.sub(r"\s+", " ", str(candidate["masked_quote"])).strip()
    family = str(candidate.get("family") or "entity_identification")
    prompts = {
        "person_from_biography": "Quale persona è descritta nel seguente testo?",
        "work_from_description": "Quale opera è descritta nel seguente testo?",
        "place_from_description": "Quale luogo è descritto nel seguente testo?",
        "event_from_description": "Quale evento è descritto nel seguente testo?",
        "institution_from_description": "Quale istituzione è descritta nel seguente testo?",
        "concept_from_definition": "A quale voce corrisponde la seguente definizione?",
    }
    lead = prompts.get(str(candidate.get("entity_type")), "Quale voce completa correttamente la descrizione?")
    options = [str(value) for value in candidate["options"]]
    return {
        "id": f"scaled-kd:{domain}:{stable_hash(candidate['id'])[:20]}",
        "source": f"synthetic_{domain}",
        "source_split": "unassigned",
        "group_id": str(candidate["source_id"]),
        "context": "",
        "question": f"{lead}\n{masked}",
        "options": options,
        "gold": int(candidate["gold"]),
        "domain": domain,
        "native_italian": True,
        "license": LICENSES["wikipedia_it"],
        "metadata": {
            "source": "wikipedia_it",
            "source_id": candidate["source_id"],
            "source_fields": candidate.get("provenance", {}),
            "answer": options[int(candidate["gold"])],
            "task_family": family,
            "distractor_provenance": candidate.get("distractor_provenance", []),
            "validation": {
                "source_backed": True,
                "gold_deterministically_verified": all(
                    candidate.get("provenance", {}).get("verification", {}).values()
                ),
                "answer_masked_from_question": "[entità]" in masked.casefold(),
                "unique_options": len(set(map(language.normalize_text, options))) == 4,
                "official_benchmark_read": False,
            },
        },
    }


def question_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        language.normalize_text(row.get("context")),
        language.normalize_text(row["question"]),
        tuple(sorted(language.normalize_text(value) for value in row["options"])),
    )


def select_language(
    facts: Sequence[dict[str, Any]],
    domain: str,
    split: str,
    quota: int,
    forbidden_signatures: set[tuple[Any, ...]] | None = None,
) -> list[dict[str, Any]]:
    if forbidden_signatures is None:
        forbidden_signatures = set()
    wanted_dev = split == "dev"
    candidates = [
        fact
        for fact in facts
        if (
            int(stable_hash("language-split", fact["group_id"])[:8], 16) % 5 == 0
        )
        == wanted_dev
        and len(
            {
                language.normalize_text(value)
                for value in fact["distractor_pool"]
                if language.normalize_text(value)
                != language.normalize_text(fact["answer"])
            }
        )
        >= 3
    ]
    candidates = list({str(fact["source_id"]): fact for fact in candidates}.values())
    if domain == "syntax":
        agreement = [fact for fact in candidates if "masked_sentence" in fact["fields"]]
        dependency = [fact for fact in candidates if "masked_sentence" not in fact["fields"]]
        agreement_quota = min(len(agreement), round(quota * 0.60))
        preferred = sorted(agreement, key=lambda row: stable_hash(split, row["source_id"]))[
            :agreement_quota
        ]
        preferred_ids = {row["source_id"] for row in preferred}
        remainder = sorted(
            [
                row
                for row in dependency + agreement
                if row["source_id"] not in preferred_ids
            ],
            key=lambda row: stable_hash(split, row["source_id"]),
        )
        ordered = preferred + remainder
    else:
        ordered = sorted(candidates, key=lambda row: stable_hash(split, row["source_id"]))

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_signatures: set[tuple[Any, ...]] = set()
    for fact in ordered:
        row = render_language_fact(fact)
        signature = question_signature(row)
        if (
            row["id"] in seen_ids
            or signature in seen_signatures
            or signature in forbidden_signatures
        ):
            continue
        seen_ids.add(str(row["id"]))
        seen_signatures.add(signature)
        selected.append(row)
        if len(selected) == quota:
            break
    if len(selected) != quota:
        raise ValueError(f"{split}/{domain}: need {quota}, found {len(selected)}")
    forbidden_signatures.update(seen_signatures)
    return selected


def select_general(
    candidates: Sequence[dict[str, Any]],
    domain: str,
    train_quota: int,
    dev_quota: int,
    train_forbidden_signatures: set[tuple[Any, ...]] | None = None,
    dev_forbidden_signatures: set[tuple[Any, ...]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_forbidden_signatures is None:
        train_forbidden_signatures = set()
    if dev_forbidden_signatures is None:
        dev_forbidden_signatures = set()
    rows = [row for row in candidates if row.get("topic") == domain]
    rows = sorted(rows, key=lambda row: stable_hash("general-split", row["source_id"]))
    if len(rows) < train_quota + dev_quota:
        raise ValueError(
            f"general/{domain}: need {train_quota + dev_quota}, found {len(rows)}"
        )
    dev = []
    dev_groups = set()
    for source in rows:
        rendered = render_general_candidate(source)
        signature = question_signature(rendered)
        if signature in dev_forbidden_signatures:
            continue
        dev.append(rendered)
        dev_groups.add(str(rendered["group_id"]))
        dev_forbidden_signatures.add(signature)
        if len(dev) == dev_quota:
            break
    train = []
    for source in rows:
        rendered = render_general_candidate(source)
        if str(rendered["group_id"]) in dev_groups:
            continue
        signature = question_signature(rendered)
        if signature in train_forbidden_signatures:
            continue
        train.append(rendered)
        train_forbidden_signatures.add(signature)
        if len(train) == train_quota:
            break
    if len(train) != train_quota or len(dev) != dev_quota:
        raise ValueError(
            f"general/{domain}: unique selection shortfall "
            f"train={len(train)}/{train_quota} dev={len(dev)}/{dev_quota}"
        )
    return train, dev


def validate_split(rows: Sequence[dict[str, Any]], split: str) -> dict[str, Any]:
    expected = TRAIN_QUOTAS if split == "train" else DEV_QUOTAS
    counts = Counter(str(row["domain"]) for row in rows)
    if counts != Counter(expected):
        raise ValueError(f"Unexpected {split} counts: {dict(counts)}")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {split}")
    signatures = [question_signature(row) for row in rows]
    if len(signatures) != len(set(signatures)):
        raise ValueError(f"Duplicate question signatures in {split}")
    for row in rows:
        if len(row["options"]) != 4 or len(set(map(language.normalize_text, row["options"]))) != 4:
            raise ValueError(f"Invalid options: {row['id']}")
        if not 0 <= int(row["gold"]) < 4:
            raise ValueError(f"Invalid gold: {row['id']}")
        if str(row["id"]).lower().startswith("italic:") or str(row["source"]).lower() == "italic":
            raise ValueError(f"Forbidden official source: {row['id']}")
    return {"rows": len(rows), "by_domain": dict(sorted(counts.items()))}


def build(
    kaikki_path: Path,
    ud_paths: Sequence[Path],
    general_candidates_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    lexical = language.load_lexical_resources(kaikki_path)
    lexicon, orthography, relations = language.lexical_facts(lexical)
    morphology, dependencies = language.ud_facts(ud_paths)
    agreements = agreement_facts(ud_paths)
    facts = {
        "lexicon": lexicon,
        "morphology": morphology,
        "orthography": orthography,
        "synonyms_and_antonyms": relations,
        "syntax": dependencies + agreements,
    }
    general_candidates = read_jsonl(general_candidates_path)
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    train_signatures: set[tuple[Any, ...]] = set()
    dev_signatures: set[tuple[Any, ...]] = set()
    for domain in sorted(LANGUAGE_DOMAINS):
        train.extend(
            select_language(
                facts[domain],
                domain,
                "train",
                TRAIN_QUOTAS[domain],
                train_signatures,
            )
        )
        dev.extend(
            select_language(
                facts[domain],
                domain,
                "dev",
                DEV_QUOTAS[domain],
                dev_signatures,
            )
        )
    for domain in sorted(GENERAL_DOMAINS):
        domain_train, domain_dev = select_general(
            general_candidates,
            domain,
            TRAIN_QUOTAS[domain],
            DEV_QUOTAS[domain],
            train_signatures,
            dev_signatures,
        )
        train.extend(domain_train)
        dev.extend(domain_dev)

    train.sort(key=lambda row: stable_hash("train-order", row["id"]))
    dev.sort(key=lambda row: stable_hash("dev-order", row["id"]))
    validation = {
        "train": validate_split(train, "train"),
        "dev": validate_split(dev, "dev"),
    }
    train_groups = {str(row["group_id"]) for row in train}
    dev_groups = {str(row["group_id"]) for row in dev}
    overlap = train_groups & dev_groups
    if overlap:
        raise ValueError(f"Train/dev source-group overlap: {len(overlap)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(dev_path, dev)
    manifest = {
        "protocol": "zagreus-scaled-sector-kd-data-v1",
        "seed": SEED,
        "official_italic_read_or_used": False,
        "selection_data": "external-source-only",
        "split_policy": "source-group-disjoint; fresh-dev-locked-before-training",
        "licenses": LICENSES,
        "quotas": {"train": TRAIN_QUOTAS, "dev": DEV_QUOTAS},
        "validation": validation,
        "source_fact_counts": {domain: len(rows) for domain, rows in facts.items()},
        "syntax_mix": {
            "dependency_facts": len(dependencies),
            "agreement_and_clitic_facts": len(agreements),
        },
        "general_candidate_rows": len(general_candidates),
        "files": {
            "train.jsonl": {
                "rows": len(train),
                "sha256": language.sha256_file(train_path),
            },
            "dev.jsonl": {
                "rows": len(dev),
                "sha256": language.sha256_file(dev_path),
            },
            "general_candidates": {
                "path": str(general_candidates_path),
                "sha256": language.sha256_file(general_candidates_path),
            },
            "kaikki": {
                "path": str(kaikki_path),
                "sha256": language.sha256_file(kaikki_path),
            },
            "ud": {
                path.name: language.sha256_file(path) for path in ud_paths
            },
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaikki", type=Path, required=True)
    parser.add_argument("--ud", type=Path, action="append", required=True)
    parser.add_argument("--general-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.kaikki, args.ud, args.general_candidates, args.output_dir)
    print("SCALED_KD_DATA=" + json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
