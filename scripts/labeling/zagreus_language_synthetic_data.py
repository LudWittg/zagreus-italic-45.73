"""Build matched synthetic Italian-language MCQ datasets.

The deterministic arm is generated from Italian Wiktionary data extracted by
Wiktextract (Kaikki) and UD Italian ISDT.  The Gemma arm is derived from the
same fact records; its output is constrained to stem rewrites and distractors
from a source-backed candidate pool.
"""

from __future__ import annotations

import argparse
import difflib
import gzip
import hashlib
import heapq
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SEED = 20260719
LEXICAL_ENTRY_LIMIT = 150000
LANGUAGE_MACRO = "language capability"
CATEGORIES = (
    "lexicon",
    "morphology",
    "orthography",
    "synonyms_and_antonyms",
    "syntax",
)
SPLIT_QUOTAS = {
    "train": {
        "lexicon": 1942,
        "morphology": 278,
        "orthography": 1926,
        "synonyms_and_antonyms": 1926,
        "syntax": 1928,
    },
    "dev": {
        "lexicon": 243,
        "morphology": 35,
        "orthography": 241,
        "synonyms_and_antonyms": 241,
        "syntax": 240,
    },
    "test": {
        "lexicon": 243,
        "morphology": 35,
        "orthography": 241,
        "synonyms_and_antonyms": 241,
        "syntax": 240,
    },
}
LICENSES = {
    "kaikki": "CC-BY-SA-3.0-and-GFDL",
    "ud_italian_isdt": "CC-BY-NC-SA-3.0",
    "italic_style": "internal-benchmark-reference-only",
}
POS_LABELS = {
    "NOUN": "nome",
    "PROPN": "nome proprio",
    "VERB": "verbo",
    "AUX": "verbo ausiliare",
    "ADJ": "aggettivo",
    "ADV": "avverbio",
    "PRON": "pronome",
    "DET": "determinante",
    "ADP": "preposizione",
    "NUM": "numerale",
    "CCONJ": "congiunzione coordinante",
    "SCONJ": "congiunzione subordinante",
}
DEP_LABELS = {
    "nsubj": "soggetto",
    "obj": "complemento oggetto",
    "iobj": "complemento indiretto",
    "obl": "complemento obliquo",
    "amod": "modificatore aggettivale",
    "advmod": "modificatore avverbiale",
    "nmod": "modificatore nominale",
    "appos": "apposizione",
}
FEATURE_LABELS = {
    "Gender": {"Masc": "maschile", "Fem": "femminile"},
    "Number": {"Sing": "singolare", "Plur": "plurale"},
    "Person": {"1": "prima persona", "2": "seconda persona", "3": "terza persona"},
    "Tense": {
        "Pres": "presente",
        "Past": "passato",
        "Imp": "imperfetto",
        "Fut": "futuro",
    },
    "Mood": {
        "Ind": "indicativo",
        "Sub": "congiuntivo",
        "Cnd": "condizionale",
        "Imp": "imperativo",
    },
    "VerbForm": {
        "Fin": "forma finita",
        "Inf": "infinito",
        "Part": "participio",
        "Ger": "gerundio",
    },
    "Degree": {"Pos": "grado positivo", "Cmp": "comparativo", "Sup": "superlativo"},
}


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("’", "'").replace("`", "'")
    return re.sub(r"\s+", " ", text).strip().casefold()


def stable_rank(value: str, seed: int = SEED) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def prompt_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_italic(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            source = json.loads(line)
            pairs = []
            for item in source["options"]:
                if not isinstance(item, dict) or len(item) != 1:
                    raise ValueError(f"Malformed ITALIC options at row {index}")
                pairs.extend(item.items())
            labels = [str(label) for label, _ in pairs]
            if source["answer"] not in labels:
                raise ValueError(f"Missing ITALIC answer at row {index}")
            rows.append(
                {
                    "id": f"italic:{index}",
                    "question": str(source["question"]),
                    "options": [str(value) for _, value in pairs],
                    "gold": labels.index(source["answer"]),
                    "category": str(source["category"]),
                    "macro_category": str(source["macro_category"]),
                }
            )
    return rows


def select_style_exemplars(rows: Sequence[dict[str, Any]], per_category: int = 3) -> list[dict[str, Any]]:
    selected = []
    for category in CATEGORIES:
        candidates = [
            row
            for row in rows
            if row["macro_category"] == LANGUAGE_MACRO and row["category"] == category
        ]
        candidates.sort(key=lambda row: stable_rank(row["id"], SEED + 11))
        if len(candidates) < per_category:
            raise ValueError(f"ITALIC category {category!r} has only {len(candidates)} rows")
        for row in candidates[:per_category]:
            selected.append(
                {
                    "style_id": row["id"],
                    "category": category,
                    "question": row["question"],
                    "options": row["options"],
                }
            )
    return selected


def iter_kaikki(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("lang_code") == "it":
                    yield row


def clean_word(value: Any) -> str | None:
    word = unicodedata.normalize("NFC", str(value or "")).strip()
    if not re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", word):
        return None
    if not 3 <= len(word) <= 22:
        return None
    return word


def clean_gloss(value: Any, word: str) -> str | None:
    gloss = re.sub(r"\s+", " ", str(value or "")).strip(" .;:")
    lowered = normalize_text(gloss)
    if not 18 <= len(gloss) <= 220:
        return None
    if lowered.startswith(("plurale di ", "femminile di ", "maschile di ", "forma di ")):
        return None
    if normalize_text(word) in {lowered, lowered.split(" (")[0]}:
        return None
    return gloss


def relation_words(row: dict[str, Any], relation: str) -> list[str]:
    values = []
    for container in [row] + list(row.get("senses") or []):
        for item in container.get(relation) or []:
            value = item.get("word") if isinstance(item, dict) else item
            cleaned = clean_word(value)
            if cleaned and normalize_text(cleaned) != normalize_text(row.get("word")):
                values.append(cleaned)
    return list(dict.fromkeys(values))


def load_lexical_resources(path: Path, entry_limit: int | None = None) -> dict[str, Any]:
    entries = []
    entry_heap: list[tuple[int, int, dict[str, Any]]] = []
    usable_entries = 0
    words = set()
    for record_index, row in enumerate(iter_kaikki(path), 1):
        word = clean_word(row.get("word"))
        if not word:
            continue
        words.add(normalize_text(word))
        glosses = []
        for sense in row.get("senses") or []:
            for value in sense.get("glosses") or []:
                gloss = clean_gloss(value, word)
                if gloss:
                    glosses.append(gloss)
        entry = {
            "word": word,
            "pos": str(row.get("pos") or "unknown"),
            "glosses": list(dict.fromkeys(glosses)),
            "synonyms": relation_words(row, "synonyms"),
            "antonyms": relation_words(row, "antonyms"),
        }
        if entry["glosses"] or entry["synonyms"] or entry["antonyms"]:
            usable_entries += 1
            if entry_limit is None:
                entries.append(entry)
            else:
                rank = int(stable_rank(f"{word}:{entry['pos']}:{record_index}", SEED + 67), 16)
                item = (-rank, record_index, entry)
                if len(entry_heap) < entry_limit:
                    heapq.heappush(entry_heap, item)
                elif rank < -entry_heap[0][0]:
                    heapq.heapreplace(entry_heap, item)
        if record_index % 50000 == 0:
            print(
                f"Kaikki records={record_index} usable_entries={usable_entries} "
                f"retained_entries={len(entries) if entry_limit is None else len(entry_heap)} "
                f"words={len(words)}",
                flush=True,
            )
    if entry_limit is not None:
        entries = [item[2] for item in entry_heap]
        entries.sort(key=lambda entry: stable_rank(f"{entry['word']}:{entry['pos']}", SEED + 67))
    print(
        f"Kaikki complete usable_entries={usable_entries} retained_entries={len(entries)} words={len(words)}",
        flush=True,
    )
    return {"entries": entries, "word_set": words}


def parse_feats(value: str) -> dict[str, str]:
    if not value or value == "_":
        return {}
    result = {}
    for item in value.split("|"):
        if "=" in item:
            key, raw = item.split("=", 1)
            result[key] = raw
    return result


def iter_conllu(paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    metadata: dict[str, str] = {}
    tokens: list[dict[str, Any]] = []

    def emit() -> dict[str, Any] | None:
        if not tokens:
            return None
        text = metadata.get("text") or " ".join(token["form"] for token in tokens)
        return {
            "sent_id": metadata.get("sent_id") or f"anonymous:{prompt_hash(text)[:16]}",
            "text": text,
            "tokens": list(tokens),
        }

    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                line = raw.rstrip("\n")
                if not line:
                    sentence = emit()
                    if sentence:
                        yield sentence
                    metadata = {}
                    tokens = []
                    continue
                if line.startswith("# ") and " = " in line:
                    key, value = line[2:].split(" = ", 1)
                    metadata[key] = value
                    continue
                if line.startswith("#"):
                    continue
                columns = line.split("\t")
                if len(columns) != 10 or not columns[0].isdigit():
                    continue
                tokens.append(
                    {
                        "id": int(columns[0]),
                        "form": columns[1],
                        "lemma": columns[2],
                        "upos": columns[3],
                        "feats": parse_feats(columns[5]),
                        "head": int(columns[6]) if columns[6].isdigit() else 0,
                        "deprel": columns[7].split(":", 1)[0],
                    }
                )
        sentence = emit()
        if sentence:
            yield sentence
        metadata = {}
        tokens = []


def format_analysis(upos: str, feats: dict[str, str]) -> str:
    parts = [POS_LABELS.get(upos, upos.lower())]
    for key in ("Mood", "Tense", "VerbForm", "Person", "Gender", "Number", "Degree"):
        if key in feats:
            parts.append(FEATURE_LABELS.get(key, {}).get(feats[key], feats[key].lower()))
    return ", ".join(dict.fromkeys(parts))


def mutate_analysis(upos: str, feats: dict[str, str]) -> list[str]:
    candidates = []
    alternatives = {
        "Gender": ("Masc", "Fem"),
        "Number": ("Sing", "Plur"),
        "Person": ("1", "2", "3"),
        "Tense": ("Pres", "Past", "Imp", "Fut"),
        "Mood": ("Ind", "Sub", "Cnd", "Imp"),
        "VerbForm": ("Fin", "Inf", "Part", "Ger"),
        "Degree": ("Pos", "Cmp", "Sup"),
    }
    for key, values in alternatives.items():
        if key not in feats:
            continue
        for value in values:
            if value != feats[key]:
                changed = dict(feats)
                changed[key] = value
                candidates.append(format_analysis(upos, changed))
    other_pos = {
        "NOUN": "ADJ",
        "ADJ": "NOUN",
        "VERB": "AUX",
        "AUX": "VERB",
        "ADV": "ADJ",
        "PRON": "DET",
        "DET": "PRON",
    }.get(upos)
    if other_pos:
        candidates.append(format_analysis(other_pos, feats))
    return list(dict.fromkeys(candidates))


def spelling_variants(word: str, word_set: set[str]) -> list[str]:
    variants = set()
    lowered = word.lower()
    for index in range(1, len(word) - 1):
        if word[index].lower() not in "aeiouàèéìòóù":
            variants.add(word[:index] + word[index] + word[index:])
        variants.add(word[:index] + word[index + 1] + word[index] + word[index + 2 :])
        variants.add(word[:index] + word[index + 1 :])
    replacements = (("cq", "q"), ("q", "cq"), ("gl", "li"), ("li", "gl"), ("sce", "scie"), ("scie", "sce"))
    for old, new in replacements:
        if old in lowered:
            index = lowered.index(old)
            variants.add(word[:index] + new + word[index + len(old) :])
    accent_map = str.maketrans("àèéìòóù", "aeeioou")
    unaccented = word.translate(accent_map)
    if unaccented != word:
        variants.add(unaccented)
    clean = []
    for value in variants:
        candidate = clean_word(value)
        if candidate and normalize_text(candidate) not in word_set and normalize_text(candidate) != normalize_text(word):
            clean.append(candidate)
    return sorted(set(clean), key=lambda value: stable_rank(f"{word}:{value}", SEED + 41))


def bounded_pool(
    values: Sequence[str],
    *,
    key: str,
    forbidden: set[str],
    limit: int = 12,
) -> list[str]:
    """Select a reproducible pool without sorting the full vocabulary per fact."""
    if not values:
        return []
    digest = stable_rank(key, SEED + 59)
    start = int(digest[:16], 16) % len(values)
    stride = 7919
    selected = []
    seen = set()
    for offset in range(min(len(values), 1000)):
        value = values[(start + offset * stride) % len(values)]
        normalized = normalize_text(value)
        if normalized in forbidden or normalized in seen:
            continue
        selected.append(value)
        seen.add(normalized)
        if len(selected) == limit:
            break
    return selected


def lexical_facts(resources: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    entries = resources["entries"][:LEXICAL_ENTRY_LIMIT]
    word_set = resources["word_set"]
    by_pos: dict[str, list[str]] = defaultdict(list)
    gloss_by_pos: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        by_pos[entry["pos"]].append(entry["word"])
        gloss_by_pos[entry["pos"]].extend(entry["glosses"])

    lexicon = []
    orthography = []
    relations = []
    for entry_index, entry in enumerate(entries, 1):
        word = entry["word"]
        pos = entry["pos"]
        for index, gloss in enumerate(entry["glosses"][:2]):
            pool = bounded_pool(
                gloss_by_pos[pos],
                key=f"lex:{word}:{index}",
                forbidden={normalize_text(gloss)},
            )
            pool = [value for value in pool if normalize_text(word) not in normalize_text(value)]
            if len(pool) >= 5:
                lexicon.append(
                    {
                        "category": "lexicon",
                        "source": "kaikki",
                        "source_id": f"kaikki:{normalize_text(word)}:{pos}:sense:{index}",
                        "group_id": f"kaikki-lemma:{normalize_text(word)}",
                        "context": "",
                        "answer": gloss,
                        "distractor_pool": pool,
                        "fields": {"word": word, "pos": pos},
                    }
                )
        variants = spelling_variants(word, word_set)
        if len(variants) >= 3:
            orthography.append(
                {
                    "category": "orthography",
                    "source": "kaikki",
                    "source_id": f"kaikki:{normalize_text(word)}:{pos}:spelling",
                    "group_id": f"kaikki-lemma:{normalize_text(word)}",
                    "context": "",
                    "answer": word,
                    "distractor_pool": variants[:10],
                    "fields": {"word": word, "pos": pos},
                }
            )
        for relation in ("synonyms", "antonyms"):
            relation_label = "sinonimo" if relation == "synonyms" else "contrario"
            for target in entry[relation][:3]:
                forbidden = {normalize_text(word), normalize_text(target)}
                forbidden.update(normalize_text(value) for value in entry["synonyms"] + entry["antonyms"])
                pool = bounded_pool(
                    by_pos[pos],
                    key=f"rel:{word}:{target}",
                    forbidden=forbidden,
                )
                if len(pool) >= 5:
                    relations.append(
                        {
                            "category": "synonyms_and_antonyms",
                            "source": "kaikki",
                            "source_id": f"kaikki:{normalize_text(word)}:{relation}:{normalize_text(target)}",
                            "group_id": f"kaikki-lemma:{normalize_text(word)}",
                            "context": "",
                            "answer": target,
                            "distractor_pool": pool,
                            "fields": {"word": word, "relation": relation_label, "pos": pos},
                        }
                    )
        if entry_index % 50000 == 0:
            print(
                f"Lexical facts entries={entry_index} lexicon={len(lexicon)} "
                f"orthography={len(orthography)} relations={len(relations)}",
                flush=True,
            )
    print(
        f"Lexical facts complete lexicon={len(lexicon)} orthography={len(orthography)} "
        f"relations={len(relations)}",
        flush=True,
    )
    return lexicon, orthography, relations


def ud_facts(
    paths: Sequence[Path],
    fact_limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    morphology = []
    syntax = []
    morphology_heap: list[tuple[int, str, dict[str, Any]]] = []
    syntax_heap: list[tuple[int, str, dict[str, Any]]] = []

    def retain(
        rows: list[dict[str, Any]],
        heap: list[tuple[int, str, dict[str, Any]]],
        row: dict[str, Any],
        salt: int,
    ) -> None:
        if fact_limit is None:
            rows.append(row)
            return
        source_id = str(row["source_id"])
        rank = int(stable_rank(source_id, salt), 16)
        item = (-rank, source_id, row)
        if len(heap) < fact_limit:
            heapq.heappush(heap, item)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, item)

    for sentence in iter_conllu(paths):
        text = sentence["text"]
        if not 20 <= len(text) <= 260:
            continue
        group_id = f"ud-sentence:{sentence['sent_id']}"
        for token in sentence["tokens"]:
            form = token["form"]
            upos = token["upos"]
            feats = token["feats"]
            if upos in POS_LABELS and feats:
                answer = format_analysis(upos, feats)
                pool = mutate_analysis(upos, feats)
                if len(pool) >= 3:
                    retain(
                        morphology,
                        morphology_heap,
                        {
                            "category": "morphology",
                            "source": "ud_italian_isdt",
                            "source_id": f"ud:{sentence['sent_id']}:token:{token['id']}:morph",
                            "group_id": group_id,
                            "context": text,
                            "answer": answer,
                            "distractor_pool": pool[:10],
                            "fields": {"token": form, "lemma": token["lemma"], "upos": upos, "feats": feats},
                        },
                        SEED + 71,
                    )
            relation = token["deprel"]
            if relation in DEP_LABELS and re.search(r"\w", form):
                pool = [label for key, label in DEP_LABELS.items() if key != relation]
                retain(
                    syntax,
                    syntax_heap,
                    {
                        "category": "syntax",
                        "source": "ud_italian_isdt",
                        "source_id": f"ud:{sentence['sent_id']}:token:{token['id']}:syntax",
                        "group_id": group_id,
                        "context": text,
                        "answer": DEP_LABELS[relation],
                        "distractor_pool": pool,
                        "fields": {"token": form, "lemma": token["lemma"], "deprel": relation},
                    },
                    SEED + 73,
                )
    if fact_limit is not None:
        morphology = [item[2] for item in morphology_heap]
        syntax = [item[2] for item in syntax_heap]
    return morphology, syntax


TEMPLATES = {
    "lexicon": (
        "Quale definizione descrive meglio la parola «{word}»?",
        "Che cosa significa «{word}»?",
        "Indica il significato corretto del termine «{word}».",
    ),
    "morphology": (
        "Qual è l'analisi morfologica corretta di «{token}» nella frase?",
        "Come si analizza morfologicamente la forma «{token}» nel contesto dato?",
        "Quali proprietà morfologiche ha «{token}» nella frase?",
    ),
    "orthography": (
        "Quale parola è scritta correttamente?",
        "Indica la grafia corretta.",
        "Quale delle seguenti forme rispetta l'ortografia italiana?",
    ),
    "synonyms_and_antonyms": (
        "Quale parola è un {relation} di «{word}»?",
        "Indica il {relation} corretto di «{word}».",
        "Quale termine esprime un {relation} di «{word}»?",
    ),
    "syntax": (
        "Quale funzione sintattica svolge «{token}» nella frase?",
        "Nella frase data, qual è il ruolo sintattico di «{token}»?",
        "Come si classifica la funzione sintattica di «{token}»?",
    ),
}


def render_fact(fact: dict[str, Any], variant: int, generator: str) -> dict[str, Any]:
    template_id = variant % len(TEMPLATES[fact["category"]])
    template = TEMPLATES[fact["category"]][template_id]
    question = template.format(**fact["fields"])
    distractors = sorted(
        fact["distractor_pool"],
        key=lambda value: stable_rank(f"{fact['source_id']}:{variant}:{value}", SEED + 73),
    )[:3]
    options = [fact["answer"]] + distractors
    rng = random.Random(int(stable_rank(f"options:{fact['source_id']}:{variant}")[:16], 16))
    rng.shuffle(options)
    gold = options.index(fact["answer"])
    record_id = f"synthetic:{fact['category']}:{prompt_hash([fact['source_id'], variant])[:18]}"
    return {
        "id": record_id,
        "source": f"synthetic_{fact['category']}",
        "source_split": "unassigned",
        "group_id": fact["group_id"],
        "context": fact["context"],
        "question": question,
        "options": options,
        "gold": gold,
        "domain": fact["category"],
        "native_italian": True,
        "license": LICENSES[fact["source"]],
        "metadata": {
            "category": fact["category"],
            "source": fact["source"],
            "source_id": fact["source_id"],
            "source_fields": fact["fields"],
            "answer": fact["answer"],
            "distractor_pool": fact["distractor_pool"],
            "template_id": template_id,
            "generator": generator,
            "teacher_revision": None,
            "prompt_hash": None,
            "validation": {"source_backed": True, "unique_options": True},
        },
    }


def record_signature(row: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    return (
        normalize_text(row.get("context")),
        normalize_text(row["question"]),
        tuple(sorted(normalize_text(value) for value in row["options"])),
    )


def allocate_category(facts: Sequence[dict[str, Any]], category: str) -> dict[str, list[dict[str, Any]]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        by_group[fact["group_id"]].append(fact)
    groups = list(by_group)
    if not groups:
        raise ValueError(f"No fact groups for {category}")
    # The assignment depends only on group_id, so a lemma or UD sentence used
    # by multiple categories can never cross splits.
    assigned = {split: [] for split in SPLIT_QUOTAS}
    for group in groups:
        bucket = int(stable_rank(group, SEED + 101)[:16], 16) % 10
        split = "dev" if bucket == 0 else "test" if bucket == 1 else "train"
        assigned[split].append(group)

    output: dict[str, list[dict[str, Any]]] = {}
    for split, split_groups in assigned.items():
        quota = SPLIT_QUOTAS[split][category]
        selected_groups = heapq.nsmallest(
            min(len(split_groups), quota),
            split_groups,
            key=lambda value: stable_rank(f"{split}:{value}", SEED + 131),
        )
        candidates = []
        seen_signatures = set()
        round_index = 0
        while len(candidates) < quota:
            added = 0
            for group in selected_groups:
                group_facts = by_group[group]
                slot_count = len(group_facts) * len(TEMPLATES[category])
                if round_index >= slot_count:
                    continue
                fact = group_facts[round_index // len(TEMPLATES[category])]
                variant = round_index % len(TEMPLATES[category])
                candidate = render_fact(fact, variant, "programmatic-v1")
                signature = record_signature(candidate)
                if signature in seen_signatures:
                    continue
                candidates.append(candidate)
                seen_signatures.add(signature)
                added += 1
                if len(candidates) == quota:
                    break
            if not added:
                break
            round_index += 1
        if len(candidates) < quota:
            raise ValueError(
                f"Only {len(candidates)} {category} candidates for {split}; need {quota}"
            )
        for row in candidates:
            row["source_split"] = split
        output[split] = candidates
    return output


def close_to_style(question: str, exemplars: Sequence[dict[str, Any]]) -> bool:
    normalized = normalize_text(question)
    for exemplar in exemplars:
        other = normalize_text(exemplar["question"])
        if normalized == other:
            return True
        if difflib.SequenceMatcher(None, normalized, other).ratio() >= 0.86:
            return True
    return False


def validate_records(
    rows: Sequence[dict[str, Any]],
    split: str,
    italic_questions: set[str],
    exemplars: Sequence[dict[str, Any]],
) -> Counter:
    counts = Counter()
    seen = set()
    for row in rows:
        counts["rows"] += 1
        counts[f"category:{row['domain']}"] += 1
        key = record_signature(row)
        if key in seen:
            raise ValueError(f"Duplicate synthetic record in {split}: {row['id']}")
        seen.add(key)
        if normalize_text(row["question"]) in italic_questions:
            raise ValueError(f"Exact ITALIC question copy: {row['id']}")
        if close_to_style(row["question"], exemplars):
            raise ValueError(f"Near-copy of style exemplar: {row['id']}")
        if len(row["options"]) != 4 or len({normalize_text(x) for x in row["options"]}) != 4:
            raise ValueError(f"Invalid options: {row['id']}")
        if not 0 <= int(row["gold"]) < 4:
            raise ValueError(f"Invalid gold: {row['id']}")
        if normalize_text(row["options"][row["gold"]]) != normalize_text(row["metadata"]["answer"]):
            raise ValueError(f"Gold/source mismatch: {row['id']}")
    for category, quota in SPLIT_QUOTAS[split].items():
        if counts[f"category:{category}"] != quota:
            raise ValueError(f"{split}/{category}: expected {quota}, found {counts[f'category:{category}']}")
    return counts


def build(
    kaikki_path: Path,
    ud_paths: Sequence[Path],
    italic_path: Path,
    output_dir: Path,
) -> None:
    italic = parse_italic(italic_path)
    if len(italic) != 10000:
        raise ValueError(f"Expected 10,000 ITALIC rows, found {len(italic)}")
    language_counts = Counter(
        row["category"] for row in italic if row["macro_category"] == LANGUAGE_MACRO
    )
    expected_counts = {
        "lexicon": 979,
        "morphology": 140,
        "orthography": 971,
        "synonyms_and_antonyms": 971,
        "syntax": 973,
    }
    if dict(language_counts) != expected_counts:
        raise ValueError(f"Unexpected ITALIC language counts: {dict(language_counts)}")
    exemplars = select_style_exemplars(italic)
    write_json(output_dir / "style_exemplars_no_answers.json", exemplars)
    write_json(output_dir / "style_exemplar_ids.json", [row["style_id"] for row in exemplars])
    write_json(
        output_dir / "italic_question_hashes.json",
        sorted(hashlib.sha256(normalize_text(row["question"]).encode("utf-8")).hexdigest() for row in italic),
    )

    lexical = load_lexical_resources(kaikki_path)
    lexicon, orthography, relations = lexical_facts(lexical)
    morphology, syntax = ud_facts(ud_paths)
    facts_by_category = {
        "lexicon": lexicon,
        "morphology": morphology,
        "orthography": orthography,
        "synonyms_and_antonyms": relations,
        "syntax": syntax,
    }
    outputs: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_QUOTAS}
    for category, facts in facts_by_category.items():
        allocated = allocate_category(facts, category)
        for split in outputs:
            outputs[split].extend(allocated[split])

    italic_questions = {normalize_text(row["question"]) for row in italic}
    validation = {}
    group_splits: dict[str, set[str]] = defaultdict(set)
    for split, rows in outputs.items():
        rows.sort(key=lambda row: stable_rank(row["id"], SEED + 151))
        validation[split] = dict(validate_records(rows, split, italic_questions, exemplars))
        for row in rows:
            group_splits[row["group_id"]].add(split)
        write_jsonl(output_dir / f"programmatic_{split}.jsonl", rows)
    leaks = {group: sorted(splits) for group, splits in group_splits.items() if len(splits) > 1}
    if leaks:
        sample = list(leaks.items())[:5]
        raise ValueError(f"Source groups cross splits: {sample}")

    fact_counts = {category: len(facts) for category, facts in facts_by_category.items()}
    group_counts = {
        category: len({fact["group_id"] for fact in facts})
        for category, facts in facts_by_category.items()
    }
    manifest = {
        "format": "zagreus-synthetic-italian-language-v1",
        "seed": SEED,
        "benchmark_informed_internal_only": True,
        "italic_sha256": sha256_file(italic_path),
        "italic_language_counts": expected_counts,
        "style_policy": "three question-and-option exemplars per category; answers never exposed",
        "style_exemplar_ids": [row["style_id"] for row in exemplars],
        "licenses": LICENSES,
        "redistribution": "Internal only; UD-derived records retain CC-BY-NC-SA restrictions.",
        "fact_counts": fact_counts,
        "group_counts": group_counts,
        "lexical_entry_policy": {
            "eligible_entries": len(lexical["entries"]),
            "processed_entries": min(len(lexical["entries"]), LEXICAL_ENTRY_LIMIT),
            "selection": "first eligible records in pinned Kaikki dump order",
            "full_dump_word_set_used_for_orthography_validation": True,
        },
        "validation": validation,
        "files": {},
    }
    for split in outputs:
        filename = f"programmatic_{split}.jsonl"
        path = output_dir / filename
        manifest["files"][filename] = {
            "rows": len(outputs[split]),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    write_json(output_dir / "data_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaikki", type=Path, required=True)
    parser.add_argument("--ud", type=Path, action="append", required=True)
    parser.add_argument("--italic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.kaikki, args.ud, args.italic, args.output_dir)


if __name__ == "__main__":
    main()
