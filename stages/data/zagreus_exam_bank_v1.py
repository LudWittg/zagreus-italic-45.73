"""Build the private, official-keyed exam-bank asset for RUNBOOK_v2.

The source PDFs and extracted question text are private campaign inputs and must
never enter a distributable artifact.  Durable public evidence contains only
hashes, counts, rules, and aggregate diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import subprocess
import tempfile
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_WHITELIST_SHA256 = (
    "77de397aa6572f89e485e1db72d5f17eba273daca617ba6e0264d323ee02b801"
)
RETRIEVAL_DATE = "2026-08-02"
PUBLISHER = "Comando Generale della Guardia di Finanza"
LICENSE_EVIDENCE = (
    "Vietata la pubblicazione, la riproduzione e la divulgazione a scopo di lucro."
)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
EMBEDDING_THRESHOLD = 0.8000
TAIL_AUDITOR_ID = "OpenAI Codex (single disclosed model auditor), session 2026-08-02"
TAIL_DECISION_ATTESTATION = (
    "Every non-inherited pair in this packet was inspected by the disclosed "
    "single model auditor; default distinct is an explicit row-level verdict, "
    "not a missing review."
)
ALLOCATION_SEED = "20260802"
SPLIT_DOMAIN_QUOTAS = {
    "exam_diag": 100,
    "dev_early": 100,
    "final_holdout": 150,
}

SOURCE_FILES = {
    "gdf_2017_ufficiali_cultura": "gdf_2017_ufficiali_cultura.zip",
    "gdf_2017_soccorso_cultura": "gdf_2017_soccorso_cultura.zip",
    "gdf_2018_aiutanti_cultura": "gdf_2018_aiutanti_cultura.zip",
    "gdf_2019_aiutanti_cultura": "gdf_2019_aiutanti_cultura.zip",
    "gdf_2020_aiutanti_cultura": "gdf_2020_aiutanti_cultura.zip",
}

# The 2017 officer bank has a much finer language taxonomy.  Passage-based
# reading-comprehension files A-C are excluded because their contexts do not
# reconstruct as self-contained table rows.  Later banks use the stable six
# ITALIC-aligned codes below; H (history/current affairs) is excluded wholesale
# rather than trying to decide mechanically which facts have gone stale.
OFFICER_2017_DOMAINS = {
    "D": "synonyms_and_antonyms",
    "E": "lexicon",
    "F": "lexicon",
    "G": "synonyms_and_antonyms",
    "H": "orthography",
    "I": "orthography",
    "L": "orthography",
    "M": "orthography",
    "N": "syntax",
    "O": "syntax",
    "P": "morphology",
    "Q": "morphology",
    "R": "morphology",
    "S": "morphology",
    "T": "morphology",
    "U": "syntax",
}
LATER_DOMAINS = {
    "E": "civic_education",
    "H": "history_mixed",
    "L": "lexicon",
    "M": "morphology",
    "O": "orthography",
    "S": "syntax",
    "T": "geography",
}

CURRENT_AFFAIRS_CUES = (
    "oggi", "attualmente", "attuale", "attuali", "recentemente", "ultimo", "ultima",
    "in carica", "anno corrente", "al momento", "piu recente",
)
STABLE_HISTORY_CUES = (
    "guerra mondiale", "risorgimento", "impero", "imperatore", "romano",
    "romana", "medioevo", "medievale", "rinascimento", "fascismo", "fascista",
    "nazismo", "nazista", "rivoluzione", "monarchia", "dinastia", "trattato",
    "armistizio", "antica grecia", "antica roma", "guerra fredda", "unificazione",
    "indipendenza", "liberazione", "secolo",
)
YEAR_RE = re.compile(r"(?<!\d)(1\d{3}|20\d{2})(?!\d)")

ROW_RE = re.compile(r"^([A-Z]{2}\d{5})\s+(.*)$")
OPTION_RE = re.compile(r"(?<![\w])([a-d])\)\s*")
GOLD_RE = re.compile(r"\s{2,}([a-d])\s*$", re.IGNORECASE)
HEADER_FRAGMENT = "vietata la pubblicazione"
PAGE_FOOTER_ARTIFACT_RE = re.compile(r"\bpagina(?:\s+\d+)?\s*$", re.IGNORECASE)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def recursively_unescape(value: object) -> str:
    text = str(value or "")
    for _ in range(8):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    return text


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", recursively_unescape(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold().replace("’", "'").replace("`", "'")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalized_question_hash(value: object) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


ITALIAN_STOPWORDS = {
    "a", "ad", "al", "alla", "alle", "allo", "ai", "agli", "anche", "che",
    "chi", "con", "da", "dal", "dalla", "dalle", "dallo", "dei", "degli",
    "del", "della", "delle", "dello", "di", "e", "ed", "e", "fra", "gli",
    "ha", "hai", "hanno", "i", "il", "in", "io", "la", "le", "lo", "ma",
    "nei", "negli", "nel", "nella", "nelle", "nello", "non", "o", "per",
    "piu", "quale", "quali", "questa", "queste", "questi", "questo", "se",
    "seguente", "seguenti", "si", "sono", "su", "sul", "sulla", "tra", "un",
    "una", "uno", "vi", "è",
}


def content_words(value: object) -> tuple[str, ...]:
    return tuple(
        token for token in normalize_text(value).split()
        if len(token) > 1 and token not in ITALIAN_STOPWORDS
    )


def word_shingles(value: object, width: int = 3) -> frozenset[str]:
    tokens = content_words(value)
    if len(tokens) < width:
        return frozenset(tokens)
    return frozenset(" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1))


def jaccard(left: set[str] | frozenset[str], right: set[str] | frozenset[str]) -> float:
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def containment(left: set[str] | frozenset[str], right: set[str] | frozenset[str]) -> float:
    denominator = min(len(left), len(right))
    return len(left & right) / denominator if denominator else 0.0


def record_key(question: str, options: Sequence[str]) -> str:
    payload = normalize_text(question) + "\x1e" + "\x1f".join(
        sorted(normalize_text(option) for option in options)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compact_piece(value: str) -> str:
    value = value.replace("\x0c", " ")
    return re.sub(r"\s+", " ", value).strip()


def find_option_columns(line: str) -> tuple[list[re.Match[str]], re.Match[str]] | None:
    matches = list(OPTION_RE.finditer(line))
    for start in range(len(matches)):
        run = matches[start : start + 4]
        if len(run) != 4 or [item.group(1) for item in run] != list("abcd"):
            continue
        gold = GOLD_RE.search(line[run[-1].end() :])
        if gold is None:
            continue
        # GOLD_RE was run on a suffix, so translate it back to absolute columns.
        absolute = re.compile(r".").match("x")
        assert absolute is not None
        gold_start = run[-1].end() + gold.start(1)
        gold_end = run[-1].end() + gold.end(1)
        proxy = _SpanMatch(gold.group(1), gold_start, gold_end)
        return run, proxy
    return None


class _SpanMatch:
    """The small Match subset needed for an absolute gold-column span."""

    def __init__(self, value: str, start: int, end: int) -> None:
        self._value = value
        self._start = start
        self._end = end

    def group(self, index: int = 0) -> str:
        if index not in (0, 1):
            raise IndexError(index)
        return self._value

    def start(self, index: int = 0) -> int:
        return self._start

    def end(self, index: int = 0) -> int:
        return self._end


def parse_record_chunk(code: str, lines: Sequence[str]) -> tuple[dict[str, Any] | None, str]:
    first = lines[0]
    columns = find_option_columns(first)
    if columns is None:
        return None, "missing_ordered_option_or_gold_columns"
    option_matches, gold = columns
    a_start, b_start, c_start, d_start = [match.start() for match in option_matches]
    gold_start = gold.start()
    if not (7 < a_start < b_start < c_start < d_start < gold_start):
        return None, "invalid_column_order"

    question_parts: list[str] = []
    option_parts: list[list[str]] = [[], [], [], []]
    for index, raw in enumerate(lines):
        line = raw.replace("\x0c", "")
        if not line.strip() or HEADER_FRAGMENT in line.casefold():
            continue
        if index == 0:
            question_piece = line[len(code) : a_start]
        else:
            question_piece = line[:a_start]
        segments = (
            line[a_start:b_start],
            line[b_start:c_start],
            line[c_start:d_start],
            line[d_start:gold_start],
        )
        question_piece = compact_piece(question_piece)
        if question_piece:
            question_parts.append(question_piece)
        for option_index, segment in enumerate(segments):
            piece = compact_piece(segment)
            if index == 0:
                piece = re.sub(rf"^{chr(97 + option_index)}\)\s*", "", piece, count=1)
            if piece:
                option_parts[option_index].append(piece)

    question = compact_piece(" ".join(question_parts))
    options = [compact_piece(" ".join(parts)) for parts in option_parts]
    if not question:
        return None, "empty_question"
    if any(not option for option in options):
        return None, "empty_option"
    normalized_options = [normalize_text(option) for option in options]
    if len(set(normalized_options)) != 4:
        return None, "duplicate_normalized_options"
    gold_letter = gold.group(1).upper()
    gold_index = ord(gold_letter) - ord("A")
    if not 0 <= gold_index < 4:
        return None, "out_of_range_gold"
    return {
        "record_code": code,
        "question": question,
        "options": options,
        "gold": gold_index,
        "gold_letter": gold_letter,
        "gold_text": options[gold_index],
        "answer_derivation": "official_pdf_final_isolated_letter_column",
    }, "accepted"


def parse_pdf_text(text: str) -> tuple[list[dict[str, Any]], Counter[str]]:
    lines = text.splitlines()
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = ROW_RE.match(line.replace("\x0c", ""))
        if match:
            starts.append((index, match.group(1)))
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for position, (start, code) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        parsed, reason = parse_record_chunk(code, lines[start:end])
        counts[reason] += 1
        if parsed is not None:
            rows.append(parsed)
    counts["detected_record_codes"] = len(starts)
    return rows, counts


def pdf_to_text(pdf_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(prefix="zagreus_exam_", suffix=".pdf") as handle:
        handle.write(pdf_bytes)
        handle.flush()
        result = subprocess.run(
            ["pdftotext", "-layout", handle.name, "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    return result.stdout


def domain_for(source_id: str, member: str) -> str | None:
    stem = Path(member).stem.upper()
    if len(stem) != 2:
        return None
    subject = stem[0]
    if source_id == "gdf_2017_ufficiali_cultura":
        return OFFICER_2017_DOMAINS.get(subject)
    return LATER_DOMAINS.get(subject)


def stable_history_item(question: str, options: Sequence[str]) -> bool:
    text = normalize_text(" ".join([question, *options]))
    years = [int(value) for value in YEAR_RE.findall(text)]
    if any(year > 2000 for year in years):
        return False
    if any(cue in text for cue in CURRENT_AFFAIRS_CUES):
        return False
    if any(1000 <= year <= 2000 for year in years):
        return True
    return any(cue in text for cue in STABLE_HISTORY_CUES)


def is_current_event_item(question: str, options: Sequence[str]) -> bool:
    text = normalize_text(" ".join([question, *options]))
    years = [int(value) for value in YEAR_RE.findall(text)]
    return any(year > 2000 for year in years) or any(
        cue in text for cue in CURRENT_AFFAIRS_CUES
    )


def index_entry(whitelist: Mapping[str, Any], source_id: str) -> Mapping[str, Any]:
    matches = [row for row in whitelist["index_snapshots"] if row["source_id"] == source_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one whitelist row for {source_id}, found {len(matches)}")
    return matches[0]


def source_asset_url(entry: Mapping[str, Any]) -> str:
    prefix = str(entry.get("asset_url_prefix") or entry["index_url"])
    return prefix.rstrip("/") + "/@@download/file_principale"


def ingest(whitelist_path: Path, source_dir: Path, output_dir: Path) -> dict[str, Any]:
    if sha256_file(whitelist_path) != EXPECTED_WHITELIST_SHA256:
        raise ValueError("SOURCE_WHITELIST.json changed after freeze")
    whitelist = read_json(whitelist_path)
    private_dir = output_dir / "private_derived"
    public_dir = output_dir / "evidence"
    private_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    source_manifests: list[dict[str, Any]] = []
    aggregate: Counter[str] = Counter()
    per_domain: Counter[str] = Counter()
    for source_id, filename in SOURCE_FILES.items():
        archive_path = source_dir / filename
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        entry = index_entry(whitelist, source_id)
        archive_sha = sha256_file(archive_path)
        member_manifests: list[dict[str, Any]] = []
        with zipfile.ZipFile(archive_path) as archive:
            for info in sorted(archive.infolist(), key=lambda value: value.filename):
                if info.is_dir() or not info.filename.casefold().endswith(".pdf"):
                    continue
                domain = domain_for(source_id, info.filename)
                member_bytes = archive.read(info)
                member_sha = sha256_bytes(member_bytes)
                if domain is None:
                    member_manifests.append({
                        "member": info.filename,
                        "member_sha256": member_sha,
                        "status": "excluded_non_target_or_current_events",
                    })
                    aggregate["excluded_pdf_members"] += 1
                    continue
                parsed_rows, counts = parse_pdf_text(pdf_to_text(member_bytes))
                member_manifests.append({
                    "member": info.filename,
                    "member_sha256": member_sha,
                    "domain": domain,
                    "status": "parsed",
                    "parse_counts": dict(sorted(counts.items())),
                })
                for key, value in counts.items():
                    aggregate[f"parse:{key}"] += value
                for parsed in parsed_rows:
                    if is_current_event_item(parsed["question"], parsed["options"]):
                        aggregate["excluded_global_current_event_rows"] += 1
                        continue
                    if domain == "history_mixed" and not stable_history_item(
                        parsed["question"], parsed["options"]
                    ):
                        aggregate["excluded_current_or_unclassified_history_rows"] += 1
                        continue
                    final_domain = "history" if domain == "history_mixed" else domain
                    record_id = f"{source_id}:{parsed['record_code']}"
                    row = {
                        "id": record_id,
                        "source": source_id,
                        "source_publisher": PUBLISHER,
                        "source_url": source_asset_url(entry),
                        "source_index_url": entry["index_url"],
                        "source_member": info.filename,
                        "source_record_locator": f"{info.filename}#{parsed['record_code']}",
                        "retrieval_date": RETRIEVAL_DATE,
                        "archive_sha256": archive_sha,
                        "source_member_sha256": member_sha,
                        "license_evidence": LICENSE_EVIDENCE,
                        "key_provenance": "official Guardia di Finanza bank PDF",
                        "domain": final_domain,
                        **parsed,
                    }
                    row["normalized_question_hash"] = normalized_question_hash(row["question"])
                    row["record_key"] = record_key(row["question"], row["options"])
                    row["group_id"] = row["record_key"]
                    all_rows.append(row)
                    per_domain[final_domain] += 1
        source_manifests.append({
            "source_id": source_id,
            "archive_filename": filename,
            "archive_sha256": archive_sha,
            "source_url": source_asset_url(entry),
            "index_url": entry["index_url"],
            "index_sha256": entry["index_sha256"],
            "members": member_manifests,
        })

    # Closed-world key consistency: identical question+option sets must name
    # the same gold answer text.  Keep one canonical source occurrence and
    # retain all provenance locators in the private row.
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_key[row["record_key"]].append(row)
    clean_rows: list[dict[str, Any]] = []
    ambiguous_rows: list[dict[str, Any]] = []
    duplicate_occurrences = 0
    for key, members in sorted(by_key.items()):
        golds = {normalize_text(row["gold_text"]) for row in members}
        if len(golds) != 1:
            ambiguous_rows.extend(members)
            continue
        canonical = sorted(members, key=lambda row: row["id"])[0]
        canonical["provenance_occurrences"] = [
            {
                "id": row["id"],
                "source": row["source"],
                "source_member": row["source_member"],
                "source_record_locator": row["source_record_locator"],
                "archive_sha256": row["archive_sha256"],
                "source_member_sha256": row["source_member_sha256"],
            }
            for row in sorted(members, key=lambda row: row["id"])
        ]
        duplicate_occurrences += len(members) - 1
        clean_rows.append(canonical)

    clean_rows.sort(key=lambda row: row["id"])
    write_jsonl(private_dir / "ingested_unique.jsonl", clean_rows)
    write_jsonl(private_dir / "closed_world_key_conflicts.jsonl", ambiguous_rows)
    manifest = {
        "protocol": "zagreus-exam-bank-v1-ingest",
        "status": "INGESTED_AWAITING_DECONTAMINATION",
        "whitelist": {
            "path": str(whitelist_path),
            "sha256": sha256_file(whitelist_path),
        },
        "rights_scope": whitelist["rights_policy"],
        "sources": source_manifests,
        "parser": {
            "command": "pdftotext -layout",
            "key_derivation": "isolated final a-d column on each official table row",
            "excluded_subjects": "passage-based A-C and math/science; mixed H is retained only by the frozen historical-only rule",
            "accepted_rows_have_reconstructed_key": True,
        },
        "counts": {
            **dict(sorted(aggregate.items())),
            "parsed_occurrences": len(all_rows),
            "closed_world_key_conflict_occurrences": len(ambiguous_rows),
            "duplicate_occurrences_collapsed": duplicate_occurrences,
            "unique_rows": len(clean_rows),
            "unique_by_domain": dict(sorted(Counter(row["domain"] for row in clean_rows).items())),
        },
        "private_outputs": {
            "ingested_unique": {
                "path": str(private_dir / "ingested_unique.jsonl"),
                "sha256": sha256_file(private_dir / "ingested_unique.jsonl"),
            },
            "closed_world_key_conflicts": {
                "path": str(private_dir / "closed_world_key_conflicts.jsonl"),
                "sha256": sha256_file(private_dir / "closed_world_key_conflicts.jsonl"),
            },
        },
        "redistribution": "Do not redistribute the private outputs or source archives.",
    }
    write_json(public_dir / "ingest_manifest.json", manifest)
    return manifest


PRIOR_POOL_NAME_RE = re.compile(
    r"(?:^|_)(?:train|training|pool|remaining|clean_pool)(?:_|\.|$)", re.IGNORECASE
)


def discover_prior_pool_paths(root: Path, excluded_root: Path) -> list[Path]:
    paths: list[Path] = []
    excluded = excluded_root.resolve()
    for path in root.rglob("*.jsonl"):
        resolved = path.resolve()
        if excluded == resolved or excluded in resolved.parents:
            continue
        lowered = str(path).casefold()
        if any(value in lowered for value in ("prediction", "official", "holdout", "test.jsonl")):
            continue
        if PRIOR_POOL_NAME_RE.search(path.name) or "source_pool" in path.name.casefold():
            paths.append(path)
    return sorted(set(paths), key=lambda path: str(path))


def row_question_surfaces(row: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("question", "source_question", "prompt"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value)
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, Mapping) or message.get("role") != "user":
                continue
            value = message.get("content")
            if isinstance(value, str) and value.strip():
                values.append(value)
    return values


def load_prior_hashes(paths: Sequence[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    hashes: set[str] = set()
    manifest: list[dict[str, Any]] = []
    for path in paths:
        rows = 0
        surfaces = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    rows += 1
                    source = json.loads(line)
                    for value in row_question_surfaces(source):
                        hashes.add(normalized_question_hash(value))
                        surfaces += 1
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot deterministically scan prior pool {path}: {error}") from error
        manifest.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": rows,
            "question_surfaces": surfaces,
        })
    return hashes, manifest


def option_value(value: object) -> str:
    if isinstance(value, Mapping):
        if "value" in value:
            return str(value["value"])
        if len(value) == 1:
            return str(next(iter(value.values())))
        raise ValueError(f"Unsupported option dictionary schema: {sorted(value)}")
    return str(value or "")


def load_official_rows(path: Path) -> list[dict[str, Any]]:
    expected = "ca877846f19a6d781ba151382e4f43b10efad1a7e375b5e0c50b047a3917e0af"
    if sha256_file(path) != expected:
        raise ValueError("Unexpected official canonical hash")
    rows = read_jsonl(path)
    if len(rows) != 10_000:
        raise ValueError(f"Expected 10,000 official rows, found {len(rows)}")
    return [{
        "question": str(row["question"]),
        "options": [option_value(value) for value in row["options"]],
        "question_hash": normalized_question_hash(row["question"]),
    } for row in rows]


def build_inverted(values: Sequence[frozenset[str]]) -> dict[str, list[int]]:
    output: dict[str, list[int]] = defaultdict(list)
    for index, tokens in enumerate(values):
        for token in tokens:
            output[token].append(index)
    return output


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def lexical_screen(
    input_path: Path,
    official_path: Path,
    prior_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    rows = read_jsonl(input_path)
    official = load_official_rows(official_path)
    prior_paths = discover_prior_pool_paths(prior_root, output_dir)
    prior_hashes, prior_manifest = load_prior_hashes(prior_paths)
    official_hashes = {row["question_hash"] for row in official}

    official_shingles = [word_shingles(row["question"]) for row in official]
    official_words = [frozenset(content_words(row["question"])) for row in official]
    shingle_inverted = build_inverted(official_shingles)
    word_inverted = build_inverted(official_words)

    retained: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        question_hash = row["normalized_question_hash"]
        triggers: list[dict[str, Any]] = []
        if question_hash in official_hashes:
            triggers.append({"kind": "official_normalized_exact", "score": 1.0})
        if question_hash in prior_hashes:
            triggers.append({"kind": "prior_pool_normalized_exact", "score": 1.0})

        candidate_shingles = word_shingles(row["question"])
        shingle_candidates: set[int] = set()
        for token in candidate_shingles:
            shingle_candidates.update(shingle_inverted.get(token, ()))
        max_shingle = 0.0
        max_shingle_index: int | None = None
        for index in shingle_candidates:
            score = jaccard(candidate_shingles, official_shingles[index])
            if score > max_shingle:
                max_shingle, max_shingle_index = score, index
        # Exact set Jaccard is the full-recall confirmation of the frozen
        # MinHash-Jaccard rule; it removes approximation false negatives.
        if max_shingle >= 0.70 and max_shingle_index is not None:
            triggers.append({
                "kind": "minhash_jaccard_exact_confirmation",
                "score": max_shingle,
                "matched_official_question_hash": official[max_shingle_index]["question_hash"],
            })

        candidate_words = frozenset(content_words(row["question"]))
        max_containment = 0.0
        max_containment_index: int | None = None
        if len(candidate_words) >= 8:
            word_candidates: set[int] = set()
            for token in candidate_words:
                word_candidates.update(word_inverted.get(token, ()))
            for index in word_candidates:
                if min(len(candidate_words), len(official_words[index])) < 8:
                    continue
                score = containment(candidate_words, official_words[index])
                if score > max_containment:
                    max_containment, max_containment_index = score, index
            if max_containment >= 0.80 and max_containment_index is not None:
                triggers.append({
                    "kind": "content_word_containment",
                    "score": max_containment,
                    "matched_official_question_hash": official[max_containment_index]["question_hash"],
                })

        row["lexical_decontamination"] = {
            "max_official_shingle_jaccard": max_shingle,
            "max_official_content_word_containment": max_containment,
            "triggers": triggers,
        }
        if triggers:
            rejected.append(row)
            for trigger in triggers:
                counts[f"rejected:{trigger['kind']}"] += 1
        else:
            retained.append(row)
            counts[f"retained:{row['domain']}"] += 1

    # Construct conservative family groups before any split.  Exact question
    # hashes always co-group; high lexical similarity also co-groups.  The
    # embedding screen may union additional groups later.
    union = UnionFind(len(retained))
    by_question_hash: dict[tuple[str, str], int] = {}
    candidate_shingles: list[frozenset[str]] = []
    candidate_words: list[frozenset[str]] = []
    shingle_seen: dict[str, list[int]] = defaultdict(list)
    word_seen: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(retained):
        question_hash = row["normalized_question_hash"]
        domain = str(row["domain"])
        question_domain_key = (question_hash, domain)
        if question_domain_key in by_question_hash:
            union.union(index, by_question_hash[question_domain_key])
        else:
            by_question_hash[question_domain_key] = index
        shingles = word_shingles(row["question"])
        words = frozenset(content_words(row["question"]))
        lexical_neighbors: set[int] = set()
        for token in shingles:
            lexical_neighbors.update(shingle_seen.get(token, ()))
        for neighbor in lexical_neighbors:
            if (
                str(retained[neighbor]["domain"]) == domain
                and jaccard(shingles, candidate_shingles[neighbor]) >= 0.70
            ):
                union.union(index, neighbor)
        if len(words) >= 8:
            word_neighbors: set[int] = set()
            for token in words:
                word_neighbors.update(word_seen.get(token, ()))
            for neighbor in word_neighbors:
                if str(retained[neighbor]["domain"]) != domain:
                    continue
                if min(len(words), len(candidate_words[neighbor])) < 8:
                    continue
                if containment(words, candidate_words[neighbor]) >= 0.80:
                    union.union(index, neighbor)
        candidate_shingles.append(shingles)
        candidate_words.append(words)
        for token in shingles:
            shingle_seen[token].append(index)
        for token in words:
            word_seen[token].append(index)

    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(retained)):
        members_by_root[union.find(index)].append(index)
    for members in members_by_root.values():
        member_ids = sorted(retained[index]["id"] for index in members)
        group_id = hashlib.sha256("\n".join(member_ids).encode("utf-8")).hexdigest()
        for index in members:
            retained[index]["group_id"] = group_id

    private_dir = output_dir / "private_derived"
    evidence_dir = output_dir / "evidence"
    retained.sort(key=lambda row: row["id"])
    rejected.sort(key=lambda row: row["id"])
    retained_path = private_dir / "lexically_decontaminated.jsonl"
    rejected_path = private_dir / "lexical_rejections.jsonl"
    write_jsonl(retained_path, retained)
    write_jsonl(rejected_path, rejected)
    group_sizes = Counter(row["group_id"] for row in retained)
    manifest = {
        "protocol": "zagreus-exam-bank-v1-lexical-decontamination",
        "status": "LEXICAL_SCREEN_PASSED_AWAITING_EMBEDDING_SCREEN",
        "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
        "official": {"path": str(official_path), "sha256": sha256_file(official_path), "rows": len(official)},
        "prior_pools": prior_manifest,
        "thresholds": {
            "normalized_exact": 1.0,
            "minhash_compatible_word_3_shingle_jaccard": 0.70,
            "content_word_containment": 0.80,
            "content_word_minimum": 8,
        },
        "counts": {
            **dict(sorted(counts.items())),
            "input_rows": len(rows),
            "rejected_rows": len(rejected),
            "retained_rows": len(retained),
            "family_groups": len(group_sizes),
            "largest_family_group": max(group_sizes.values(), default=0),
            "retained_by_domain": dict(sorted(Counter(row["domain"] for row in retained).items())),
        },
        "private_outputs": {
            "retained": {"path": str(retained_path), "sha256": sha256_file(retained_path)},
            "rejected": {"path": str(rejected_path), "sha256": sha256_file(rejected_path)},
        },
    }
    write_json(evidence_dir / "lexical_decontamination_manifest.json", manifest)
    return manifest


def semantic_texts(row: Mapping[str, Any]) -> tuple[str, str]:
    question = str(row["question"]).strip()
    options = " || ".join(sorted(str(value).strip() for value in row["options"]))
    return question, f"{question}\n\n{options}"


def nearest_cosine(candidate: Any, reference: Any, batch_size: int) -> tuple[Any, Any]:
    import numpy as np

    scores = np.empty(candidate.shape[0], dtype=np.float32)
    indices = np.empty(candidate.shape[0], dtype=np.int64)
    reference_t = reference.T
    for start in range(0, candidate.shape[0], batch_size):
        end = min(start + batch_size, candidate.shape[0])
        matrix = candidate[start:end] @ reference_t
        local = matrix.argmax(axis=1)
        indices[start:end] = local
        scores[start:end] = matrix[np.arange(end - start), local]
    return scores, indices


def embedding_screen(
    input_path: Path,
    official_path: Path,
    model_path: Path,
    output_dir: Path,
    tail_packet: Path,
    batch_size: int,
    tail_size: int,
) -> dict[str, Any]:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if model_path.name != EMBEDDING_REVISION:
        raise ValueError(f"Expected pinned model revision directory {EMBEDDING_REVISION}")
    rows = read_jsonl(input_path)
    official = load_official_rows(official_path)
    model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
    official_pairs = [semantic_texts(row) for row in official]
    candidate_pairs = [semantic_texts(row) for row in rows]

    def encode(values: Sequence[str]) -> Any:
        return model.encode(
            list(values),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)

    official_q = encode([pair[0] for pair in official_pairs])
    official_full = encode([pair[1] for pair in official_pairs])
    candidate_q = encode([pair[0] for pair in candidate_pairs])
    candidate_full = encode([pair[1] for pair in candidate_pairs])
    q_scores, q_indices = nearest_cosine(candidate_q, official_q, 256)
    full_scores, full_indices = nearest_cosine(candidate_full, official_full, 256)

    retained: list[dict[str, Any]] = []
    retained_positions: list[int] = []
    rejected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for position, row in enumerate(rows):
        use_full = bool(full_scores[position] > q_scores[position])
        score = float(full_scores[position] if use_full else q_scores[position])
        official_index = int(full_indices[position] if use_full else q_indices[position])
        row["embedding_decontamination"] = {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_revision": EMBEDDING_REVISION,
            "max_cosine": score,
            "matched_representation": "question_plus_sorted_options" if use_full else "question",
            "matched_official_question_hash": official[official_index]["question_hash"],
            "threshold": EMBEDDING_THRESHOLD,
        }
        if score >= EMBEDDING_THRESHOLD:
            rejected.append(row)
            counts[f"rejected:{row['domain']}"] += 1
        else:
            retained.append(row)
            retained_positions.append(position)
            counts[f"retained:{row['domain']}"] += 1

    # Start from lexical family groups, then conservatively union candidate
    # pairs whose question or question+option embedding crosses the same 0.80
    # near-duplicate threshold.  This happens before split allocation.
    union = UnionFind(len(retained))
    first_by_group: dict[tuple[str, str], int] = {}
    for index, row in enumerate(retained):
        prior_group = str(row["group_id"])
        prior_group_domain = (prior_group, str(row["domain"]))
        if prior_group_domain in first_by_group:
            union.union(index, first_by_group[prior_group_domain])
        else:
            first_by_group[prior_group_domain] = index
    retained_q = candidate_q[np.asarray(retained_positions, dtype=np.int64)]
    retained_full = candidate_full[np.asarray(retained_positions, dtype=np.int64)]
    for start in range(0, len(retained), 128):
        end = min(start + 128, len(retained))
        q_matrix = retained_q[start:end] @ retained_q.T
        full_matrix = retained_full[start:end] @ retained_full.T
        for local in range(end - start):
            current = start + local
            neighbors = np.flatnonzero(
                np.maximum(q_matrix[local], full_matrix[local]) >= EMBEDDING_THRESHOLD
            )
            for neighbor in neighbors:
                neighbor_int = int(neighbor)
                if (
                    neighbor_int < current
                    and retained[neighbor_int]["domain"] == retained[current]["domain"]
                ):
                    union.union(current, neighbor_int)
    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(retained)):
        members_by_root[union.find(index)].append(index)
    for members in members_by_root.values():
        member_ids = sorted(retained[index]["id"] for index in members)
        group_id = hashlib.sha256("\n".join(member_ids).encode("utf-8")).hexdigest()
        for index in members:
            retained[index]["group_id"] = group_id

    private_dir = output_dir / "private_derived"
    evidence_dir = output_dir / "evidence"
    retained.sort(key=lambda row: row["id"])
    rejected.sort(key=lambda row: row["id"])
    retained_path = private_dir / "semantic_provisional_retained.jsonl"
    rejected_path = private_dir / "embedding_rejections.jsonl"
    write_jsonl(retained_path, retained)
    write_jsonl(rejected_path, rejected)

    tail = sorted(
        retained,
        key=lambda row: (-float(row["embedding_decontamination"]["max_cosine"]), row["id"]),
    )[:tail_size]
    official_by_hash = {row["question_hash"]: row for row in official}
    packet_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(tail, 1):
        matched_hash = row["embedding_decontamination"]["matched_official_question_hash"]
        matched = official_by_hash[matched_hash]
        audit_id = hashlib.sha256(
            f"zagreus-exam-v1-tail-1:{row['id']}:{matched_hash}".encode("utf-8")
        ).hexdigest()
        packet_rows.append({
            "audit_id": audit_id,
            "rank": rank,
            "candidate_id": row["id"],
            "cosine": row["embedding_decontamination"]["max_cosine"],
            "candidate_question": row["question"],
            "candidate_options": row["options"],
            "official_question": matched["question"],
            "official_options": matched["options"],
            "allowed_verdicts": ["distinct", "semantic_duplicate", "ambiguous"],
            "verdict": None,
            "notes": None,
        })
        evidence_rows.append({
            "audit_id": audit_id,
            "rank": rank,
            "candidate_id": row["id"],
            "candidate_question_hash": row["normalized_question_hash"],
            "matched_official_question_hash": matched_hash,
            "cosine": row["embedding_decontamination"]["max_cosine"],
            "status": "awaiting_disclosed_single_model_auditor",
        })
    write_jsonl(tail_packet, packet_rows)
    evidence_index = evidence_dir / "tail_audit_round_1_index.jsonl"
    write_jsonl(evidence_index, evidence_rows)
    group_sizes = Counter(row["group_id"] for row in retained)
    manifest = {
        "protocol": "zagreus-exam-bank-v1-embedding-screen",
        "status": "AWAITING_SINGLE_MODEL_TAIL_AUDIT",
        "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
        "official": {"path": str(official_path), "sha256": sha256_file(official_path)},
        "embedding": {
            "model": EMBEDDING_MODEL,
            "revision": EMBEDDING_REVISION,
            "local_snapshot": str(model_path),
            "local_snapshot_manifest_sha256": hashlib.sha256(
                "\n".join(
                    f"{sha256_file(path)}  {path.relative_to(model_path)}"
                    for path in sorted(model_path.rglob("*")) if path.is_file()
                ).encode("utf-8")
            ).hexdigest(),
            "sentence_transformers_version": "5.1.2",
            "threshold": EMBEDDING_THRESHOLD,
            "representations": ["question", "question_plus_sorted_options"],
            "device": "cpu",
        },
        "counts": {
            **dict(sorted(counts.items())),
            "input_rows": len(rows),
            "automatic_embedding_rejections": len(rejected),
            "provisional_retained": len(retained),
            "tail_audit_rows": len(tail),
            "family_groups": len(group_sizes),
            "largest_family_group": max(group_sizes.values(), default=0),
            "retained_by_domain": dict(sorted(Counter(row["domain"] for row in retained).items())),
        },
        "maximum_retained_cosine": max(
            (float(row["embedding_decontamination"]["max_cosine"]) for row in retained),
            default=None,
        ),
        "tail_packet": {
            "path": str(tail_packet),
            "sha256": sha256_file(tail_packet),
            "contains_official_text": True,
            "must_remain_under_tmp": True,
        },
        "public_tail_index": {"path": str(evidence_index), "sha256": sha256_file(evidence_index)},
        "private_outputs": {
            "provisional_retained": {"path": str(retained_path), "sha256": sha256_file(retained_path)},
            "automatic_rejections": {"path": str(rejected_path), "sha256": sha256_file(rejected_path)},
        },
    }
    write_json(evidence_dir / "embedding_screen_manifest.json", manifest)
    return manifest


def build_tail_packet(
    retained: Sequence[Mapping[str, Any]],
    official_path: Path,
    output_dir: Path,
    tail_packet: Path,
    round_number: int,
    tail_size: int,
    prior_adjudications: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Materialize a private tail packet and a text-free public index."""
    official = load_official_rows(official_path)
    official_by_hash = {row["question_hash"]: row for row in official}
    prior_distinct = {
        (str(row["candidate_id"]), str(row["matched_official_question_hash"])): row
        for row in prior_adjudications
        if row.get("verdict") == "distinct"
    }
    tail = sorted(
        retained,
        key=lambda row: (-float(row["embedding_decontamination"]["max_cosine"]), row["id"]),
    )[:tail_size]
    packet_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(tail, 1):
        matched_hash = str(
            row["embedding_decontamination"]["matched_official_question_hash"]
        )
        matched = official_by_hash[matched_hash]
        audit_id = hashlib.sha256(
            f"zagreus-exam-v1-tail-{round_number}:{row['id']}:{matched_hash}".encode("utf-8")
        ).hexdigest()
        prior = prior_distinct.get((str(row["id"]), matched_hash))
        packet_rows.append({
            "audit_id": audit_id,
            "round": round_number,
            "rank": rank,
            "candidate_id": row["id"],
            "cosine": row["embedding_decontamination"]["max_cosine"],
            "candidate_question": row["question"],
            "candidate_options": row["options"],
            "official_question": matched["question"],
            "official_options": matched["options"],
            "prior_distinct_audit_id": prior.get("audit_id") if prior else None,
            "allowed_verdicts": ["distinct", "semantic_duplicate", "ambiguous"],
            "verdict": None,
            "notes": None,
        })
        evidence_rows.append({
            "audit_id": audit_id,
            "round": round_number,
            "rank": rank,
            "candidate_id": row["id"],
            "candidate_question_hash": row["normalized_question_hash"],
            "matched_official_question_hash": matched_hash,
            "cosine": row["embedding_decontamination"]["max_cosine"],
            "prior_distinct_audit_id": prior.get("audit_id") if prior else None,
            "status": "awaiting_disclosed_single_model_auditor",
        })
    write_jsonl(tail_packet, packet_rows)
    evidence_index = output_dir / "evidence" / f"tail_audit_round_{round_number}_index.jsonl"
    write_jsonl(evidence_index, evidence_rows)
    return {
        "round": round_number,
        "rows": len(packet_rows),
        "new_pairs": sum(row["prior_distinct_audit_id"] is None for row in packet_rows),
        "inherited_pairs": sum(row["prior_distinct_audit_id"] is not None for row in packet_rows),
        "packet_path": str(tail_packet),
        "packet_sha256": sha256_file(tail_packet),
        "public_index_path": str(evidence_index),
        "public_index_sha256": sha256_file(evidence_index),
    }


def finalize_tail_audit(
    input_path: Path,
    official_path: Path,
    output_dir: Path,
    tail_packet: Path,
    decisions_path: Path,
    round_number: int,
    tail_size: int,
) -> dict[str, Any]:
    """Apply a disclosed compact audit record without inventing human reviewers."""
    rows = read_jsonl(input_path)
    packet = read_jsonl(tail_packet)
    decisions = read_json(decisions_path)
    if decisions.get("packet_sha256") != sha256_file(tail_packet):
        raise ValueError("Tail decision record does not match the frozen packet")
    if decisions.get("review_mode") != "single_model_auditor":
        raise ValueError("Only the disclosed single-model-auditor mode is valid")
    if decisions.get("auditor_id") != TAIL_AUDITOR_ID:
        raise ValueError("Tail auditor identity differs from the frozen identity")
    if decisions.get("attestation") != TAIL_DECISION_ATTESTATION:
        raise ValueError("Missing exact all-pairs inspection attestation")
    if decisions.get("default_verdict") != "distinct":
        raise ValueError("The compact decision format requires explicit default distinct")
    if int(decisions.get("reviewed_rows", -1)) != len(packet):
        raise ValueError("Reviewed-row count does not cover the packet")
    if len(packet) != tail_size:
        raise ValueError(f"Expected a full {tail_size}-row tail packet, found {len(packet)}")
    ranks = [int(row["rank"]) for row in packet]
    if ranks != list(range(1, len(packet) + 1)):
        raise ValueError("Tail packet ranks are not contiguous")

    non_distinct_raw = decisions.get("non_distinct", {})
    if not isinstance(non_distinct_raw, dict):
        raise ValueError("non_distinct must map ranks to verdict records")
    non_distinct: dict[int, Mapping[str, Any]] = {}
    for raw_rank, record in non_distinct_raw.items():
        rank = int(raw_rank)
        if rank < 1 or rank > len(packet):
            raise ValueError(f"Decision rank out of range: {rank}")
        if not isinstance(record, dict) or record.get("verdict") not in {
            "semantic_duplicate", "ambiguous"
        }:
            raise ValueError(f"Invalid non-distinct verdict at rank {rank}")
        if not str(record.get("notes", "")).strip():
            raise ValueError(f"Missing notes for non-distinct rank {rank}")
        non_distinct[rank] = record

    by_id = {str(row["id"]): row for row in rows}
    failed_group_ids: set[str] = set()
    adjudications: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    verdict_counts: Counter[str] = Counter()
    new_reviewed = 0
    for audit_row in packet:
        rank = int(audit_row["rank"])
        candidate_id = str(audit_row["candidate_id"])
        if candidate_id not in by_id:
            raise ValueError(f"Tail candidate is absent from input: {candidate_id}")
        decision = non_distinct.get(rank)
        verdict = str(decision["verdict"]) if decision else "distinct"
        notes = str(decision["notes"]) if decision else "Inspected; no same-fact or ambiguous overlap found."
        inherited = bool(audit_row.get("prior_distinct_audit_id")) and decision is None
        review_source = "inherited_unchanged_distinct_pair" if inherited else "new_single_model_review"
        if not inherited:
            new_reviewed += 1
        if verdict != "distinct":
            failed_group_ids.add(str(by_id[candidate_id]["group_id"]))
        matched_hash = str(
            by_id[candidate_id]["embedding_decontamination"]["matched_official_question_hash"]
        )
        record = {
            "audit_id": audit_row["audit_id"],
            "round": round_number,
            "rank": rank,
            "candidate_id": candidate_id,
            "candidate_question_hash": by_id[candidate_id]["normalized_question_hash"],
            "matched_official_question_hash": matched_hash,
            "cosine": audit_row["cosine"],
            "verdict": verdict,
            "notes": notes,
            "review_mode": "single_model_auditor",
            "auditor_id": TAIL_AUDITOR_ID,
            "review_source": review_source,
            "prior_distinct_audit_id": audit_row.get("prior_distinct_audit_id"),
        }
        adjudications.append(record)
        public_rows.append({key: value for key, value in record.items() if key != "notes"})
        verdict_counts[verdict] += 1

    removed = [row for row in rows if str(row["group_id"]) in failed_group_ids]
    retained = [row for row in rows if str(row["group_id"]) not in failed_group_ids]
    private_dir = output_dir / "private_derived"
    evidence_dir = output_dir / "evidence"
    adjudication_path = private_dir / f"tail_audit_round_{round_number}_adjudication.jsonl"
    public_path = evidence_dir / f"tail_audit_round_{round_number}_adjudication.jsonl"
    retained_path = private_dir / f"retained_after_tail_round_{round_number}.jsonl"
    removed_path = private_dir / f"removed_by_tail_round_{round_number}.jsonl"
    write_jsonl(adjudication_path, adjudications)
    write_jsonl(public_path, public_rows)
    write_jsonl(retained_path, retained)
    write_jsonl(removed_path, removed)

    zero_removal_pass = round_number >= 2 and not failed_group_ids
    status = "PASSED_ZERO_REMOVAL_ROUND" if zero_removal_pass else "NEXT_ROUND_REQUIRED"
    next_packet_manifest: dict[str, Any] | None = None
    if not zero_removal_pass:
        prior_adjudications: list[dict[str, Any]] = []
        for prior_round in range(1, round_number + 1):
            prior_path = private_dir / f"tail_audit_round_{prior_round}_adjudication.jsonl"
            if prior_path.exists():
                prior_adjudications.extend(read_jsonl(prior_path))
        next_packet = Path(f"/tmp/zagreus_exam_bank_v1_tail_round_{round_number + 1}.jsonl")
        next_packet_manifest = build_tail_packet(
            retained,
            official_path,
            output_dir,
            next_packet,
            round_number + 1,
            tail_size,
            prior_adjudications,
        )

    manifest = {
        "protocol": "zagreus-exam-bank-v1-disclosed-single-model-tail-audit",
        "status": status,
        "round": round_number,
        "review_mode": "single_model_auditor",
        "auditor_id": TAIL_AUDITOR_ID,
        "no_independent_human_review_claimed": True,
        "packet": {"path": str(tail_packet), "sha256": sha256_file(tail_packet)},
        "decisions": {"path": str(decisions_path), "sha256": sha256_file(decisions_path)},
        "counts": {
            "packet_rows": len(packet),
            "new_model_reviews": new_reviewed,
            "inherited_unchanged_pairs": len(packet) - new_reviewed,
            "failed_candidate_rows_in_packet": len(non_distinct),
            "removed_family_groups": len(failed_group_ids),
            "removed_rows_in_linked_groups": len(removed),
            "retained_rows": len(retained),
            "verdicts": dict(sorted(verdict_counts.items())),
            "removed_by_domain": dict(sorted(Counter(row["domain"] for row in removed).items())),
            "retained_by_domain": dict(sorted(Counter(row["domain"] for row in retained).items())),
        },
        "zero_removal_round_passed": zero_removal_pass,
        "private_outputs": {
            "adjudication": {"path": str(adjudication_path), "sha256": sha256_file(adjudication_path)},
            "retained": {"path": str(retained_path), "sha256": sha256_file(retained_path)},
            "removed": {"path": str(removed_path), "sha256": sha256_file(removed_path)},
        },
        "public_adjudication": {"path": str(public_path), "sha256": sha256_file(public_path)},
        "next_packet": next_packet_manifest,
    }
    write_json(evidence_dir / f"tail_audit_round_{round_number}_manifest.json", manifest)
    return manifest


def stable_allocation_order(group_id: str, domain: str, split: str) -> str:
    return hashlib.sha256(
        f"{ALLOCATION_SEED}:{domain}:{split}:{group_id}".encode("utf-8")
    ).hexdigest()


def exact_group_subset(
    groups: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    target: int,
    domain: str,
    split: str,
) -> set[str] | None:
    ordered = sorted(
        groups,
        key=lambda item: (stable_allocation_order(item[0], domain, split), item[0]),
    )
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for group_id, members in ordered:
        size = len(members)
        if size > target:
            continue
        for subtotal, chosen in sorted(list(reachable.items()), reverse=True):
            new_total = subtotal + size
            if new_total > target or new_total in reachable:
                continue
            reachable[new_total] = (*chosen, group_id)
        if target in reachable:
            return set(reachable[target])
    return None


def allocate_splits(
    input_path: Path,
    output_dir: Path,
    passed_tail_manifest: Path,
) -> dict[str, Any]:
    tail_manifest = read_json(passed_tail_manifest)
    if tail_manifest.get("status") != "PASSED_ZERO_REMOVAL_ROUND":
        raise ValueError("Allocation requires a passed zero-removal tail round")
    if tail_manifest["private_outputs"]["retained"]["sha256"] != sha256_file(input_path):
        raise ValueError("Allocation input differs from the tail-passed retained pool")
    input_rows = read_jsonl(input_path)
    parser_artifact_row_ids = {
        str(row["id"])
        for row in input_rows
        if any(PAGE_FOOTER_ARTIFACT_RE.search(str(option).strip()) for option in row["options"])
    }
    parser_artifact_rows = [
        row for row in input_rows if str(row["id"]) in parser_artifact_row_ids
    ]
    rows = [
        row for row in input_rows if str(row["id"]) not in parser_artifact_row_ids
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["group_id"])].append(row)

    domains_by_question_hash: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        domains_by_question_hash[str(row["normalized_question_hash"])].add(str(row["domain"]))
    cross_domain_question_hashes = {
        question_hash for question_hash, domains in domains_by_question_hash.items()
        if len(domains) > 1
    }
    cross_domain_stem_groups = {
        str(row["group_id"]) for row in rows
        if str(row["normalized_question_hash"]) in cross_domain_question_hashes
    }

    cross_domain_groups = {
        group_id for group_id, members in groups.items()
        if len({str(row["domain"]) for row in members}) != 1
    }
    clean_groups = {
        group_id: members for group_id, members in groups.items()
        if group_id not in cross_domain_groups and group_id not in cross_domain_stem_groups
    }
    groups_by_domain: dict[str, list[tuple[str, Sequence[Mapping[str, Any]]]]] = defaultdict(list)
    for group_id, members in clean_groups.items():
        groups_by_domain[str(members[0]["domain"])].append((group_id, members))

    required_per_domain = sum(SPLIT_DOMAIN_QUOTAS.values())
    eligible_domains = sorted(
        domain for domain, domain_groups in groups_by_domain.items()
        if sum(len(members) for _, members in domain_groups) >= required_per_domain
    )
    allocations: dict[str, list[dict[str, Any]]] = {
        "exam_diag": [], "dev_early": [], "final_holdout": [], "exam_train": []
    }
    used_groups: set[str] = set()
    successfully_allocated: list[str] = []
    allocation_failures: dict[str, str] = {}
    for domain in eligible_domains:
        domain_used: set[str] = set()
        domain_split_groups: dict[str, set[str]] = {}
        possible = True
        for split, quota in SPLIT_DOMAIN_QUOTAS.items():
            available = [
                item for item in groups_by_domain[domain]
                if item[0] not in domain_used
            ]
            chosen = exact_group_subset(available, quota, domain, split)
            if chosen is None:
                allocation_failures[domain] = f"no exact whole-group subset for {split}={quota}"
                possible = False
                break
            domain_split_groups[split] = chosen
            domain_used.update(chosen)
        if not possible:
            continue
        successfully_allocated.append(domain)
        used_groups.update(domain_used)
        for split, chosen in domain_split_groups.items():
            for group_id in sorted(chosen):
                allocations[split].extend(clean_groups[group_id])

    for domain in successfully_allocated:
        for group_id, members in groups_by_domain[domain]:
            if group_id not in used_groups:
                allocations["exam_train"].extend(members)

    split_dir = output_dir / "private_derived" / "splits"
    for split, split_rows in allocations.items():
        split_rows.sort(key=lambda row: str(row["id"]))
        write_jsonl(split_dir / f"{split}.jsonl", split_rows)

    group_sets = {
        split: {str(row["group_id"]) for row in split_rows}
        for split, split_rows in allocations.items()
    }
    question_sets = {
        split: {str(row["normalized_question_hash"]) for row in split_rows}
        for split, split_rows in allocations.items()
    }
    group_overlaps: dict[str, int] = {}
    question_overlaps: dict[str, int] = {}
    split_names = list(allocations)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1:]:
            pair = f"{left}__{right}"
            group_overlaps[pair] = len(group_sets[left] & group_sets[right])
            question_overlaps[pair] = len(question_sets[left] & question_sets[right])
    if any(group_overlaps.values()) or any(question_overlaps.values()):
        raise AssertionError("Split overlap invariant failed")

    split_counts = {split: len(split_rows) for split, split_rows in allocations.items()}
    split_by_domain = {
        split: dict(sorted(Counter(row["domain"] for row in split_rows).items()))
        for split, split_rows in allocations.items()
    }
    gate_checks = {
        "tail_zero_removal_round_passed": True,
        "at_least_six_allocated_sectors": len(successfully_allocated) >= 6,
        "exam_diag_at_least_600": split_counts["exam_diag"] >= 600,
        "dev_early_at_least_600": split_counts["dev_early"] >= 600,
        "final_holdout_at_least_900": split_counts["final_holdout"] >= 900,
        "exam_train_at_least_2000": split_counts["exam_train"] >= 2000,
        "zero_group_overlap": not any(group_overlaps.values()),
        "zero_normalized_question_overlap": not any(question_overlaps.values()),
        "all_rows_below_embedding_threshold": all(
            float(row["embedding_decontamination"]["max_cosine"]) < EMBEDDING_THRESHOLD
            for split_rows in allocations.values() for row in split_rows
        ),
        "all_keys_mechanically_reconstructed": all(
            row.get("answer_derivation") == "official_pdf_final_isolated_letter_column"
            and isinstance(row.get("gold"), int) and 0 <= int(row["gold"]) < 4
            for split_rows in allocations.values() for row in split_rows
        ),
        "zero_pdf_footer_artifacts": all(
            not any(PAGE_FOOTER_ARTIFACT_RE.search(str(option).strip()) for option in row["options"])
            for split_rows in allocations.values() for row in split_rows
        ),
    }
    manifest = {
        "protocol": "zagreus-exam-bank-v1-group-disjoint-allocation",
        "status": "PASSED" if all(gate_checks.values()) else "FAILED",
        "allocation_seed": ALLOCATION_SEED,
        "quotas_per_allocated_domain": SPLIT_DOMAIN_QUOTAS,
        "eligible_domains_before_subset_solve": eligible_domains,
        "allocated_domains": successfully_allocated,
        "allocation_failures": allocation_failures,
        "excluded_cross_domain_groups": len(cross_domain_groups),
        "excluded_cross_domain_normalized_stems": len(cross_domain_question_hashes),
        "excluded_groups_linked_to_cross_domain_stems": len(cross_domain_stem_groups),
        "excluded_pdf_footer_artifact_rows": len(parser_artifact_rows),
        "counts": {
            "splits": split_counts,
            "by_domain": split_by_domain,
            "source_rows_after_tail": len(input_rows),
            "source_rows_after_parser_artifact_filter": len(rows),
            "rows_in_allocated_domains": sum(split_counts.values()),
        },
        "overlaps": {
            "groups": group_overlaps,
            "normalized_questions": question_overlaps,
        },
        "gate_0_checks_available_at_allocation": gate_checks,
        "private_splits": {
            split: {
                "path": str(split_dir / f"{split}.jsonl"),
                "sha256": sha256_file(split_dir / f"{split}.jsonl"),
            }
            for split in allocations
        },
    }
    evidence_path = output_dir / "evidence" / "allocation_manifest.json"
    write_json(evidence_path, manifest)
    return manifest


def public_json_has_private_text_fields(path: Path) -> list[str]:
    forbidden = {"question", "options", "candidate_question", "candidate_options",
                 "official_question", "official_options", "gold_text"}
    violations: list[str] = []

    def visit(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_location = f"{location}.{key}"
                if str(key) in forbidden:
                    violations.append(child_location)
                visit(child, child_location)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{location}[{index}]")

    if path.suffix == ".jsonl":
        for line_number, row in enumerate(read_jsonl(path), 1):
            visit(row, f"{path}:{line_number}")
    else:
        visit(read_json(path), str(path))
    return violations


def finalize_gate_0(
    whitelist_path: Path,
    source_dir: Path,
    official_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    evidence_dir = output_dir / "evidence"
    allocation = read_json(evidence_dir / "allocation_manifest.json")
    ingest_manifest = read_json(evidence_dir / "ingest_manifest.json")
    lexical_manifest = read_json(evidence_dir / "lexical_decontamination_manifest.json")
    embedding_manifest = read_json(evidence_dir / "embedding_screen_manifest.json")
    tail_manifest = read_json(evidence_dir / "tail_audit_round_3_manifest.json")
    whitelist = read_json(whitelist_path)
    if sha256_file(whitelist_path) != EXPECTED_WHITELIST_SHA256:
        raise ValueError("Gate 0 whitelist differs from its frozen hash")

    splits = {
        name: read_jsonl(Path(record["path"]))
        for name, record in allocation["private_splits"].items()
    }
    split_hashes_match = all(
        sha256_file(Path(record["path"])) == record["sha256"]
        for record in allocation["private_splits"].values()
    )
    official_hashes = {row["question_hash"] for row in load_official_rows(official_path)}
    split_rows = [row for rows in splits.values() for row in rows]
    exact_official_overlap = sum(
        str(row["normalized_question_hash"]) in official_hashes for row in split_rows
    )
    lexical_trigger_rows = sum(
        bool(row["lexical_decontamination"]["triggers"]) for row in split_rows
    )
    embedding_trigger_rows = sum(
        float(row["embedding_decontamination"]["max_cosine"]) >= EMBEDDING_THRESHOLD
        for row in split_rows
    )

    allowed_prefixes = tuple(str(value) for value in whitelist["allowed_url_prefixes"])
    source_urls_allowed = all(
        any(str(row["source_url"]).startswith(prefix) for prefix in allowed_prefixes)
        for row in split_rows
    )
    rights = whitelist["license_evidence"]
    rights_passed = bool(rights.get("local_training")) and bool(
        rights.get("artifact_redistribution")
    ) and not bool(whitelist["rights_policy"].get("permit_source_row_redistribution"))
    archive_by_source = {
        str(record["source_id"]): record for record in ingest_manifest["sources"]
    }
    archive_hashes_match = True
    for source_id, filename in SOURCE_FILES.items():
        source_record = archive_by_source.get(source_id)
        if source_record is None or sha256_file(source_dir / filename) != source_record["archive_sha256"]:
            archive_hashes_match = False

    prior_paths = [
        str(record["path"]) for record in lexical_manifest.get("prior_pools", [])
    ]
    denylist_absent = not any("5_shots.jsonl" in path for path in prior_paths)
    public_field_violations: list[str] = []
    for path in sorted(evidence_dir.rglob("*.json")) + sorted(evidence_dir.rglob("*.jsonl")):
        if "superseded" in path.parts:
            continue
        public_field_violations.extend(public_json_has_private_text_fields(path))

    harness_files = [
        Path("zagreus_frozen_italic_eval.py"),
        Path("colab_official_harness_check.py"),
        official_path,
    ]
    harness_manifest = {
        "protocol": "zagreus-runbook-v2-harness-freeze-before-paid-compute",
        "files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in harness_files
        ],
    }
    write_json(output_dir / "harness_hashes_before.json", harness_manifest)

    checks = {
        "source_whitelist_hash_frozen": sha256_file(whitelist_path) == EXPECTED_WHITELIST_SHA256,
        "source_rights_gate_passed": rights_passed,
        "source_archive_hashes_match_ingest_manifest": archive_hashes_match,
        "all_source_urls_under_allowed_prefixes": source_urls_allowed,
        "official_key_reconstruction_100_percent": all(
            row.get("answer_derivation") == "official_pdf_final_isolated_letter_column"
            and isinstance(row.get("gold"), int) and 0 <= int(row["gold"]) < 4
            for row in split_rows
        ),
        "zero_exact_official_overlap": exact_official_overlap == 0,
        "zero_lexical_decontamination_triggers": lexical_trigger_rows == 0,
        "zero_embedding_decontamination_triggers": embedding_trigger_rows == 0,
        "tail_zero_removal_round_passed": tail_manifest.get("status") == "PASSED_ZERO_REMOVAL_ROUND",
        "allocation_manifest_passed": allocation.get("status") == "PASSED",
        "allocation_checks_all_passed": all(
            allocation.get("gate_0_checks_available_at_allocation", {}).values()
        ),
        "split_hashes_match_allocation_manifest": split_hashes_match,
        "five_shots_denylist_absent_from_prior_scan": denylist_absent,
        "public_evidence_contains_no_question_or_option_fields": not public_field_violations,
    }
    manifest = {
        "protocol": "zagreus-runbook-v2-gate-0",
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "counts": {
            "sectors": len(allocation["allocated_domains"]),
            "split_rows": {name: len(rows) for name, rows in splits.items()},
            "exact_official_overlap": exact_official_overlap,
            "lexical_trigger_rows": lexical_trigger_rows,
            "embedding_trigger_rows": embedding_trigger_rows,
            "public_private_text_field_violations": len(public_field_violations),
        },
        "violations": {"public_private_text_fields": public_field_violations},
        "inputs": {
            "whitelist": {"path": str(whitelist_path), "sha256": sha256_file(whitelist_path)},
            "allocation_manifest": {
                "path": str(evidence_dir / "allocation_manifest.json"),
                "sha256": sha256_file(evidence_dir / "allocation_manifest.json"),
            },
            "tail_manifest": {
                "path": str(evidence_dir / "tail_audit_round_3_manifest.json"),
                "sha256": sha256_file(evidence_dir / "tail_audit_round_3_manifest.json"),
            },
        },
        "harness_freeze": {
            "path": str(output_dir / "harness_hashes_before.json"),
            "sha256": sha256_file(output_dir / "harness_hashes_before.json"),
        },
        "gpu_spend_authorized": all(checks.values()),
        "privacy": "Source rows and official audit text remain private and are not redistributable.",
    }
    write_json(evidence_dir / "GATE_0.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument(
        "--whitelist", type=Path,
        default=Path("results/zagreus_exam_bank_v1/SOURCE_WHITELIST.json"),
    )
    ingest_parser.add_argument(
        "--source-dir", type=Path,
        default=Path("results/zagreus_exam_bank_v1/private_sources"),
    )
    ingest_parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/zagreus_exam_bank_v1"),
    )
    screen_parser = subparsers.add_parser("screen-lexical")
    screen_parser.add_argument(
        "--input", type=Path,
        default=Path("results/zagreus_exam_bank_v1/private_derived/ingested_unique.jsonl"),
    )
    screen_parser.add_argument(
        "--official", type=Path, default=Path("results/italic_canonical.jsonl")
    )
    screen_parser.add_argument("--prior-root", type=Path, default=Path("results"))
    screen_parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/zagreus_exam_bank_v1"),
    )
    embedding_parser = subparsers.add_parser("screen-embedding")
    embedding_parser.add_argument(
        "--input", type=Path,
        default=Path("results/zagreus_exam_bank_v1/private_derived/lexically_decontaminated.jsonl"),
    )
    embedding_parser.add_argument(
        "--official", type=Path, default=Path("results/italic_canonical.jsonl")
    )
    embedding_parser.add_argument(
        "--model-path", type=Path,
        default=Path(
            "results/zagreus_v4/model_cache/"
            "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/"
            f"snapshots/{EMBEDDING_REVISION}"
        ),
    )
    embedding_parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/zagreus_exam_bank_v1"),
    )
    embedding_parser.add_argument(
        "--tail-packet", type=Path,
        default=Path("/tmp/zagreus_exam_bank_v1_tail_round_1.jsonl"),
    )
    embedding_parser.add_argument("--batch-size", type=int, default=64)
    embedding_parser.add_argument("--tail-size", type=int, default=300)
    tail_parser = subparsers.add_parser("finalize-tail")
    tail_parser.add_argument("--round", type=int, required=True)
    tail_parser.add_argument("--input", type=Path, required=True)
    tail_parser.add_argument(
        "--official", type=Path, default=Path("results/italic_canonical.jsonl")
    )
    tail_parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/zagreus_exam_bank_v1"),
    )
    tail_parser.add_argument("--tail-packet", type=Path, required=True)
    tail_parser.add_argument("--decisions", type=Path, required=True)
    tail_parser.add_argument("--tail-size", type=int, default=300)
    allocation_parser = subparsers.add_parser("allocate")
    allocation_parser.add_argument("--input", type=Path, required=True)
    allocation_parser.add_argument("--passed-tail-manifest", type=Path, required=True)
    allocation_parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/zagreus_exam_bank_v1"),
    )
    gate_parser = subparsers.add_parser("gate-0")
    gate_parser.add_argument(
        "--whitelist", type=Path,
        default=Path("results/zagreus_exam_bank_v1/SOURCE_WHITELIST.json"),
    )
    gate_parser.add_argument(
        "--source-dir", type=Path,
        default=Path("results/zagreus_exam_bank_v1/private_sources"),
    )
    gate_parser.add_argument(
        "--official", type=Path, default=Path("results/italic_canonical.jsonl")
    )
    gate_parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/zagreus_exam_bank_v1"),
    )
    args = parser.parse_args()
    if args.command == "ingest":
        manifest = ingest(args.whitelist, args.source_dir, args.output_dir)
        print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "screen-lexical":
        manifest = lexical_screen(args.input, args.official, args.prior_root, args.output_dir)
        print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "screen-embedding":
        manifest = embedding_screen(
            args.input,
            args.official,
            args.model_path,
            args.output_dir,
            args.tail_packet,
            args.batch_size,
            args.tail_size,
        )
        print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "finalize-tail":
        manifest = finalize_tail_audit(
            args.input,
            args.official,
            args.output_dir,
            args.tail_packet,
            args.decisions,
            args.round,
            args.tail_size,
        )
        print(json.dumps({
            "status": manifest["status"],
            "counts": manifest["counts"],
            "next_packet": manifest["next_packet"],
        }, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "allocate":
        manifest = allocate_splits(args.input, args.output_dir, args.passed_tail_manifest)
        print(json.dumps({
            "status": manifest["status"],
            "allocated_domains": manifest["allocated_domains"],
            "counts": manifest["counts"],
            "gate_checks": manifest["gate_0_checks_available_at_allocation"],
        }, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "gate-0":
        manifest = finalize_gate_0(
            args.whitelist, args.source_dir, args.official, args.output_dir
        )
        print(json.dumps({
            "status": manifest["status"],
            "checks": manifest["checks"],
            "counts": manifest["counts"],
            "gpu_spend_authorized": manifest["gpu_spend_authorized"],
        }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
