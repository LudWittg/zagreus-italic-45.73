#!/usr/bin/env python3
"""Remote preflight and packaging for PROPOSITIONAL-v1 T1."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path("/workspace/zagreus_propositional_t1")
JOB = ROOT / "job"
RESULTS = ROOT / "results"
EGRESS = ROOT / "egress"
EXPECTED = {
    JOB / "t1_input.jsonl": "b499eefb0dfb5a443a01d1b5ef4dd92501d26f8204f7e83db06884c8f00c6bcd",
    JOB / "T1_INPUT_MANIFEST.json": "b0a4c05ba5242da251ad98d6c7804b3273e1536d9a6e165161f51fad4e9c9c8f",
    JOB / "GATE_T.json": "9ac3b5746d654684dad41d2b5b29b6eff19de6fe7431e9e8eb9063ba58bfdbd8",
    JOB / "zagreus_propositional_t1.py": "a02d4854722f964c4f5102014b27d4096ee1c6cbedb3eb172089edce625ce599",
    JOB / "zagreus_scaled_kd_teacher.py": "05e86b6de26baf8cd4c09ef852eb7c237ea902abcbf705bf911805595b2696f1",
    JOB / "zagreus_language_synthetic_data.py": "aa10540f81989e754a10fa021920484370b74274424eb0db3ed7356080aba680",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def preflight() -> None:
    hashes = {}
    for path, expected in EXPECTED.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise RuntimeError(f"Hash mismatch: {path}: {actual} != {expected}")
        hashes[str(path.relative_to(ROOT))] = actual
    gate_t = json.loads((JOB / "GATE_T.json").read_text())
    if gate_t.get("status") != "FALLBACK_TIMEBOX":
        raise RuntimeError("T1 fallback authorization missing")
    manifest = json.loads((JOB / "T1_INPUT_MANIFEST.json").read_text())
    if manifest.get("status") != "FROZEN_BEFORE_NEW_TEACHER_INFERENCE":
        raise RuntimeError("T1 input not frozen")
    write_json(RESULTS / "preflight.json", {"protocol": "prop-t1-preflight-v1", "hashes": hashes})


def run() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--break-system-packages", "--disable-pip-version-check", "transformers==5.13.1", "accelerate", "huggingface_hub", "sentencepiece"],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(JOB / "zagreus_propositional_t1.py"), "--input", str(JOB / "t1_input.jsonl"), "--output-dir", str(RESULTS / "t1"), "--batch-size", "24"],
        check=True,
    )


def package() -> None:
    gate = RESULTS / "t1" / "GATE_L.json"
    if not gate.is_file():
        raise FileNotFoundError(gate)
    EGRESS.mkdir(parents=True, exist_ok=True)
    archive = EGRESS / "zagreus_propositional_t1_evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(RESULTS, arcname="results")
    write_json(
        EGRESS / "FINAL_MANIFEST.json",
        {
            "protocol": "zagreus-propositional-v1-t1-package",
            "gate_l_sha256": sha256(gate),
            "labeled_sha256": sha256(RESULTS / "t1" / "t1_labeled.jsonl"),
            "evidence_archive_sha256": sha256(archive),
            "official_italic_rows_read_or_used": 0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run", "package", "all"))
    args = parser.parse_args()
    if args.command in ("preflight", "all"):
        preflight()
    if args.command in ("run", "all"):
        run()
    if args.command in ("package", "all"):
        package()


if __name__ == "__main__":
    main()
