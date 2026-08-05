#!/usr/bin/env python3
"""Remote preparation, ARM execution, and packaging for PROPOSITIONAL-v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path("/workspace/zagreus_propositional_train")
JOB = ROOT / "job"
UPLOADS = ROOT / "uploads"
RESULTS = ROOT / "results"
BASE_ARCHIVE = UPLOADS / "zagreus_scaled_kd_final_model.tar"
BASE_ROOT = ROOT / "base"
BASE_MODEL = BASE_ROOT / "zagreus_scaled_kd" / "final_model"
ADAPTER = JOB / "adapter"
P2F_MODEL = ROOT / "p2f_merged"
EXPECTED = {
    BASE_ARCHIVE: "a1b0be91dcf47517685596e3cb77009c3e80bfa1bd1e9f077fe7ae4316de2f25",
    JOB / "zagreus_propositional_train.py": "451c4d298810e0bc5c2ac4f642a3c971db01f237af6953a37d926c8feb8ee93f",
    JOB / "zagreus_scaled_kd_train.py": "f559fc210d8b63b1f45db259f3565796a7ef0c62a2d073ff1ffda354552bf9cd",
    JOB / "zagreus_interface_calibration.py": "43a0cb7eec48c6be6a7214c421ba898f931b4253d2020e5c8d688d11ac88601f",
    JOB / "zagreus_scaled_kd_teacher.py": "05e86b6de26baf8cd4c09ef852eb7c237ea902abcbf705bf911805595b2696f1",
    JOB / "zagreus_language_synthetic_data.py": "aa10540f81989e754a10fa021920484370b74274424eb0db3ed7356080aba680",
    JOB / "zagreus_p2f_official_merge.py": "b1552a06423223f097559749aea2eb29b8afdb21beb3b584709694508e8e8d7f",
    JOB / "TRAINING_PREDECLARATION.json": "900fae1d861d374590e99898503dc6b5cd4cd17eb38164f2aba08b637efa7b28",
    JOB / "T4_OPERATIONAL_AMENDMENT.json": "d8dbff1171347adb86585b4f541e7238555f25da9eb12261258e02f471250288",
    JOB / "train.jsonl": "ee11c843242316f864de291dfd357fed59f4248d40c5d99d2be2968c127f39e9",
    JOB / "anchors.jsonl": "913023f76d46f968010baff871204705a67c854884adf91a3b80e9029857559f",
    JOB / "dev.jsonl": "3f943adb939e50d53ebf28eeef84244cb0d4fd4dd29f7af97af924a3e1612732",
    JOB / "p2f_predictions.jsonl": "abde7cd4936901225de989620119f482740af816408dc45557c2d003a3a19bf4",
    ADAPTER / "adapter_model.safetensors": "da7004c751e243f92fff6c663697d61e4d664a09c28870aa8b4c17f14f7eca7f",
    ADAPTER / "adapter_config.json": "417a3c173d0e6929274e47ad0abeda3d3fe59301766fa026a20c157296f6ee5b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def prepare() -> None:
    hashes = {}
    for path, expected in EXPECTED.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Hash mismatch: {path}: {actual} != {expected}")
        hashes[str(path.relative_to(ROOT))] = actual
    if BASE_ROOT.exists() or P2F_MODEL.exists():
        raise FileExistsError("Refusing to overwrite prepared model directories")
    BASE_ROOT.mkdir(parents=True)
    with tarfile.open(BASE_ARCHIVE) as archive:
        archive.extractall(BASE_ROOT, filter="data")
    run([
        sys.executable, str(JOB / "zagreus_p2f_official_merge.py"),
        "--base-model", str(BASE_MODEL), "--adapter", str(ADAPTER), "--output", str(P2F_MODEL),
    ])
    merged = sha256(P2F_MODEL / "model.safetensors")
    if merged != "9ddff35c45d863344f4cdab314b5078280fb9da2a16be3018544a923a4388b7a":
        raise RuntimeError("P2F merge hash mismatch")
    write_json(RESULTS / "preflight.json", {
        "protocol": "zagreus-propositional-v1-remote-preflight",
        "payload_hashes": hashes,
        "merged_model_weight_sha256": merged,
        "official_italic_rows_read_or_used": 0,
    })


def train(arm: str) -> None:
    output = RESULTS / f"arm_{arm.lower()}"
    if output.exists():
        raise FileExistsError(output)
    run([
        sys.executable, str(JOB / "zagreus_propositional_train.py"),
        "--arm", arm, "--base-model", str(P2F_MODEL),
        "--train", str(JOB / "train.jsonl"), "--dev", str(JOB / "dev.jsonl"),
        "--anchors", str(JOB / "anchors.jsonl"),
        "--p2f-predictions", str(JOB / "p2f_predictions.jsonl"),
        "--output-dir", str(output), "--eval-batch-size", "64", "--max-length", "1024",
    ])


def package(arm: str) -> None:
    output = RESULTS / f"arm_{arm.lower()}"
    manifest = output / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    egress = ROOT / "egress"
    egress.mkdir(parents=True, exist_ok=True)
    evidence = egress / f"arm_{arm.lower()}_evidence.tar.gz"
    excluded = {output / "selected_model"}
    with tarfile.open(evidence, "w:gz") as archive:
        for path in sorted(RESULTS.rglob("*")):
            if any(path == item or item in path.parents for item in excluded):
                continue
            archive.add(path, arcname=f"results/{path.relative_to(RESULTS)}", recursive=False)
        archive.add(JOB / "TRAINING_PREDECLARATION.json", arcname="TRAINING_PREDECLARATION.json")
        amendment = JOB / ("T3_OPERATIONAL_AMENDMENT.json" if arm == "A" else "T4_OPERATIONAL_AMENDMENT.json")
        archive.add(amendment, arcname=amendment.name)
    selected = output / "selected_model"
    model_archive = None
    model_hash = None
    if selected.is_dir():
        model_archive = egress / f"arm_{arm.lower()}_selected_bf16.tar"
        with tarfile.open(model_archive, "w") as archive:
            archive.add(selected, arcname="final_model")
        model_hash = sha256(model_archive)
    write_json(egress / f"arm_{arm.lower()}_FINAL_MANIFEST.json", {
        "protocol": "zagreus-propositional-v1-arm-package",
        "arm": arm,
        "training_manifest_sha256": sha256(manifest),
        "evidence_archive_sha256": sha256(evidence),
        "selected_model_archive_sha256": model_hash,
        "selected_model_exists": model_archive is not None,
        "official_italic_rows_read_or_used": 0,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train", "package", "all"))
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    args = parser.parse_args()
    if args.command in ("prepare", "all"):
        prepare()
    if args.command in ("train", "all"):
        train(args.arm)
    if args.command in ("package", "all"):
        package(args.arm)


if __name__ == "__main__":
    main()
