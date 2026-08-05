#!/usr/bin/env python3
"""Snapshot or run one locked, pinned ITALIC read for ENSEMBLE-v1."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


OFFICIAL_COMMIT = "92df420ff686babeea54e217b9f90f8471374916"
RUN_EVAL_SHA256 = "ed4e4c63109245146040b234018ffb720ed088e6d55aef074cd9755dc3ed94f1"
REQUIREMENTS_SHA256 = "00b1a10b994cdefdfae034c4af29f6716ef16c8432fda24ff9134232b7b28490"
HELPER_SHA256 = "85e652219734049af9108865b1eff286007e828e5b581e87b7784aa58a3823e9"
LOCKED_RUNNER_SHA256 = "34b85aab0d3d200706d557df07e4fa3b2c9846e2abc9f866f0d68e6528b22ddf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_hashes(repo: Path) -> dict[str, str]:
    tracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    result = {}
    for raw in tracked:
        if not raw:
            continue
        relative = raw.decode("utf-8")
        path = repo / relative
        if not path.is_file():
            raise RuntimeError(f"tracked harness file missing: {relative}")
        result[relative] = sha256(path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("snapshot", "run"))
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--read-root", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--expected-weight-sha")
    parser.add_argument("--standing", type=float, default=0.3904)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.shared_root / "ITALIC"
    helper_path = args.shared_root / "colab_official_interface_eval.py"
    runner = args.shared_root / "zagreus_italic_locked_runner.py"
    before_path = args.shared_root / "harness_hashes_before.json"
    if subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip() != OFFICIAL_COMMIT:
        raise ValueError("official commit drift")
    if sha256(repo / "run_eval.py") != RUN_EVAL_SHA256 or sha256(repo / "requirements.txt") != REQUIREMENTS_SHA256:
        raise ValueError("pinned harness hash drift")
    if sha256(helper_path) != HELPER_SHA256 or sha256(runner) != LOCKED_RUNNER_SHA256:
        raise ValueError("frozen helper hash drift")
    if args.mode == "snapshot":
        if before_path.exists():
            raise FileExistsError(f"refusing to overwrite {before_path}")
        hashes = repo_hashes(repo)
        before_path.write_text(json.dumps(hashes, indent=2) + "\n")
        print(json.dumps({"status": "PASS", "files": len(hashes), "official_rows_loaded": 0}))
        return
    if None in (args.read_root, args.model, args.candidate_id, args.expected_weight_sha):
        raise ValueError("run mode requires read-root, model, candidate-id, and expected-weight-sha")
    read_root: Path = args.read_root
    model: Path = args.model
    read_lock = read_root / "official_read.lock"
    data_lock = read_root / "official_data_open.lock"
    results = read_root / "official"
    if sha256(model / "model.safetensors") != args.expected_weight_sha:
        raise ValueError("candidate weight mismatch")
    if not read_lock.is_file() or data_lock.exists() or results.exists():
        raise RuntimeError("official boundary state invalid")
    before = json.loads(before_path.read_text())
    if repo_hashes(repo) != before:
        raise ValueError("harness drifted before official read")
    spec = importlib.util.spec_from_file_location("official_eval", helper_path)
    helper = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(helper)
    helper.ROOT = read_root
    helper.REPO = repo
    helper.RESULTS = results
    helper.FINAL_MODEL = model
    helper.CHAT_TEMPLATE = model / "chat_template.jinja"
    output_dir = results / args.candidate_id
    output_dir.mkdir(parents=True, exist_ok=False)
    served_name = f"zagreus-{args.candidate_id}-official"
    server = helper.start_server(str(model), served_name, output_dir / "vllm_server.log")
    started = time.perf_counter()
    try:
        environment = dict(
            os.environ,
            ZAGREUS_OFFICIAL_REPO=str(repo),
            ZAGREUS_OFFICIAL_DATA_LOCK=str(data_lock),
        )
        helper.run(
            [
                sys.executable, str(runner), "--config-name", "config.yaml",
                f"model={served_name}", "provider=custom_openai", "api_key=EMPTY",
                "provider_kwargs.base_url=http://127.0.0.1:8000/v1",
                f"data.output_dir={output_dir}",
            ],
            cwd=repo,
            env=environment,
        )
    finally:
        helper.stop_server(server)
        subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], check=False)
    result_path = helper.final_result(output_dir)
    payload = json.loads(result_path.read_text())
    if len(payload.get("results", [])) != 10_000 or not data_lock.is_file():
        raise RuntimeError("incomplete or unlocked official result")
    decomposition = helper.decompose(payload)
    compressed = result_path.with_suffix(".json.gz")
    with result_path.open("rb") as source, gzip.open(compressed, "wb", compresslevel=6) as target:
        shutil.copyfileobj(source, target)
    after = repo_hashes(repo)
    (read_root / "harness_hashes_final.json").write_text(json.dumps(after, indent=2) + "\n")
    if after != before:
        raise RuntimeError("harness tree changed during official read")
    accuracy = float(decomposition["official_accuracy"])
    summary = {
        "protocol": "zagreus-ensemble-v1-official",
        "candidate_id": args.candidate_id,
        "candidate_model_weight_sha256": args.expected_weight_sha,
        "official_commit": OFFICIAL_COMMIT,
        "official_rows_evaluated": 10_000,
        "candidate_official_accuracy": accuracy,
        "standing_official_accuracy": args.standing,
        "strictly_improves_standing_model": accuracy > args.standing,
        "promotion_rule": f"candidate_official_accuracy > {args.standing}",
        "official_metrics": payload["metrics"],
        "decomposition": decomposition,
        "elapsed_seconds": time.perf_counter() - started,
        "result_file": str(result_path.relative_to(read_root)),
        "compressed_result_file": str(compressed.relative_to(read_root)),
        "harness_hashes_equal": True,
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("ENSEMBLE_OFFICIAL_RESULT=" + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
