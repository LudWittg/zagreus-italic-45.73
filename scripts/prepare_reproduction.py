#!/usr/bin/env python3
"""Download and verify the exact public training bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


BUNDLE = "mcsp/zagreus-italic-45.73-reproduction"
EXPECTED = {
    "signal.jsonl": "648937a2eadcbd7a797cfe44fcf905bfb163a952ef1c24341f70484503c76cd5",
    "anchors.jsonl": "913023f76d46f968010baff871204705a67c854884adf91a3b80e9029857559f",
    "initialization/model.safetensors": "209c861e792c812e98d4287ae5c3cd6eaa6161e38342432cd83868786ec697b6",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"hash mismatch for {path}: {actual} != {expected}")


def unpack(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "rb") as compressed, destination.open("wb") as output:
        shutil.copyfileobj(compressed, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=BUNDLE)
    parser.add_argument("--output", type=Path, default=Path("reproduction"))
    args = parser.parse_args()

    snapshot = Path(snapshot_download(args.repo, repo_type="dataset"))
    args.output.mkdir(parents=True, exist_ok=True)
    unpack(snapshot / "data/signal.jsonl.gz", args.output / "signal.jsonl")
    unpack(snapshot / "data/anchors.jsonl.gz", args.output / "anchors.jsonl")
    shutil.copytree(snapshot / "initialization", args.output / "initialization", dirs_exist_ok=True)

    checked(args.output / "signal.jsonl", EXPECTED["signal.jsonl"])
    checked(args.output / "anchors.jsonl", EXPECTED["anchors.jsonl"])
    checked(
        args.output / "initialization/model.safetensors",
        EXPECTED["initialization/model.safetensors"],
    )
    print(f"Verified reproduction bundle in {args.output.resolve()}")


if __name__ == "__main__":
    main()
