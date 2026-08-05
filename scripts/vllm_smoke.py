#!/usr/bin/env python3
"""Synthetic-only vLLM smoke for one ENSEMBLE-v1 candidate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--helper", type=Path, required=True)
    args = parser.parse_args()
    if (args.root / "official_read.lock").exists() or (args.root / "official_data_open.lock").exists():
        raise FileExistsError("smoke must run before every official-data lock")
    spec = importlib.util.spec_from_file_location("official_helper", args.helper)
    helper = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(helper)
    helper.CHAT_TEMPLATE = args.model / "chat_template.jinja"
    served_name = f"zagreus-{args.candidate_id}-smoke"
    server = helper.start_server(
        str(args.model), served_name, args.root / f"{args.candidate_id}_vllm_smoke.log"
    )
    try:
        payload = {
            "model": served_name,
            "messages": [{"role": "user", "content": (
                "Rispondi soltanto con la lettera corretta. Quanto fa 1+1?\n"
                "A. 1\nB. 2\nC. 3\nD. 4\nRisposta:"
            )}],
            "temperature": 0,
            "max_tokens": 2,
        }
        request = urllib.request.Request(
            "http://127.0.0.1:8000/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
        output = result["choices"][0]["message"]["content"].strip()
        if not re.fullmatch(r"[A-D]", output):
            raise RuntimeError(f"vLLM smoke did not return one exact letter: {output!r}")
        manifest = {
            "protocol": "zagreus-ensemble-v1-vllm-synthetic-smoke",
            "candidate_id": args.candidate_id,
            "passed": True,
            "output": output,
            "official_rows_loaded": 0,
        }
        (args.root / f"{args.candidate_id}_vllm_smoke.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(json.dumps(manifest, sort_keys=True))
    finally:
        helper.stop_server(server)


if __name__ == "__main__":
    main()
