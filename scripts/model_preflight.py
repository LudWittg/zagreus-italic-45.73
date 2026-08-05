#!/usr/bin/env python3
"""Bake and verify one ENSEMBLE-v1 candidate before the official-data lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PLAIN = "{% for message in messages %}{{ message['content'] }}{% if not loop.last %}\n\n{% endif %}{% endfor %}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-weight-sha", required=True)
    parser.add_argument("--candidate-id", required=True)
    args = parser.parse_args()
    if (args.root / "official_read.lock").exists() or (args.root / "official_data_open.lock").exists():
        raise FileExistsError("model preflight must run before every official-data lock")
    actual = sha256(args.model / "model.safetensors")
    if actual != args.expected_weight_sha:
        raise ValueError(f"candidate weight mismatch: {actual}")
    tokenizer_config = json.loads((args.model / "tokenizer_config.json").read_text())
    tokenizer_class = tokenizer_config.get("tokenizer_class")
    if tokenizer_class not in {"TokenizersBackend", "PreTrainedTokenizerFast"}:
        raise ValueError(f"unexpected tokenizer class: {tokenizer_class}")
    tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"
    tokenizer_config["chat_template"] = PLAIN
    tokenizer_config["clean_up_tokenization_spaces"] = False
    (args.model / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n"
    )
    (args.model / "chat_template.jinja").write_text(PLAIN + "\n")
    generation = json.loads((args.model / "generation_config.json").read_text())
    generation.update({"do_sample": False, "max_new_tokens": 2})
    (args.model / "generation_config.json").write_text(json.dumps(generation, indent=2) + "\n")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Domanda sintetica\nA. uno\nB. due\nC. tre\nD. quattro"}],
        tokenize=False,
    )
    expected_render = "Domanda sintetica\nA. uno\nB. due\nC. tre\nD. quattro"
    if rendered.strip() != expected_render or any(
        marker in rendered for marker in ("<start_of_turn>", "<end_of_turn>", "assistant", "model")
    ):
        raise RuntimeError(f"plain-template render drift: {rendered!r}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, local_files_only=True
    )
    del model
    manifest = {
        "protocol": "zagreus-ensemble-v1-official-model-preflight",
        "candidate_id": args.candidate_id,
        "model_weight_sha256": actual,
        "plain_template": True,
        "tokenizer_class_compatibility": f"{tokenizer_class}->PreTrainedTokenizerFast",
        "round_trip": True,
        "deterministic_max_new_tokens": 2,
        "official_rows_loaded": 0,
    }
    output = args.root / f"{args.candidate_id}_model_preflight.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
