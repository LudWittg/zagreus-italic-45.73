#!/usr/bin/env python3
"""Verify and bake the promoted checkpoint for its clean public repository."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_WEIGHT = "0c0a42e642e009d6aff7dd6663707ee0f877218b826ffb1d4ab3abd46687a991"
PLAIN = "{% for message in messages %}{{ message['content'] }}{% if not loop.last %}\\n\\n{% endif %}{% endfor %}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    model = args.model

    actual = sha256(model / "model.safetensors")
    if actual != EXPECTED_WEIGHT:
        raise ValueError(f"weight mismatch: {actual}")

    tokenizer = json.loads((model / "tokenizer_config.json").read_text())
    tokenizer["tokenizer_class"] = "PreTrainedTokenizerFast"
    tokenizer["chat_template"] = PLAIN
    tokenizer["clean_up_tokenization_spaces"] = False
    (model / "tokenizer_config.json").write_text(json.dumps(tokenizer, ensure_ascii=False, indent=2) + "\n")
    (model / "chat_template.jinja").write_text(PLAIN + "\n")

    generation = json.loads((model / "generation_config.json").read_text())
    generation.update({"do_sample": False, "max_new_tokens": 2})
    (model / "generation_config.json").write_text(json.dumps(generation, indent=2) + "\n")

    print(json.dumps({"model_weight_sha256": actual, "plain_template": True, "status": "PASS"}))


if __name__ == "__main__":
    main()
