# Zagreus ITALIC 45.73

Zagreus is a 437.8M-parameter causal language model specialized for Italian
multiple-choice evaluation. The promoted checkpoint scored **45.73% accuracy**
on the 10,000-item official ITALIC evaluation.

[Download the model on Hugging Face](https://huggingface.co/mcsp/zagreus-italic-45.73)

## Result

| Evaluation | Rows | Accuracy | Valid outputs | Exact one-character outputs |
|---|---:|---:|---:|---:|
| ITALIC official, pinned commit `92df420` | 10,000 | **45.73%** | 99.97% | 99.95% |

The evaluated model weight has SHA-256:

```text
0c0a42e642e009d6aff7dd6663707ee0f877218b826ffb1d4ab3abd46687a991
```

The official run used the unmodified benchmark harness, deterministic decoding,
and the benchmark's five-shot conversational prompt. The harness tree matched
its pre-lock snapshot after evaluation. No ITALIC row was used for training.

## Method

The checkpoint extends a format-matched initialization with four epochs of
full-parameter training over a 21k-example pool. Each sequence contains five
sampled demonstrations and one target. Demonstration sampling and option
permutation happen online; all six assistant letters are supervised.

Key settings:

- seed `20260831`, 6,564 optimizer steps;
- effective batch size 16 and 20% anchor sampling;
- peak LR `1.5e-4`, 5% warmup, cosine decay to `1.5e-5`;
- fused AdamW, weight decay `0.05`, gradient clipping `1.0`;
- BF16 autocast with FP32 master weights;
- dynamic padding and sparse vocabulary projection at supervised positions;
- no benchmark examples in the training pool.

The training run processed 113.3M tokens at 34,785 tokens/s on one H100 80GB.

## Reproduction assets

- [`scripts/train.py`](scripts/train.py): final online-augmentation trainer.
- [`scripts/model_preflight.py`](scripts/model_preflight.py): weight, tokenizer,
  plain-template, and model-load verification.
- [`scripts/vllm_smoke.py`](scripts/vllm_smoke.py): synthetic one-letter serving
  smoke test.
- [`evidence/official_summary.json`](evidence/official_summary.json): sanitized
  official aggregate result.
- [`evidence/training_summary.json`](evidence/training_summary.json): frozen
  training configuration and selected checkpoint metadata.
- [`model-card/README.md`](model-card/README.md): Hugging Face model card.

Official row-level outputs, benchmark fixtures, private training inputs, and
teacher labels are intentionally excluded.

## Inference

The model expects the benchmark-aligned multi-turn structure and returns an
answer letter. See [`examples/inference.py`](examples/inference.py) for a
synthetic five-shot example.

The frozen serving stack was `transformers==4.55.2` and `vllm==0.10.2`.

## Limitations

This is a narrow multiple-choice model, not a general Italian assistant. Its
45.73% score is specific to the official ITALIC protocol. It may emit an
invalid answer on rare inputs: the official valid rate was 99.97%. The model's
training-data package is not included in this release.

## Provenance

The public checkpoint is copied from immutable private source commit
`f3582df4c53417115cd8dc8bb074f1628c233a1e`, path
`candidates/final_push_track_c_seed_20260831_step6564`. Publication does not
alter `model.safetensors`; the SHA-256 above is the promotion identity.

