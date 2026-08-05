# Zagreus ITALIC 45.73

Zagreus ITALIC 45.73 is a 437.8M-parameter causal language model specialized
for Italian multiple-choice reasoning. It scored **45.73%** on the complete
10,000-question ITALIC evaluation.

**[Model weights](https://huggingface.co/mcsp/zagreus-italic-45.73)** ·
**[Training bundle](https://huggingface.co/datasets/mcsp/zagreus-italic-45.73-reproduction)** ·
**[Release](https://github.com/LudWittg/zagreus-italic-45.73/releases/latest)**

## Result

| Metric | Value |
|---|---:|
| Official ITALIC accuracy | **45.73%** |
| Evaluated rows | 10,000 |
| Valid answer rate | 99.97% |
| Exact one-character outputs | 99.95% |
| Mean output length | 1.0 character |

Evaluated weight SHA-256:

```text
0c0a42e642e009d6aff7dd6663707ee0f877218b826ffb1d4ab3abd46687a991
```

The run used deterministic decoding and the unmodified official ITALIC harness
at commit `92df420ff686babeea54e217b9f90f8471374916`. The tracked harness
tree matched its pre-run snapshot after evaluation. No official ITALIC
question, option, answer, or demonstration was used for training or teacher
labeling.

## Training method

Each training example is rendered online as a 12-message conversation:

```text
system instruction
  → 5 × (demonstration question → demonstration answer letter)
  → target question
  → target answer letter
```

The five demonstrations always come from the training pool. For every target
exposure, the renderer deterministically resamples the demonstrations, chooses
a 3-, 4-, or 5-option layout, permutes the options, balances the displayed
answer letter, and selects one of four teacher views. Repeated epochs therefore
do not replay identical sequences.

All five demonstration letters and the target letter are supervised. The model
also learns immediate EOS after the target answer.

### Data streams

Each epoch mixes:

- **21,000 signal rows (80%)** with four-view probability targets from
  `google/gemma-4-E4B-it`; 9,713 rows additionally carry authorized hard gold;
- **3,000 anchor rows**, resampled to **20% of epoch exposures**, with a frozen
  reference distribution from the initialization checkpoint.

The exact rows, posterior targets, anchor rows, and initialization files are in
the [reproduction bundle](https://huggingface.co/datasets/mcsp/zagreus-italic-45.73-reproduction).

### Objective

For signal rows:

```text
L_target = 0.5 L_gold-CE + 0.5 L_E4B-KD(T=2)   when authorized gold exists
L_target = L_E4B-KD(T=2)                        otherwise

L_signal = (5 L_demo-CE + L_target) / 6 + 0.1 L_EOS
```

The temperature-2 KD term uses `T²` scaling. Option permutations are inverted
before target CE/KD, so losses are computed in semantic-option space rather
than display-letter space.

For anchor rows:

```text
L_anchor = 2.0 × mean_turn KL(P_init || P_student), T=1
```

### Optimization

| Setting | Value |
|---|---:|
| Initialization | Published frozen checkpoint, SHA-256 `209c861e…` |
| Seed | `20260831` |
| Epochs | 4 |
| Optimizer steps | 6,564 |
| Effective batch | 16 |
| Peak learning rate | `1.5e-4` |
| Warmup | 5% linear |
| Schedule | Cosine to `1.5e-5` over all 6,564 steps |
| Optimizer | Fused AdamW |
| Weight decay | `0.05` |
| Gradient clipping | `1.0` |
| Precision | BF16 autocast, FP32 master weights |
| Hardware | 1 × NVIDIA H100 80GB |
| Tokens | 113,309,980 |
| Throughput | 34,785 tokens/s |
| **Training time** | **3,257 seconds — 54 minutes 17 seconds** |

The 54-minute figure is measured wall time for the released checkpoint's full
four-epoch training job; model download and the official evaluation are not
included.

### Total compute

| Component | Measured | GPU-hours |
|---|---:|---:|
| Training run, four epochs, 6,564 steps | 3,257.39 s | 0.905 |
| Teacher target generation — `google/gemma-4-E4B-it` revision `ee0ef6023621cff504d758262d4e04895a5af4a2`, 21,000 rows × 4 views | 0.985 h | 0.985 |
| Corpus construction and decontamination | CPU only | 0 |
| Synthetic question generation | none | 0 |
| **Total** | | **1.890** |

Both measured figures are wall time from the run manifests, not estimates:
the training run from `evidence/training_summary.json`, the teacher labeling
pass from its phase ledger. Figures cover the released checkpoint's own
training and the data generation it consumed; initialization is the published
frozen checkpoint `209c861e…` listed above.

**No synthetic question content was generated for this model.** All training
rows come from real, decontaminated sources — `efederici/pinocchio` under
item-level screening, officially answer-keyed exam banks, and deterministic
source-derived replay. The only generated artifact is the teacher's soft
probability distribution over answer letters.

### Efficient supervised-position projection

The transformer processes every prompt token, but only seven positions require
full vocabulary logits: five demonstration letters, the target letter, and
target EOS. The trainer gathers those hidden states before applying the
128,256-way vocabulary projection. Dynamic padding, length-bucketed batches,
disabled gradient checkpointing, and token-budgeted microbatches provide the
remaining throughput improvement without changing effective-batch weights.

## Reproduce the terminal training run

The repository now contains the complete Python dependency closure for the
trainer. The companion Hugging Face bundle contains the exact hash-locked data
and frozen initialization checkpoint.

The reference environment was Python 3.12.3, CUDA 12.8, PyTorch 2.8.0,
Transformers 5.13.1, and one H100 80GB. On RunPod, the source run used
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.

```bash
python -m pip install -r requirements-training.txt
python scripts/prepare_reproduction.py --output reproduction

python scripts/train.py \
  --seed 20260831 \
  --base-model reproduction/initialization \
  --signal reproduction/signal.jsonl \
  --anchors reproduction/anchors.jsonl \
  --output-dir reproduced-run \
  --epochs 4 \
  --batch-size 16 \
  --checkpoint-interval 500 \
  --max-length 4096 \
  --pool-multiple 8 \
  --max-padded-tokens 16000 \
  --skip-eval
```

`--skip-eval` excludes benchmark-shaped selection fixtures from the
reproduction package and saves the terminal checkpoint directly. Selection
evaluation does not contribute gradients. The expected terminal weight hash is
`0c0a42e642e009d6aff7dd6663707ee0f877218b826ffb1d4ab3abd46687a991`;
minor low-level nondeterminism can occur on a different CUDA stack or GPU.

The exact run settings are also machine-readable in
[`configs/final_run.json`](configs/final_run.json).

## Code map

- [`scripts/train.py`](scripts/train.py): online renderer, mixed objective,
  sparse-head path, and training loop.
- [`scripts/zagreus_ensemble_t3_corpus.py`](scripts/zagreus_ensemble_t3_corpus.py):
  deterministic option-count and option-layout transformations.
- [`scripts/zagreus_ensemble_train.py`](scripts/zagreus_ensemble_train.py):
  prompt rendering, loss helpers, evaluation helpers, and model saving.
- [`scripts/zagreus_format_probe.py`](scripts/zagreus_format_probe.py):
  matched-interface scoring utilities.
- [`scripts/zagreus_ensemble_t2_common.py`](scripts/zagreus_ensemble_t2_common.py):
  shared integrity and posterior utilities.
- [`scripts/prepare_reproduction.py`](scripts/prepare_reproduction.py):
  bundle download, decompression, and SHA-256 verification.
- [`scripts/official_eval.py`](scripts/official_eval.py): pinned aggregate
  evaluation wrapper and harness-integrity checks.
- [`evidence/training_summary.json`](evidence/training_summary.json): selected
  run settings and immutable hashes.
- [`evidence/official_summary.json`](evidence/official_summary.json): sanitized
  aggregate result.

Official row-level outputs and benchmark fixtures are intentionally excluded.

## Inference

```bash
python -m pip install -r requirements-serving.txt
python examples/inference.py
```

The model expects the five-shot conversational structure and returns one answer
letter. The included example is synthetic and contains no benchmark item. The
evaluated serving stack was `transformers==4.55.2` and `vllm==0.10.2`.

## Limitations and terms

- This is a benchmark-oriented multiple-choice model, not a general Italian
  assistant.
- The reported score applies to the official five-shot protocol; differently
  templated performance may be materially lower.
- Rare invalid answers remain possible.
- Distillation can preserve teacher mistakes and calibration bias.
- The reproduction bundle includes upstream material with non-commercial and
  attribution/share-alike restrictions; consult its dataset card before use.

The Hub safetensors inventory reports 560.9M tensor parameters because tied
input/output embeddings are stored twice. The model has 437.8M unique trainable
parameters.
