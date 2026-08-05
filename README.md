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

The released checkpoint is the terminal link of a five-stage chain from
`mii-llm/zagreus-0.4B-ita`. Full per-stage accounting, code, and checkpoint
hashes are in [REPRODUCTION.md](REPRODUCTION.md).

| Component | GPU-hours |
|---|---:|
| Stage 1–5 training, base to released checkpoint | 5.070 |
| Teacher target generation (`google/gemma-4-E4B-it` rev `ee0ef602…`, 21,000 rows × 4 views) | 0.985 |
| Corpus construction and decontamination | 0 — CPU only |
| Synthetic question generation | 0 — none |
| **Total (measured)** | **6.055** |

The terminal run alone is 0.905 GPU-hours (3,257.39 s). Stages 1 and 2 also
used soft-KD and required their own labeling passes, which are not separately
itemised in the campaign records; including them places the full total below
7.6 GPU-hours. All figures are measured wall time from run manifests.

**No synthetic question content was generated for this model.** All training
rows come from real, decontaminated sources — `efederici/pinocchio` under
item-level screening, officially answer-keyed exam banks, and deterministic
source-derived replay. The only generated artifact is the teacher's soft
probability distribution over answer letters.

### Decontamination

No official ITALIC question, option, answer, or demonstration was used for
training or teacher labeling at any stage. The frozen screen, applied against
all 10,000 official rows, is:

1. recursive HTML-entity decoding and normalized exact match;
2. MinHash Jaccard ≥ 0.70 on word trigrams;
3. content-word containment ≥ 0.80 with ≥ 8 content words;
4. pinned MiniLM `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` cosine ≥ 0.80 on
   **both** the question and the question-plus-sorted-options surfaces;
5. a mechanical tail recheck required to return zero triggers.

Implementation: `stages/data/zagreus_final_instrument.py`, primitives in
`stages/data/zagreus_exam_bank_v1.py`, item-level driver in
`stages/data/zagreus_pinocchio_v4_itemlevel.py`.

**Coverage differs by component.** The Pinocchio pool (11,359 rows) and exam
bank (2,550) were screened prospectively at build time. The deterministic
replay slice (7,163) was built from Kaikki Italian Wiktionary, UD Italian ISDT
and Italian Wikipedia with ITALIC never an input, and is guarded at load by
`validate_clean()`, which rejects official-looking paths, row ids and sources.
It received no *similarity* screen at build time and was audited
retrospectively with the identical screens: **0 normalized-exact matches** and
6 lexical hits across the 10,370-row stage-1 split. Full breakdown and the
embedding-flag analysis are in [REPRODUCTION.md](REPRODUCTION.md#decontamination).

### Reproducing the teacher targets

`scripts/labeling/` contains the byte-identical code that produced the teacher
targets consumed by the training run. Hashes are the ones recorded in the
labeling job's own preflight manifest:

| File | SHA-256 (first 16) | Role |
|---|---|---|
| `zagreus_propositional_t1.py` | `a02d4854722f964c` | Labeling driver; enforces the locked teacher and I5 guards |
| `zagreus_scaled_kd_teacher.py` | `05e86b6de26baf8c` | Gemma inference core; emits four-view answer-letter log-probabilities |
| `zagreus_language_synthetic_data.py` | `aa10540f81989e75` | Imported **only** for `prompt_hash` and `sha256_file`; no generation path is used |
| `runpod_propositional_t1_job.py` | `e89573a42ccde71f` | Orchestration wrapper and preflight hash manifest |

The teacher identity is hardcoded and locked in
`zagreus_scaled_kd_teacher.py`:

```python
MODEL_ID       = "google/gemma-4-E4B-it"
MODEL_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
```

Passing a different model or revision raises
`"The teacher model and revision are locked for this experiment"`.

For each row the teacher scores four deterministic option permutations derived
from the row id, restricts the final-position logits to the displayed answer
letters, and stores `log_softmax` over those candidates. Only the letter
distribution is retained; no free text is generated and no row content is
modified. Input paths that look like official benchmark files are rejected
before any inference runs.

The third file is included because byte-identical reproduction requires the
import to resolve. Only its two hashing helpers are reachable from this path —
its generation routines are not called by the labeling pipeline and produced
none of this model's training data.

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

Install with [uv](https://github.com/astral-sh/uv) for a fast, resolved
environment:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements-training.txt
```

On the reference RunPod image torch 2.8.0+cu128 is preinstalled and the pin is
already satisfied. On a fresh machine, PyPI's default `torch==2.8.0` wheel may
carry a different CUDA build, so point uv at the CUDA 12.8 index to match the
reference environment:

```bash
uv pip install -r requirements-training.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

`python -m pip install -r requirements-training.txt` remains equivalent if uv
is unavailable — the pins, not the installer, define the environment. To harden
further, `uv pip compile requirements-training.txt -o requirements-training.lock`
produces a fully-pinned transitive lock.

Then prepare the data and run:

```bash
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
uv pip install -r requirements-serving.txt   # or: python -m pip install -r requirements-serving.txt
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
