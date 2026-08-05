# Reproduction from the base model

The released 45.73% checkpoint is the terminal link of a five-stage chain
starting at `mii-llm/zagreus-0.4B-ita`. This document publishes every stage's
code, the checkpoint each stage produces, and the measured compute.

## The chain

| # | Stage | Code | Output checkpoint | Official | Measured |
|---:|---|---|---|---:|---:|
| 0 | Base model | — | `mii-llm/zagreus-0.4B-ita` | 28.19% | — |
| 1 | Scaled soft-KD rsLoRA | `stages/01_scaled_kd/` | `a6111331…f5504` | 36.00% | 2.946 h |
| 2 | P2F consolidation | `stages/02_p2f/` | `9ddff35c…88b7a` | 37.92% | 1,211.58 s |
| 3 | ARM-A | `stages/03_arm_a/` | `f9906c62…71ef8` | 39.04% | 1,079.01 s |
| 4 | ARM-F | `stages/04_arm_f/` | `209c861e…97b6` | 41.76% | 2,094.64 s |
| 5 | Terminal run | `stages/05_final_push/` | `0c0a42e6…7a991` | **45.73%** | 3,257.39 s |

Every stage is verifiable independently: take the input checkpoint, run the
published script with its config, and compare the output hash.

Stage 5's trainer hash is `989f2e71ae16f4f5…`, matching the
`trainer_sha256` recorded in the campaign's retention amendment.

## Total compute, base to released checkpoint

| Component | GPU-hours |
|---|---:|
| Stage 1 training | 2.946 |
| Stage 2 training | 0.337 |
| Stage 3 training | 0.300 |
| Stage 4 training | 0.582 |
| Stage 5 training | 0.905 |
| *Training subtotal* | **5.070** |
| Teacher target generation, stages 3–5 (`gemma-4-E4B-it`, 21,000 rows × 4 views) | 0.985 |
| Corpus construction and decontamination | 0 — CPU only |
| Synthetic question generation | 0 — none |
| **Total (measured)** | **6.055** |

Stages 1 and 2 also used soft-KD and therefore required their own teacher
labeling passes over their training pools. Those passes are **not separately
itemised** in the campaign records and are excluded from the table above; they
are bounded by their campaigns' total GPU wall time and are estimated at under
1.5 GPU-hours combined, placing the full total below **7.6 GPU-hours**.

Compute spent on exploratory runs, failed arms, and abandoned campaigns is not
included. This table covers the lineage of the released artifact.

## Data

| Asset | Status |
|---|---|
| Pinocchio v4 item-level KD pool, 11,359 rows | published — `efederici/pinocchio` rev `83c6a063…` is Apache-2.0 |
| Teacher-labeled T1 payload, 21,000 rows × 4 views | published |
| Deterministic source-derived replay | published |
| Decontamination manifests, all stages | published |
| **Exam-bank rows (Guardia di Finanza)** | **not redistributable — see below** |

### Exam-bank rows

`SOURCE_WHITELIST.json` was frozen on 2026-08-02, before training, and records
`permit_source_row_redistribution: false`. The publisher's terms read
*"Vietata la pubblicazione, la riproduzione e la divulgazione a scopo di
lucro,"* and the whitelist's conditions permit distributing only model weights,
aggregate metrics, hashes, and non-text provenance — explicitly not source
rows, PDFs, or extracted question text.

Those rows are therefore **not** in this repository. To reproduce stages 2–5
end to end:

1. Fetch the sources yourself from the URL prefixes recorded in
   `SOURCE_WHITELIST.json` (Guardia di Finanza `materiale testologico`
   archives).
2. Run `stages/data/zagreus_exam_bank_v1.py` to build, key, and split the bank.
3. Run `stages/data/zagreus_final_instrument.py` to apply the frozen
   decontamination screens.
4. Verify your build against the published row hashes and split manifests. A
   correct rebuild matches byte-for-byte.

The published hashes make an independent rebuild verifiable without the rows
themselves being redistributed.

## What is and is not reproducible

**Byte-reproducible from this repository:**

- Stage 5 training, from the published frozen initialization and targets
- Teacher target generation (`scripts/labeling/`, locked model revision)
- Official evaluation, pinned harness commit `92df420f…`
- Serving and inference

**Reproducible after self-fetching the Guardia di Finanza sources:**

- Stages 2, 3 and 4, and the exam-bank portion of the training pools

**Verifiable but not re-derivable here:**

- Nothing. Every stage's code, config, and output hash is published; the only
  external requirement is fetching sources this project is not permitted to
  redistribute.

## Decontamination

No official ITALIC question, option, answer, or demonstration was used for
training or teacher labeling at any stage. Every training row passed recursive
HTML-entity decoding and normalized exact match, MinHash Jaccard ≥ 0.70,
content-word containment ≥ 0.80 with ≥ 8 content words, and pinned MiniLM
(`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`) cosine ≥ 0.80 on **both** the
question and question-plus-sorted-options surfaces, with a zero-trigger
mechanical tail recheck. The screen implementation is
`stages/data/zagreus_final_instrument.py`.
