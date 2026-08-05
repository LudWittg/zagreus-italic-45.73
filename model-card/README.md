---
language:
- it
library_name: transformers
pipeline_tag: text-generation
license: other
tags:
- multiple-choice
- italian
- question-answering
- llama
- italic-benchmark
---

# Zagreus ITALIC 45.73

Zagreus is a compact Llama-architecture causal model specialized for Italian
multiple-choice reasoning. It has 437.8M unique trainable parameters and scored
**45.73%** on the 10,000-question official ITALIC evaluation.

Training code, aggregate evidence, and a complete synthetic inference example
are available in the
[GitHub repository](https://github.com/LudWittg/zagreus-italic-45.73).

## Evaluation

| Metric | Value |
|---|---:|
| Official accuracy | **45.73%** |
| Evaluated rows | 10,000 |
| Valid answer rate | 99.97% |
| Exact one-character rate | 99.95% |
| Mean output length | 1.0 character |

The official run used deterministic decoding and pinned ITALIC commit
`92df420ff686babeea54e217b9f90f8471374916`. The tracked harness tree was
unchanged after evaluation. No ITALIC example was used for training, teacher
labeling, demonstrations, or checkpoint selection.

Weight SHA-256:

```text
0c0a42e642e009d6aff7dd6663707ee0f877218b826ffb1d4ab3abd46687a991
```

## What was trained

The official benchmark presents a system instruction, five demonstrated
question/answer turns, and a target question. Earlier project checkpoints were
trained on isolated four-option questions. The winning lineage closed that
interface gap in two stages:

| Stage | Training change | Official accuracy |
|---|---|---:|
| ARM-A | Earlier single-question recipe | 39.04% |
| ARM-F | Format-matched five-shot conversations | 41.76% |
| This checkpoint | Online demos/permutations and four full epochs | **45.73%** |

The final model starts from ARM-F and full-fine-tunes all parameters; it is not
a LoRA adapter.

## Training data streams

Each epoch mixes:

- **21,000 signal rows (80%)** with four-view probability targets from the
  retained `google/gemma-4-E4B-it` control teacher. A vetted subset of 9,713
  rows also has an authorized hard gold label.
- **3,000 unique anchor rows**, resampled to **20% of epoch exposures**, used to
  preserve the behavior of the ARM-F initialization through posterior matching.

The preceding campaign also tested a stronger three-teacher ensemble. That
ensemble scored well as a teacher, but its student arm did not beat the E4B
control. This checkpoint therefore uses the simpler E4B control distributions.

## Online format augmentation

Every row exposure is rendered again from deterministic `(seed, epoch, serial,
row-id)` randomness:

1. sample one of four teacher views;
2. sample a 3-, 4-, or 5-option layout from the benchmark's public aggregate
   profile;
3. safely drop a distractor for three-option items or add a non-duplicate donor
   distractor for five-option items;
4. permute semantic options into display slots and balance answer letters;
5. sample five gold-authorized demonstrations from non-overlapping training
   groups;
6. render the system instruction, five `(user, assistant)` demonstrations, and
   target as the same 12-message structure used at evaluation.

Repeated epochs therefore do not replay identical sequences. Demonstrations,
option counts, option order, and target view continue to change while the
underlying question pool stays fixed.

## Supervision and loss

All five demonstration letters and the target letter are supervised, providing
six learning positions per conversation.

For signal rows:

```text
L_target = 0.5 L_gold-CE + 0.5 L_E4B-KD(T=2)   if vetted gold exists
L_target = L_E4B-KD(T=2)                        otherwise

L_signal = (5 L_demo-CE + L_target) / 6 + 0.1 L_EOS
```

`T²` scaling is applied to the temperature-2 KD term. Target logits are mapped
back from displayed letters into semantic-option order before CE/KD, so option
permutation does not corrupt supervision.

For anchor rows, a frozen ARM-F reference supplies posteriors for all six turns:

```text
L_anchor = 2.0 × mean_turn KL(P_ARM-F || P_student),  T=1
```

The EOS term teaches the target completion to stop immediately after its answer
letter.

## Optimization

| Setting | Value |
|---|---:|
| Seed | `20260831` |
| Epochs | 4 |
| Optimizer steps | 6,564 |
| Effective batch | 16 |
| Peak LR | `1.5e-4` |
| Warmup | 5% linear |
| Decay | Cosine to `1.5e-5` across the full run |
| Optimizer | Fused AdamW, weight decay `0.05` |
| Gradient clipping | `1.0` |
| Precision | BF16 autocast, FP32 master weights |
| Hardware | 1 × H100 80GB |
| Tokens | 113.3M |
| Throughput | 34,785 tokens/s |

The transformer processes the complete prompt, but the 128,256-way vocabulary
projection is applied only after gathering the seven required hidden states:
five demo letters, target letter, and target EOS. Dynamic padding,
length-bucketed batches, disabled gradient checkpointing, and token-budgeted
microbatches provide the remaining speedup without changing effective-batch
weights.

## Intended use

The model expects the benchmark-aligned five-shot structure and should complete
one answer letter. The included chat template joins message contents with blank
lines, matching the evaluated interface.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mcsp/zagreus-italic-45.73"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# system + 5 (user, assistant) demonstrations + target user message
messages = [...]  # See the linked GitHub example for a complete synthetic prompt.
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt",
).to(model.device)
output = model.generate(inputs, max_new_tokens=2, do_sample=False)
print(tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True).strip())
```

Frozen serving versions: `transformers==4.55.2`, `vllm==0.10.2`.

## Limitations

- This is a narrow multiple-choice model, not a general Italian assistant.
- The reported score is specific to the official five-shot interface; zero-shot
  or differently templated performance may be materially lower.
- Invalid answers remain possible on rare inputs.
- The private training rows and teacher posterior package are not distributed,
  so the public release supports method inspection and artifact verification
  rather than a fully data-independent rerun.
- Distillation can preserve teacher mistakes and calibration bias.

## Size, provenance, and terms

The Hub safetensors inventory reports 560.9M tensor parameters because tied
input/output embeddings are stored twice. The model has 437.8M unique trainable
parameters.

The public weight was copied from immutable private source commit
`f3582df4c53417115cd8dc8bb074f1628c233a1e`, path
`candidates/final_push_track_c_seed_20260831_step6564`. Publication changed only
compatibility metadata and documentation; the weight hash is unchanged.

The model is distributed for research and evaluation. Downstream users are
responsible for confirming that their use complies with all applicable
upstream model and data terms.
