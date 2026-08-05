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

Zagreus ITALIC 45.73 is a 437.8M-parameter causal language model specialized
for Italian multiple-choice reasoning. It scored **45.73%** on the complete
10,000-question official ITALIC evaluation.

- [Training code and report](https://github.com/LudWittg/zagreus-italic-45.73)
- [Exact training bundle](https://huggingface.co/datasets/mcsp/zagreus-italic-45.73-reproduction)

## Evaluation

| Metric | Value |
|---|---:|
| Official accuracy | **45.73%** |
| Evaluated rows | 10,000 |
| Valid answer rate | 99.97% |
| Exact one-character rate | 99.95% |
| Mean output length | 1.0 character |

The official run used deterministic decoding and the unmodified ITALIC harness
at commit `92df420ff686babeea54e217b9f90f8471374916`. The tracked harness
tree was unchanged after evaluation. No official ITALIC question, option,
answer, or demonstration was used for training or teacher labeling.

Weight SHA-256:

```text
0c0a42e642e009d6aff7dd6663707ee0f877218b826ffb1d4ab3abd46687a991
```

## Training

Each example is rendered online as a 12-message conversation containing a
system instruction, five training-pool demonstrations, and one target. The
renderer resamples demonstrations, option count, option order, displayed answer
letter, and teacher view throughout training. All six assistant letters are
supervised, followed by target EOS.

The data mixture contains 21,000 signal rows with four-view
`google/gemma-4-E4B-it` posteriors and 3,000 anchor rows resampled to 20% of
epoch exposures. Of the signal rows, 9,713 also have authorized hard gold.

For signal rows:

```text
L_target = 0.5 L_gold-CE + 0.5 L_E4B-KD(T=2)   when authorized gold exists
L_target = L_E4B-KD(T=2)                        otherwise
L_signal = (5 L_demo-CE + L_target) / 6 + 0.1 L_EOS
```

For anchors:

```text
L_anchor = 2.0 × mean_turn KL(P_init || P_student), T=1
```

Option permutations are inverted before target CE/KD, so supervision remains
in semantic-option space.

| Setting | Value |
|---|---:|
| Seed | `20260831` |
| Epochs | 4 |
| Optimizer steps | 6,564 |
| Effective batch | 16 |
| Peak LR | `1.5e-4` |
| Schedule | 5% warmup, cosine to `1.5e-5` |
| Optimizer | Fused AdamW, weight decay `0.05` |
| Precision | BF16 autocast, FP32 master weights |
| Hardware | 1 × NVIDIA H100 80GB |
| Tokens | 113,309,980 |
| Throughput | 34,785 tokens/s |
| **Training time** | **3,257 seconds — 54 minutes 17 seconds** |

Only the seven supervised hidden states are projected into the 128,256-token
vocabulary. Dynamic padding, length bucketing, disabled gradient checkpointing,
and token-budgeted microbatches provide the remaining speedup.

The complete reproducibility package contains the exact data, teacher targets,
frozen initialization checkpoint, dependency versions, and terminal training
command. The reported 54 minutes is measured wall time for the full four-epoch
training job; model download and official evaluation are not included.

## Use

The model expects a system message, five `(user, assistant)` demonstrations,
and a target user message. It should complete one answer letter.

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

messages = [...]  # system + five demonstrations + target
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

## Limitations and terms

- This is a narrow multiple-choice model, not a general Italian assistant.
- The score is specific to the official five-shot interface.
- Rare invalid answers remain possible.
- Distillation can preserve teacher errors and calibration bias.
- The reproduction bundle contains upstream material with non-commercial and
  attribution/share-alike restrictions; consult its dataset card before use.

The Hub inventory reports 560.9M tensor parameters because tied embeddings are
stored twice; the model has 437.8M unique trainable parameters. The model is
distributed for research and evaluation under the terms of its upstream model
and data sources.
