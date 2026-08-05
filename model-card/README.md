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

Zagreus is a 437.8M-parameter Llama-architecture causal model specialized for
Italian multiple-choice questions. It scored **45.73%** on the 10,000-row
official ITALIC evaluation using the benchmark's five-shot conversational
prompt and deterministic decoding.

## Evaluation

| Metric | Value |
|---|---:|
| Official accuracy | **45.73%** |
| Rows | 10,000 |
| Valid answer rate | 99.97% |
| Exact one-character rate | 99.95% |
| Mean output length | 1.0 character |

The evaluation used pinned ITALIC commit
`92df420ff686babeea54e217b9f90f8471374916`. The tracked harness tree was
unchanged after the run. ITALIC examples were not used during training.

Weight SHA-256:

```text
0c0a42e642e009d6aff7dd6663707ee0f877218b826ffb1d4ab3abd46687a991
```

Aggregate evidence and training code are available in the
[GitHub repository](https://github.com/LudWittg/zagreus-italic-45.73).

## Intended use

Use this checkpoint for Italian multiple-choice inference with a system
instruction, five `(user, assistant)` demonstrations, and one target user turn.
The assistant completions are answer letters. The included plain chat template
concatenates message contents with blank lines, matching the evaluated serving
interface.

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

messages = [
    {"role": "system", "content": "Rispondi soltanto con la lettera corretta."},
    {"role": "user", "content": "Quanto fa 1+1?\nA. 1\nB. 2\nC. 3\nD. 4\nRisposta:"},
    {"role": "assistant", "content": "B"},
    {"role": "user", "content": "Quanto fa 2+1?\nA. 3\nB. 4\nC. 5\nD. 6\nRisposta:"},
    {"role": "assistant", "content": "A"},
    {"role": "user", "content": "Quale parola è un colore?\nA. correre\nB. blu\nC. presto\nD. tavolo\nRisposta:"},
    {"role": "assistant", "content": "B"},
    {"role": "user", "content": "Quanto fa 2×2?\nA. 2\nB. 3\nC. 4\nD. 5\nRisposta:"},
    {"role": "assistant", "content": "C"},
    {"role": "user", "content": "Quale animale miagola?\nA. gatto\nB. cane\nC. cavallo\nD. pesce\nRisposta:"},
    {"role": "assistant", "content": "A"},
    {"role": "user", "content": "Quanto fa 3+3?\nA. 5\nB. 6\nC. 7\nD. 8\nRisposta:"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt",
).to(model.device)
output = model.generate(inputs, max_new_tokens=2, do_sample=False)
print(tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True).strip())
```

## Training

The final run used four epochs, 6,564 optimizer steps, online five-demo
sampling, online option permutation, supervision on all six assistant turns,
20% anchors, and effective batch size 16. The peak learning rate was `1.5e-4`
with 5% warmup and cosine decay to `1.5e-5`. Training used BF16 autocast with
FP32 master weights on one H100 80GB.

## Limitations and license

This is a narrow benchmark-oriented model rather than a general conversational
assistant. Results outside the official prompt protocol may differ materially.
Rare invalid outputs remain possible. The model is distributed for research and
evaluation; downstream users are responsible for confirming that their use is
compatible with all applicable upstream data and model terms.

