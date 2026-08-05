#!/usr/bin/env python3
"""Synthetic five-shot inference example for mcsp/zagreus-italic-45.73."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "mcsp/zagreus-italic-45.73"


def question(text: str, options: list[str]) -> str:
    letters = "ABCD"[: len(options)]
    rendered = "\n".join(f"{letter}. {option}" for letter, option in zip(letters, options))
    return f"{text}\n{rendered}\nRisposta:"


messages = [{"role": "system", "content": "Rispondi soltanto con la lettera corretta."}]
demonstrations = [
    (question("Quanto fa 1+1?", ["1", "2", "3", "4"]), "B"),
    (question("Quanto fa 2+1?", ["3", "4", "5", "6"]), "A"),
    (question("Quale parola è un colore?", ["correre", "blu", "presto", "tavolo"]), "B"),
    (question("Quanto fa 2×2?", ["2", "3", "4", "5"]), "C"),
    (question("Quale animale miagola?", ["gatto", "cane", "cavallo", "pesce"]), "A"),
]
for prompt, answer in demonstrations:
    messages.extend(({"role": "user", "content": prompt}, {"role": "assistant", "content": answer}))
messages.append({"role": "user", "content": question("Quanto fa 3+3?", ["5", "6", "7", "8"])})

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt",
).to(model.device)
output = model.generate(inputs, max_new_tokens=2, do_sample=False)
answer = tokenizer.decode(output[0, inputs.shape[1] :], skip_special_tokens=True).strip()
print(answer)
