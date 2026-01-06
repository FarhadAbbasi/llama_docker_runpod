from __future__ import annotations

import os
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN env var is required on the GPU worker to download gated model weights.")

# Load once per container cold start
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    token=HF_TOKEN,
)

model.eval()

@torch.inference_mode()
def generate_response(messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.7) -> str:
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    encoded = tokenizer(
        chat_prompt,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(model.device)

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
