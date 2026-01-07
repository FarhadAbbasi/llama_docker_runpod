# RunPod serverless worker image
# Note: choose a CUDA base compatible with the GPU types you deploy on RunPod.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY inference.py handler.py ./

# RunPod expects your container to start the worker (runpod serverless)
CMD ["python3", "-u", "handler.py"]




_________________________________________________________

from __future__ import annotations
from typing import Any, Dict
import json
import runpod


from inference import generate_response

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod serverless handler.
    Expected input:
      { "messages": [{"role":"system|user|assistant","content":"..."}],
        "max_tokens": 512,
        "temperature": 0.7
      }
    Output:
      { "raw_text": "<model_output_text>" }
    """
    inp = job.get("input", {}) or {}
    messages = inp.get("messages") or []
    max_tokens = inp.get("max_tokens", 512)
    temperature = inp.get("temperature", 0.7)

    if not isinstance(messages, list) or not messages:
        return {"raw_text": ""}

    text = generate_response(messages=messages, max_tokens=max_tokens, temperature=temperature)
    return {"raw_text": text}
    
    # return {"raw_text": "Received messages:\n" + json.dumps(messages, indent=2) }

runpod.serverless.start({"handler": handler})


__________________________________________________________________

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
________________________________________________________
runpod==1.7.5
torch==2.4.1
transformers==4.44.2
accelerate==0.34.2
sentencepiece==0.2.0
