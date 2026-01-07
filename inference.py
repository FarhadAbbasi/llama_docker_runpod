# vllm_inference.py
from __future__ import annotations

import os
from typing import Dict, List

from vllm import LLM, SamplingParams

# ----------------------------
# Environment / low-memory defaults
# ----------------------------
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")

# IMPORTANT: for gated models, set HUGGING_FACE_HUB_TOKEN (or HF_TOKEN and we map it)
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
if HF_TOKEN and not os.getenv("HUGGING_FACE_HUB_TOKEN"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN env var is required on the GPU worker to download gated model weights.")

# Disk/cache: avoid /root/.cache (your error showed /root had ~363MB free)
# Put HF cache + vLLM downloads in /workspace (usually larger)
HF_HOME = os.getenv("HF_HOME", "/workspace/.cache/huggingface")
VLLM_DOWNLOAD_DIR = os.getenv("VLLM_DOWNLOAD_DIR", "/workspace/.cache/vllm_downloads")
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_HOME, "transformers"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(HF_HOME, "hub"))

# vLLM memory/perf knobs (start conservative)
# Keep these low for first successful test, then increase gradually
GPU_MEMORY_UTILIZATION = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.80"))
MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "2048"))  # lower = less KV cache
MAX_NUM_SEQS = int(os.getenv("VLLM_MAX_NUM_SEQS", "1"))       # low concurrency for initial stability
ENFORCE_EAGER = os.getenv("VLLM_ENFORCE_EAGER", "true").lower() == "true"

# ----------------------------
# Model init (once per worker cold start)
# ----------------------------
# NOTE: model weights are still large; ensure the serverless worker has enough disk in /workspace.
# If /workspace is also small, you MUST choose a serverless template with more ephemeral disk or attach storage.
llm = LLM(
    model=MODEL_ID,
    dtype="half",  # fp16
    download_dir=VLLM_DOWNLOAD_DIR,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    max_model_len=MAX_MODEL_LEN,
    max_num_seqs=MAX_NUM_SEQS,
    enforce_eager=ENFORCE_EAGER,
    tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
    trust_remote_code=False,
)

# ----------------------------
# Chat formatting
# ----------------------------
def _simple_chat_template(messages: List[Dict[str, str]]) -> str:
    """
    Minimal chat template that works well for Llama-Instruct style models.
    This avoids needing transformers' tokenizer.apply_chat_template.
    """
    system_parts = []
    convo_parts = []

    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue

        if role == "system":
            system_parts.append(content)
        elif role == "user":
            convo_parts.append(f"User: {content}")
        elif role == "assistant":
            convo_parts.append(f"Assistant: {content}")
        else:
            # unknown role treated as user
            convo_parts.append(f"User: {content}")

    system_block = ""
    if system_parts:
        system_block = "System: " + "\n".join(system_parts) + "\n\n"

    # End with Assistant: cue for generation
    prompt = system_block + "\n".join(convo_parts) + "\nAssistant:"
    return prompt

def generate_response_vllm(
    messages: List[Dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    prompt = _simple_chat_template(messages)

    # vLLM sampling params
    params = SamplingParams(
        temperature=max(0.0, float(temperature)),
        max_tokens=int(max_tokens),
        top_p=float(os.getenv("VLLM_TOP_P", "0.9")),
        # keep it safe for first test
        stop=["\nUser:"],
    )

    outputs = llm.generate([prompt], sampling_params=params)
    if not outputs:
        return ""

    # vLLM returns list[RequestOutput]; take first candidate
    out0 = outputs[0]
    if not out0.outputs:
        return ""
    return (out0.outputs[0].text or "").strip()



# ___________________________________________________________________
# from __future__ import annotations

# import os
# from typing import Any, Dict, List

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
# HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# if not HF_TOKEN:
#     raise RuntimeError("HF_TOKEN env var is required on the GPU worker to download gated model weights.")

# # Load once per container cold start
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     device_map="auto",
#     torch_dtype=torch.float16,
#     token=HF_TOKEN,
# )

# model.eval()

# @torch.inference_mode()
# def generate_response(messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.7) -> str:
#     chat_prompt = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#     )

#     encoded = tokenizer(
#         chat_prompt,
#         return_tensors="pt",
#         return_attention_mask=True,
#     ).to(model.device)

#     input_ids = encoded["input_ids"]
#     attention_mask = encoded["attention_mask"]

#     out = model.generate(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         max_new_tokens=int(max_tokens),
#         temperature=float(temperature),
#         do_sample=True,
#         pad_token_id=tokenizer.eos_token_id,
#     )

#     return tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
