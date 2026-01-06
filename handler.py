from __future__ import annotations

from typing import Any, Dict

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

runpod.serverless.start({"handler": handler})
