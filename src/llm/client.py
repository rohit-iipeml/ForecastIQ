from __future__ import annotations

import os
import time
from typing import Optional

from dotenv import load_dotenv
from groq import Groq


def llm_generate_text(
    prompt: str,
    system: Optional[str] = None,
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> str:
    """Generate text via Groq chat completions with simple retry on rate limits."""
    load_dotenv()
    key = os.getenv("GROQ_API_KEY", "api_key")
    if not key or key.strip() in {"", "api_key"}:
        raise RuntimeError("GROQ_API_KEY missing or still placeholder. Update .env with your real key.")

    client = Groq(api_key=key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_exc = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("Groq returned empty response text.")
            return text
        except Exception as exc:
            last_exc = exc
            status = getattr(exc, "status_code", None)
            msg = str(exc).lower()
            is_rate = status == 429 or "rate limit" in msg or "too many requests" in msg
            if is_rate and attempt < 2:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise RuntimeError(f"Groq text generation failed: {exc}") from exc

    raise RuntimeError(f"Groq text generation failed after retries: {last_exc}")
