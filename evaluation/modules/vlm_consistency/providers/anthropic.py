
from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

MESSAGES_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_VERSION = "2023-06-01"


def _extract_text(resp: Dict[str, Any]) -> str:
    parts = resp.get("content", [])
    for p in parts:
        if p.get("type") == "text" and isinstance(p.get("text"), str):
            return p["text"]
    raise ValueError(f"Could not extract text from Anthropic response: {resp}")


def _build_image_block(b64_image: str, media_type: str = "image/jpeg") -> Dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": b64_image,
        },
    }


def _raise_with_body(r: requests.Response) -> None:
    try:
        body = r.json()
    except Exception:
        body = r.text[:1000]
    raise RuntimeError(
        f"Anthropic API {r.status_code}: {json.dumps(body, indent=2, ensure_ascii=False)}"
    )


def call_vlm(
    *,
    system_prompt: str,
    user_text: str,
    images_b64: List[Tuple[str, str]],
    schema: Dict[str, Any],
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
    timeout: int = 180,
    max_retries: int = 3,
) -> Dict[str, Any]:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    user_content: List[Dict[str, Any]] = []
    for label, b64 in images_b64:
        user_content.append({"type": "text", "text": f"[{label}]"})
        user_content.append(_build_image_block(b64))
    user_content.append({"type": "text", "text": user_text})

    body: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_content}],
    }

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                MESSAGES_URL, headers=headers, json=body, timeout=timeout,
            )
            if r.status_code != 200:
                _raise_with_body(r)
            resp = r.json()
            text = _extract_text(resp)
            return json.loads(text)
        except Exception as e:
            last_err = e
            print(f"  [anthropic attempt {attempt+1}/{max_retries}] {e}", file=sys.stderr)
            if attempt + 1 < max_retries:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Anthropic request failed after {max_retries} attempts: {last_err}"
    )


def call_llm(
    *,
    system_prompt: str,
    user_text: str,
    schema: Dict[str, Any],
    api_key: str,
    model: str = DEFAULT_MODEL,
    images_b64: Optional[List[Tuple[str, str]]] = None,
    max_tokens: int = 4096,
    timeout: int = 120,
    max_retries: int = 3,
) -> Dict[str, Any]:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    if images_b64:
        user_content: Any = []
        for label, b64 in images_b64:
            user_content.append({"type": "text", "text": f"[{label}]"})
            user_content.append(_build_image_block(b64))
        user_content.append({"type": "text", "text": user_text})
    else:
        user_content = user_text

    body: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_content}],
    }

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                MESSAGES_URL, headers=headers, json=body, timeout=timeout,
            )
            if r.status_code != 200:
                _raise_with_body(r)
            resp = r.json()
            text = _extract_text(resp)
            return json.loads(text)
        except Exception as e:
            last_err = e
            print(f"  [anthropic attempt {attempt+1}/{max_retries}] {e}", file=sys.stderr)
            if attempt + 1 < max_retries:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Anthropic rubric request failed after {max_retries} attempts: {last_err}"
    )
