
from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

DEFAULT_MODEL = "gemini-2.5-flash"


def _api_url(model: str) -> str:
    return (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:generateContent"
    )


def _extract_text(resp: Dict[str, Any]) -> str:
    candidates = resp.get("candidates", [])
    if not candidates:
        raise ValueError(f"No candidates in Gemini response: {resp}")
    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [p["text"] for p in parts if "text" in p]
    if texts:
        return "\n".join(texts)
    raise ValueError(f"Could not extract text from Gemini response: {resp}")


def _build_image_part(b64_image: str, mime_type: str = "image/jpeg") -> Dict[str, Any]:
    return {
        "inline_data": {
            "mime_type": mime_type,
            "data": b64_image,
        }
    }


def _raise_with_body(r: requests.Response) -> None:
    try:
        body = r.json()
    except Exception:
        body = r.text[:1000]
    raise RuntimeError(
        f"Gemini API {r.status_code}: {json.dumps(body, indent=2, ensure_ascii=False)}"
    )


def call_vlm(
    *,
    system_prompt: str,
    user_text: str,
    images_b64: List[Tuple[str, str]],
    schema: Dict[str, Any],
    api_key: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 180,
    max_retries: int = 3,
) -> Dict[str, Any]:
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }

    parts: List[Dict[str, Any]] = []
    for label, b64 in images_b64:
        parts.append({"text": f"[{label}]"})
        parts.append(_build_image_part(b64))
    parts.append({"text": user_text})

    body: Dict[str, Any] = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }

    url = _api_url(model)
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                url, headers=headers, json=body, timeout=timeout,
            )
            if r.status_code != 200:
                _raise_with_body(r)
            resp = r.json()
            text = _extract_text(resp)
            return json.loads(text)
        except Exception as e:
            last_err = e
            print(f"  [gemini attempt {attempt+1}/{max_retries}] {e}", file=sys.stderr)
            if attempt + 1 < max_retries:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Gemini request failed after {max_retries} attempts: {last_err}"
    )


def call_llm(
    *,
    system_prompt: str,
    user_text: str,
    schema: Dict[str, Any],
    api_key: str,
    model: str = DEFAULT_MODEL,
    images_b64: Optional[List[Tuple[str, str]]] = None,
    timeout: int = 120,
    max_retries: int = 3,
) -> Dict[str, Any]:
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }

    parts: List[Dict[str, Any]] = []
    if images_b64:
        for label, b64 in images_b64:
            parts.append({"text": f"[{label}]"})
            parts.append(_build_image_part(b64))
    parts.append({"text": user_text})

    body: Dict[str, Any] = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }

    url = _api_url(model)
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                url, headers=headers, json=body, timeout=timeout,
            )
            if r.status_code != 200:
                _raise_with_body(r)
            resp = r.json()
            text = _extract_text(resp)
            return json.loads(text)
        except Exception as e:
            last_err = e
            print(f"  [gemini attempt {attempt+1}/{max_retries}] {e}", file=sys.stderr)
            if attempt + 1 < max_retries:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Gemini rubric request failed after {max_retries} attempts: {last_err}"
    )
