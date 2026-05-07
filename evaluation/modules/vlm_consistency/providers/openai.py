
from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-4.1"


def _extract_output_text(resp: Dict[str, Any]) -> str:
    ot = resp.get("output_text")
    if isinstance(ot, str) and ot.strip():
        return ot
    out_items = resp.get("output", [])
    texts: List[str] = []
    for item in out_items:
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") in ("output_text", "text") and c.get("text"):
                    texts.append(c["text"])
    if texts:
        return "\n".join(texts)
    raise ValueError(f"Could not extract text from OpenAI response: {resp}")


def _build_image_content(b64_image: str, media_type: str = "image/jpeg") -> Dict[str, Any]:
    return {
        "type": "input_image",
        "image_url": f"data:{media_type};base64,{b64_image}",
    }


def _raise_with_body(r: requests.Response) -> None:
    try:
        body = r.json()
    except Exception:
        body = r.text[:1000]
    raise RuntimeError(
        f"OpenAI API {r.status_code}: {json.dumps(body, indent=2, ensure_ascii=False)}"
    )


def call_vlm(
    *,
    system_prompt: str,
    user_text: str,
    images_b64: List[Tuple[str, str]],
    schema: Dict[str, Any],
    schema_name: str = "scene_alignment_output",
    api_key: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 180,
    max_retries: int = 3,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    user_content: List[Dict[str, Any]] = []
    for label, b64 in images_b64:
        user_content.append({"type": "input_text", "text": f"[{label}]"})
        user_content.append(_build_image_content(b64))
    user_content.append({"type": "input_text", "text": user_text})

    body: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    }

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                RESPONSES_URL, headers=headers, json=body, timeout=timeout,
            )
            if r.status_code != 200:
                _raise_with_body(r)
            resp = r.json()
            text = _extract_output_text(resp)
            return json.loads(text)
        except Exception as e:
            last_err = e
            print(f"  [openai attempt {attempt+1}/{max_retries}] {e}", file=sys.stderr)
            if attempt + 1 < max_retries:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"OpenAI request failed after {max_retries} attempts: {last_err}"
    )


def call_llm(
    *,
    system_prompt: str,
    user_text: str,
    schema: Dict[str, Any],
    schema_name: str = "rubric_output",
    api_key: str,
    model: str = DEFAULT_MODEL,
    images_b64: Optional[List[Tuple[str, str]]] = None,
    timeout: int = 120,
    max_retries: int = 3,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if images_b64:
        user_content: Any = []
        for label, b64 in images_b64:
            user_content.append({"type": "input_text", "text": f"[{label}]"})
            user_content.append(_build_image_content(b64))
        user_content.append({"type": "input_text", "text": user_text})
    else:
        user_content = user_text

    body: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    }

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                RESPONSES_URL, headers=headers, json=body, timeout=timeout,
            )
            if r.status_code != 200:
                _raise_with_body(r)
            resp = r.json()
            text = _extract_output_text(resp)
            return json.loads(text)
        except Exception as e:
            last_err = e
            print(f"  [openai attempt {attempt+1}/{max_retries}] {e}", file=sys.stderr)
            if attempt + 1 < max_retries:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"OpenAI rubric request failed after {max_retries} attempts: {last_err}"
    )
