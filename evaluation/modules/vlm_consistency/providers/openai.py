
from __future__ import annotations

import json
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-4.1"


def _log_usage(provider: str, model: str, usage: Any, *, api: str = "") -> None:
    """Token accounting hook (evaluation/vlm_usage.py); never fatal."""
    try:
        from evaluation.vlm_usage import log_usage
    except ImportError:  # module run outside the repo
        return
    try:
        log_usage(provider, model, usage, api=api)
    except Exception:  # noqa: BLE001
        pass


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


class _OpenAIHTTPError(RuntimeError):
    """Non-200 response; carries the status and the server's retry hint."""

    def __init__(self, message: str, status_code: int, retry_after: Optional[float]) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


def _retry_after_seconds(r: requests.Response) -> Optional[float]:
    """Server-suggested wait: the Retry-After header, else the 'try again in
    Xs' phrase OpenAI puts in 429 bodies."""
    header = r.headers.get("Retry-After") if r.headers else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    match = re.search(r"try again in ([0-9.]+)\s*(ms|s)", r.text or "")
    if match:
        value = float(match.group(1))
        return value / 1000.0 if match.group(2) == "ms" else value
    return None


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Exponential backoff (2, 4, 8, ... s, capped at 60) with jitter; a
    429/5xx that names a wait uses at least that wait. Rate limits are per
    minute, so a few seconds of patience beats failing the row."""
    base = min(60.0, 2.0 ** (attempt + 1))
    hint = getattr(exc, "retry_after", None)
    if isinstance(hint, (int, float)) and hint > 0:
        base = max(base, float(hint) + 1.0)
    return base + random.uniform(0.0, 1.5)


def _raise_with_body(r: requests.Response) -> None:
    try:
        body = r.json()
    except Exception:
        body = r.text[:1000]
    raise _OpenAIHTTPError(
        f"OpenAI API {r.status_code}: {json.dumps(body, indent=2, ensure_ascii=False)}",
        status_code=r.status_code,
        retry_after=_retry_after_seconds(r),
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
    max_retries: int = 6,
    record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Judge call; when ``record`` is given it is filled with what the API
    returned besides the answer (resolved model snapshot, response id, usage,
    attempts, raw output text)."""
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
            _log_usage("openai", model, resp.get("usage"), api="responses")
            text = _extract_output_text(resp)
            parsed = json.loads(text)
            if record is not None:
                record.update(
                    {
                        "provider": "openai",
                        "model_requested": model,
                        "model_resolved": resp.get("model"),
                        "response_id": resp.get("id"),
                        "created_at": resp.get("created_at"),
                        "usage": resp.get("usage"),
                        "attempts": attempt + 1,
                        "output_text": text,
                    }
                )
            return parsed
        except Exception as e:
            last_err = e
            print(f"  [openai attempt {attempt+1}/{max_retries}] {e}", file=sys.stderr)
            if attempt + 1 < max_retries:
                time.sleep(_retry_delay(e, attempt))

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
    max_retries: int = 6,
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
            _log_usage("openai", model, resp.get("usage"), api="responses")
            text = _extract_output_text(resp)
            return json.loads(text)
        except Exception as e:
            last_err = e
            print(f"  [openai attempt {attempt+1}/{max_retries}] {e}", file=sys.stderr)
            if attempt + 1 < max_retries:
                time.sleep(_retry_delay(e, attempt))

    raise RuntimeError(
        f"OpenAI rubric request failed after {max_retries} attempts: {last_err}"
    )
