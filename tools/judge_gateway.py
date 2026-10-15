#!/usr/bin/env python3
"""Run ``tools/run_wmeval_lite.py`` (or another judge-calling tool, ``--script``)
with hosted-judge calls routed through an OpenAI-compatible gateway (default: NVIDIA inference, which serves the same
``gpt-5.4`` judge as ``azure/openai/gpt-5.4``).

Nothing in the evaluator changes: judge model names and prompts stay as pinned in
``suite.yaml``; only the transport (base URL, credential, provider model prefix)
differs. Results produced this way should record the gateway in their notes.

Usage::

    python tools/judge_gateway.py --key-file ~/.ssh/vlm_as_judge_key -- \\
        run --suite eval_sets/wmeval_lite_v1 --model-root runs/odyssey \\
        --dimensions prompt_scene_consistency,lpips --output-dir reports/wmeval_lite/odyssey

    python tools/judge_gateway.py --key-file ~/.credentials/nv_vlm_api_key \\
        --script tools/freeze_reference_events.py -- LiquidContainer/fluid_rest --passes 5 --register

The gateway URL is exported as ``WMEVAL_JUDGE_TRANSPORT`` so tools that freeze
judge outputs can record it next to the judge model.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def install(base_url: str, prefix: str, key: str) -> None:
    os.environ["OPENAI_API_KEY"] = key
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["WMEVAL_JUDGE_TRANSPORT"] = f"{base_url} (model prefix {prefix})"

    def route(model):
        model = str(model)
        return model if model.startswith(prefix) else prefix + model

    # openai SDK (causal LPIPS, CausalStep): base URL comes from OPENAI_BASE_URL;
    # prefix the model on every create() call.
    from openai.resources.chat import completions as chat_completions
    from openai.resources import responses as responses_module

    original_chat = chat_completions.Completions.create
    original_responses = responses_module.Responses.create

    def chat_create(self, *args, **kwargs):
        kwargs["model"] = route(kwargs.get("model"))
        return original_chat(self, *args, **kwargs)

    def responses_create(self, *args, **kwargs):
        kwargs["model"] = route(kwargs.get("model"))
        return original_responses(self, *args, **kwargs)

    chat_completions.Completions.create = chat_create
    responses_module.Responses.create = responses_create

    # Prompt-consistency provider posts to the Responses endpoint with requests.
    sys.path.insert(0, str(ROOT))
    import requests
    from evaluation.modules.vlm_consistency.providers import openai as provider

    provider.RESPONSES_URL = base_url.rstrip("/") + "/responses"
    original_post = requests.post

    def post(url, *args, **kwargs):
        if str(url) == provider.RESPONSES_URL and isinstance(kwargs.get("json"), dict):
            kwargs["json"] = {**kwargs["json"], "model": route(kwargs["json"].get("model"))}
            headers = dict(kwargs.get("headers") or {})
            headers["Authorization"] = f"Bearer {key}"
            kwargs["headers"] = headers
        return original_post(url, *args, **kwargs)

    provider.requests.post = post


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--base-url", default="https://inference-api.nvidia.com/v1")
    parser.add_argument("--model-prefix", default="azure/openai/")
    parser.add_argument("--script", type=Path, default=ROOT / "tools/run_wmeval_lite.py")
    parser.add_argument("lite_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    lite_args = args.lite_args[1:] if args.lite_args[:1] == ["--"] else args.lite_args
    install(args.base_url, args.model_prefix, args.key_file.expanduser().read_text().strip())
    sys.argv = [str(args.script.resolve()), *lite_args]
    runpy.run_path(sys.argv[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
