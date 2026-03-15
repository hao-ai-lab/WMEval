
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import torch

from ..context import EvaluationContext
from ..video_io import build_evaluation_context

try:
    from PIL import Image
except ImportError:
    Image = None

_IMPORT_ERROR_MSG = None
CLIPTokenizer = CLIPTextModelWithProjection = CLIPVisionModelWithProjection = None
Compose = Resize = CenterCrop = ToTensor = Normalize = InterpolationMode = None

try:
    from torchvision.transforms import (
        CenterCrop,
        Compose,
        InterpolationMode,
        Normalize,
        Resize,
        ToTensor,
    )
except ImportError as e:
    _IMPORT_ERROR_MSG = f"torchvision: {e}. pip install torchvision"
    Compose = Resize = CenterCrop = ToTensor = Normalize = InterpolationMode = None

try:
    from transformers import (
        CLIPTextModelWithProjection,
        CLIPTokenizer,
        CLIPVisionModelWithProjection,
    )
except ImportError as e:
    _IMPORT_ERROR_MSG = (_IMPORT_ERROR_MSG or "") + f" transformers: {e}. pip install transformers"
    CLIPTokenizer = CLIPTextModelWithProjection = CLIPVisionModelWithProjection = None

DEFAULT_MODEL_ID = "Searchium-ai/clip4clip-webvid150k"
DEFAULT_MAX_FRAMES = 32
CLIP_SIZE = 224


def ensure_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def build_clip4clip_preprocess(size: int = CLIP_SIZE):
    if Compose is None or Resize is None:
        raise RuntimeError("torchvision.transforms not available. pip install torchvision")
    return Compose(
        [
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size),
            lambda img: img.convert("RGB"),
            ToTensor(),
            Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def run_clip4clip(
    context: EvaluationContext,
    device: Optional[str] = None,
    model_id: str = DEFAULT_MODEL_ID,
) -> Dict[str, Any]:

    dev = ensure_device(device)

    base = {
        "module": "clip4clip",
        "reference_video": context.reference_video,
        "candidate_video": context.candidate_video,
        "reference_metadata": context.reference.metadata.to_dict(),
        "candidate_metadata": context.candidate.metadata.to_dict(),
        "shared_duration_sec": context.shared_duration_sec,
        "eval_fps": context.eval_fps,
        "max_pairs": context.max_pairs,
        "alignment_mode": context.alignment_mode,
        "device": str(dev),
        "text_prompt": context.text_prompt or "",
    }

    if not context.candidate.frames:
        return {**base, "error": "no candidate frames available"}

    if CLIPTokenizer is None or CLIPTextModelWithProjection is None or Compose is None:
        return {**base, "error": _IMPORT_ERROR_MSG or "pip install transformers torchvision"}
    
    if Image is None:
        return {**base, "error": "PIL (Pillow) not installed. pip install Pillow"}

    frames = list(context.candidate.frames)[:DEFAULT_MAX_FRAMES]
    text_prompt = context.text_prompt or ""

    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    text_model = CLIPTextModelWithProjection.from_pretrained(model_id).to(dev).eval()
    vision_model = CLIPVisionModelWithProjection.from_pretrained(model_id).to(dev).eval()
    preprocess = build_clip4clip_preprocess(size=CLIP_SIZE)

    text_inputs = tokenizer(text=text_prompt, return_tensors="pt", truncation=True)
   
    with torch.no_grad():
        text_outputs = text_model(
            input_ids=text_inputs["input_ids"].to(dev),
            attention_mask=text_inputs["attention_mask"].to(dev),
        )
        
        if hasattr(text_outputs, "text_embeds"):
            text_emb = text_outputs.text_embeds
        else:
            text_emb = text_outputs.last_hidden_state[:, 0, :]
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    clip_tensors: List[torch.Tensor] = []
    
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (CLIP_SIZE, CLIP_SIZE), interpolation=cv2.INTER_AREA)
        clip_tensors.append(preprocess(Image.fromarray(rgb)))
    video_tensor = torch.stack(clip_tensors, dim=0).to(dev)

    with torch.no_grad():
        vision_outputs = vision_model(pixel_values=video_tensor)
       
        if hasattr(vision_outputs, "image_embeds"):
            img_embs = vision_outputs.image_embeds
        elif hasattr(vision_outputs, "pooler_output"):
            img_embs = vision_outputs.pooler_output
        else:
            img_embs = vision_outputs.last_hidden_state[:, 0, :]
        
        img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
        video_emb = img_embs.mean(dim=0, keepdim=True)
    
    sim = float((text_emb * video_emb).sum().item())

    out = {**base}
    out["num_frames_used"] = len(frames)
    out["similarity_cosine"] = sim
    out["model_id"] = model_id
    out["timestamps_sec"] = context.candidate.timestamps_sec[:len(frames)]
    out["candidate_frame_indices"] = context.candidate.frame_indices[:len(frames)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="CLIP4Clip: video + text -> similarity score.",
    )
    ap.add_argument("video", help="Path to video file")
    ap.add_argument("text", help="Text prompt")
    ap.add_argument("--device", default=None, help="cuda, mps, or cpu")
    ap.add_argument("--max-pairs", type=int, default=32)
    ap.add_argument("--eval-fps", type=float, default=4.0)
    ap.add_argument("--json", action="store_true", help="Output JSON only")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}", file=sys.stderr)
        return 1

    context = build_evaluation_context(
        reference_video=str(video_path),
        candidate_video=str(video_path),
        eval_fps=args.eval_fps,
        max_pairs=args.max_pairs,
        text_prompt=args.text,
    )
    result = run_clip4clip(context=context, device=args.device)
    
    if "error" in result:
        print(result["error"], file=sys.stderr)
        if args.json:
            print(json.dumps(result, indent=2))
        return 1
    
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    
    print(f"Similarity (cosine): {result['similarity_cosine']:.4f}")
    print(f"Frames used: {result['num_frames_used']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
