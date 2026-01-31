from __future__ import annotations

from typing import Optional
import torch
from PIL import Image

from .config import ModelConfig
from .models.vlm import VisionCaptioner


def _resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.float16
    return torch.float32


def load_model(config: Optional[ModelConfig] = None, device: Optional[str] = None) -> VisionCaptioner:
    config = config or ModelConfig()
    device = _resolve_device(device)
    dtype = _resolve_dtype(device)
    return VisionCaptioner(config=config, device=device, dtype=dtype)


def generate(
    model: VisionCaptioner,
    image: Image.Image | str,
    prompt: Optional[str] = None,
    **kwargs,
) -> str:
    return model.generate(image=image, prompt=prompt, **kwargs)
