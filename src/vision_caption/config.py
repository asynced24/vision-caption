from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class ModelConfig:
    vision_model: str = "google/siglip-so400m-patch14-384"
    language_model: str = "Qwen/Qwen2-1.5B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj")
    lora_adapter_path: Optional[str] = None
    projector_path: Optional[str] = None  # Path to trained projector weights

    prompt: str = "Describe the image."
    max_new_tokens: int = 64
