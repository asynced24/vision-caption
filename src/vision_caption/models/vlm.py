from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
from PIL import Image

from ..config import ModelConfig
from .language_decoder import LanguageDecoder
from .projector import Projector
from .vision_encoder import VisionEncoder


class VisionCaptioner(nn.Module):
    def __init__(self, config: ModelConfig, device: str, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype

        self.decoder = LanguageDecoder(
            model_name=config.language_model,
            dtype=dtype,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_target_modules=config.lora_target_modules,
            lora_adapter_path=config.lora_adapter_path,
        )
        self.encoder = VisionEncoder(config.vision_model, dtype=dtype)
        self.projector = Projector(
            vision_dim=self.encoder.hidden_size,
            text_dim=self.decoder.hidden_size,
        )

        self.to(device)
        self.eval()

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        inputs = self.encoder.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].to(device=self.device, dtype=self.dtype)

    def _prepare_prompt(self, prompt: str) -> torch.Tensor:
        if prompt.strip():
            tokens = self.decoder.tokenizer(prompt, return_tensors="pt")
            return tokens["input_ids"].to(self.device)
        bos_id = self.decoder.tokenizer.bos_token_id or self.decoder.tokenizer.eos_token_id
        return torch.tensor([[bos_id]], device=self.device)

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image | str,
        prompt: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 0.8,
        top_p: float = 0.9,
        do_sample: bool = True,
        **gen_kwargs: Any,
    ) -> str:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        prompt_text = prompt if prompt is not None else self.config.prompt
        pixel_values = self._prepare_image(image)
        input_ids = self._prepare_prompt(prompt_text)

        vision_features = self.encoder(pixel_values)
        image_embeds = self.projector(vision_features).to(dtype=self.dtype)

        text_embeds = self.decoder.embed_text(input_ids)
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        image_mask = torch.ones(
            (inputs_embeds.size(0), image_embeds.size(1)), device=self.device, dtype=torch.long
        )
        text_mask = torch.ones(
            (inputs_embeds.size(0), text_embeds.size(1)), device=self.device, dtype=torch.long
        )
        attention_mask = torch.cat([image_mask, text_mask], dim=1)

        output_ids = self.decoder.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.decoder.tokenizer.pad_token_id,
            eos_token_id=self.decoder.tokenizer.eos_token_id,
            **gen_kwargs,
        )

        text = self.decoder.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if prompt_text and text.startswith(prompt_text):
            text = text[len(prompt_text) :].lstrip(" :\n")
        return text.strip()
