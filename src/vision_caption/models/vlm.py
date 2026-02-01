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

        # Load smaller vision encoder first to reduce memory fragmentation
        self.encoder = VisionEncoder(config.vision_model, dtype=dtype)
        self.decoder = LanguageDecoder(
            model_name=config.language_model,
            dtype=dtype,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_target_modules=config.lora_target_modules,
            lora_adapter_path=config.lora_adapter_path,
        )
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

    def _sample_next_token(
        self, logits: torch.Tensor, temperature: float, top_p: float, do_sample: bool
    ) -> torch.Tensor:
        if not do_sample:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)

        # Top-p (nucleus) sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

        next_token_sorted = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, next_token_sorted)
        return next_token

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

        max_tokens = max_new_tokens or self.config.max_new_tokens
        eos_id = self.decoder.tokenizer.eos_token_id
        generated_ids: list[int] = []

        # Manual autoregressive loop — bypasses generate() quirks with inputs_embeds
        for _ in range(max_tokens):
            outputs = self.decoder.model.base_model(
                inputs_embeds=inputs_embeds,
                use_cache=False,
                return_dict=True,
            )
            next_logits = outputs.logits[:, -1, :]
            next_token = self._sample_next_token(next_logits, temperature, top_p, do_sample)

            token_id = next_token.item()
            if token_id == eos_id:
                break
            generated_ids.append(token_id)

            # Append new token embedding for next iteration
            next_embed = self.decoder.embed_text(next_token)
            inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)

        text = self.decoder.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()