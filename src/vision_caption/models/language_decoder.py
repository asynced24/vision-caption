from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


class LanguageDecoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: list[str] | tuple[str, ...],
        lora_adapter_path: str | None = None,
    ) -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True
        )
        base_model.config.pad_token_id = self.tokenizer.pad_token_id

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base_model, lora_config)

        if lora_adapter_path:
            self.model.load_adapter(lora_adapter_path, adapter_name="default")
            self.model.set_adapter("default")

        self.hidden_size = base_model.config.hidden_size

    def embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)
