from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def _ensure_dynamic_cache_compat() -> None:
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return

    if not hasattr(DynamicCache, "get_seq_length"):
        DynamicCache.get_seq_length = lambda self: 0

    patches = {
        "seen_tokens": property(lambda self: self.get_seq_length()),
        "get_max_length": lambda self, *args, **kwargs: None,
        "get_usable_length": lambda self, *args, **kwargs: self.get_seq_length(),
    }
    for name, value in patches.items():
        if not hasattr(DynamicCache, name):
            setattr(DynamicCache, name, value)

    # Phi-3 calls DynamicCache.from_legacy_cache(None) when use_cache=True and no cache is passed.
    # Provide a safe fallback that returns an empty cache.
    if hasattr(DynamicCache, "from_legacy_cache"):
        original_from_legacy_cache = DynamicCache.from_legacy_cache

        def _from_legacy_cache(cache):
            if cache is None:
                return DynamicCache()
            return original_from_legacy_cache(cache)

        DynamicCache.from_legacy_cache = staticmethod(_from_legacy_cache)


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

        _ensure_dynamic_cache_compat()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        base_model.config.pad_token_id = self.tokenizer.pad_token_id

        # LoRA for parameter-efficient fine-tuning
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
