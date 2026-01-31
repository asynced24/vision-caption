import torch
import torch.nn as nn
from transformers import SiglipImageProcessor, SiglipVisionModel


class VisionEncoder(nn.Module):
    def __init__(self, model_name: str, dtype: torch.dtype) -> None:
        super().__init__()
        self.model = SiglipVisionModel.from_pretrained(model_name, torch_dtype=dtype)
        self.processor = SiglipImageProcessor.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state
