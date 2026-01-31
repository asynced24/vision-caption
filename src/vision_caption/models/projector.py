import torch
import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, vision_dim: int, text_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or vision_dim * 4
        self.net = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, text_dim),
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return self.net(vision_features)
