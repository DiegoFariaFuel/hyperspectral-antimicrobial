from __future__ import annotations

import torch
from torch import nn


class SpectralCNN1D(nn.Module):
    def __init__(self, class_count: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, class_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).squeeze(-1))


class CubeCNN3D(nn.Module):
    def __init__(self, class_count: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, ceil_mode=True),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(64, class_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


def build_model(name: str, class_count: int) -> nn.Module:
    if name == "spectral":
        return SpectralCNN1D(class_count)
    if name == "cube3d":
        return CubeCNN3D(class_count)
    raise ValueError(f"unsupported model: {name}")


def model_input(cubes: torch.Tensor, model_name: str) -> torch.Tensor:
    """Convert (N,H,W,B) cubes to the tensor layout expected by a model."""
    if model_name == "spectral":
        return cubes.mean(dim=(1, 2)).unsqueeze(1)  # (N,1,B)
    if model_name == "cube3d":
        return cubes.permute(0, 3, 1, 2).unsqueeze(1)  # (N,1,B,H,W)
    raise ValueError(f"unsupported model: {model_name}")

