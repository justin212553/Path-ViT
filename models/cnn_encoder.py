"""
CNN Feature Extractor
- 패치 수준의 Local Morphological Features 및 세포핵 밀집도 포착
- Pretrained backbone으로 feature map 추출 후 embed_dim으로 projection
"""
import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, pretrained: bool = True):
        super().__init__()

        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(base.children())[:-2])  # (B, 2048, H/32, W/32)

        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B, C, 1, 1)
            nn.Flatten(),              # (B, C)
            nn.Linear(2048, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N_patches, 3, H, W) - 슬라이드에서 추출한 패치 배치
        Returns:
            features: (N_patches, embed_dim)
        """
        feat_map = self.backbone(x)   # (N, C, h, w)
        return self.proj(feat_map)    # (N, embed_dim)
