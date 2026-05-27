"""
CNN Feature Extractor
- 패치 수준의 Local Morphological Features 및 세포핵 밀집도 포착
- Pretrained backbone으로 feature map 추출 후 embed_dim으로 projection
"""
import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet50", embed_dim: int = 512, pretrained: bool = True):
        super().__init__()

        # TODO: EfficientNet, ConvNeXt 등 다른 backbone 실험
        # TODO: 병리 이미지 특화 pretrained weight 사용 고려 (e.g., UNI, CONCH, PLIP)
        if backbone == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.backbone = nn.Sequential(*list(base.children())[:-2])  # (B, 2048, H/32, W/32)
            cnn_out_dim = 2048
        elif backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            cnn_out_dim = 512
        else:
            raise NotImplementedError(f"Backbone '{backbone}' not implemented")

        # 채널 축소 후 embed_dim으로 projection
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B, C, 1, 1)
            nn.Flatten(),              # (B, C)
            nn.Linear(cnn_out_dim, embed_dim),
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
