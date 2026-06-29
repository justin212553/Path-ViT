"""
CNN Feature Extractor
- 패치 수준의 Local Morphological Features 및 세포핵 밀집도 포착
- Pretrained backbone으로 feature map 추출 후 embed_dim으로 projection
"""
import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, pretrained: bool = True, with_backbone: bool = True):
        """
        Args:
            with_backbone: False면 ResNet50 backbone을 생성하지 않는다 — 사전 추출된
                           pooled feature(data/extract_features.py 산출물)만 사용하는
                           모드에서 불필요한 GPU 메모리/로딩 시간을 아낀다.
        """
        super().__init__()

        self.backbone = None
        if with_backbone:
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.backbone = nn.Sequential(*list(base.children())[:-2])  # (B, 2048, H/32, W/32)

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B, C, 1, 1)
            nn.Flatten(),              # (B, C)
        )
        self.proj = nn.Sequential(
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
        if self.backbone is None:
            raise RuntimeError("backbone이 없는 CNNEncoder(with_backbone=False)입니다 — "
                                "forward_pooled()으로 사전 추출된 feature를 전달하세요.")
        feat_map = self.backbone(x)        # (N, C, h, w)
        return self.proj(self.pool(feat_map))  # (N, embed_dim)

    def forward_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled: (N_patches, 2048) - backbone+pool까지 미리 계산해 캐싱해둔 feature
        Returns:
            features: (N_patches, embed_dim)
        """
        return self.proj(pooled)
