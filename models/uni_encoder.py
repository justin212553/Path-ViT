"""
UNIEncoder — Mahmood Lab UNI(ViT-L/16) frozen tile encoder

cnn_encoder.py::CNNEncoder(ResNet50 Lunit SwAV)와 같은 자리를 대체하는 병리 foundation
model 기반 패치 인코더. UNI는 패치 이미지 1장을 곧장 (1024,) 벡터로 뽑아내는 ViT라서
CNNEncoder처럼 별도 spatial pooling(AdaptiveAvgPool2d) 단계가 필요 없다 — timm이
num_classes=0으로 이미 pooled(cls token) 출력을 반환한다.

가중치는 HuggingFace Hub의 gated repo(MahmoodLab/UNI)에서 받는다 — 최초 1회, 사용자가
huggingface.co/MahmoodLab/UNI에서 접근 승인을 받고 .env에 HF_TOKEN을 설정해야 한다
(utils.load_env() 참조). 캐시: ~/.cache/huggingface/hub/

init_values=1e-5는 UNI 체크포인트가 갖고 있는 LayerScale 파라미터(blocks.*.ls1/ls2.gamma)를
맞추기 위해 필수다 — 이게 없으면 timm 기본 vit_large_patch16_224 설정에는 LayerScale 모듈이
없어서 state_dict 로드가 "Unexpected key(s)" 에러로 실패한다.
"""
import torch
import torch.nn as nn
import timm

BACKBONE_DIM = 1024
UNI_HF_ID    = "hf_hub:MahmoodLab/UNI"


def _build_backbone(pretrained: bool) -> nn.Module:
    return timm.create_model(
        UNI_HF_ID,
        pretrained=pretrained,
        num_classes=0,
        init_values=1e-5,
        dynamic_img_size=True,  # 224 외 해상도 입력도 허용(positional embedding 보간)
    )


class UNIEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, pretrained: bool = True, with_backbone: bool = True):
        """
        Args:
            with_backbone: False면 backbone을 생성하지 않는다 — 사전 추출된
                           pooled feature(data/extract_features.py 산출물)만 사용하는
                           모드에서 불필요한 GPU 메모리/로딩 시간을 아낀다.
        """
        super().__init__()
        self.backbone = _build_backbone(pretrained) if with_backbone else None
        self.proj = nn.Sequential(
            nn.Linear(BACKBONE_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N_patches, 3, 224, 224)
        Returns:
            features: (N_patches, embed_dim)
        """
        if self.backbone is None:
            raise RuntimeError("backbone이 없는 UNIEncoder(with_backbone=False)입니다 — "
                               "forward_pooled()으로 사전 추출된 feature를 전달하세요.")
        pooled = self.backbone(x)  # (N_patches, 1024) — UNI는 ViT라 이미 pooled 출력
        return self.proj(pooled)

    def forward_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled: (N_patches, 1024) - backbone까지 미리 계산해 캐싱해둔 feature
        Returns:
            features: (N_patches, embed_dim)
        """
        return self.proj(pooled)
