"""
UNI2hEncoder — Mahmood Lab UNI2-h(ViT-H/14) frozen tile encoder

models/uni_encoder.py::UNIEncoder(UNI, ViT-L/16, 1024-dim)의 상위 호환 버전. UNI2-h는 2025-01
공개된 더 크고(ViT-H, 6.81억 파라미터) 더 많이 학습된(200M+ 타일, H&E+IHC 혼합) 버전으로,
patch size(16→14)·embed_dim(1024→1536)·FFN 구조(SwiGLU)까지 바뀌어 timm 로딩 kwargs가
UNIEncoder와 다르다 — 그래서 같은 클래스를 확장하지 않고 별도 파일로 둔다(UNI/UNI2-h 둘 다
당분간 롤백 가능하게 병행 유지).

가중치는 HuggingFace Hub의 gated repo(MahmoodLab/UNI2-h)에서 받는다 — 최초 1회, 사용자가
huggingface.co/MahmoodLab/UNI2-h에서 접근 승인을 받고 .env에 HF_TOKEN을 설정해야 한다
(utils.load_env() 참조, UNI와 동일한 HF_TOKEN 재사용 가능). 캐시: ~/.cache/huggingface/hub/

timm_kwargs는 MahmoodLab 공식 모델 카드(huggingface.co/MahmoodLab/UNI2-h) 그대로다 —
mlp_layer(SwiGLUPacked)/act_layer(SiLU)/reg_tokens(8) 중 하나라도 빠지면 state_dict 로드가
"Unexpected/Missing key(s)" 에러로 실패한다(UNIEncoder의 init_values=1e-5 필수 사유와 동일
문제 클래스).
"""
import torch
import torch.nn as nn
import timm
from timm.layers import SwiGLUPacked

BACKBONE_DIM = 1536
UNI2H_HF_ID  = "hf_hub:MahmoodLab/UNI2-h"

_TIMM_KWARGS = dict(
    img_size=224,
    patch_size=14,
    depth=24,
    num_heads=24,
    init_values=1e-5,
    embed_dim=BACKBONE_DIM,
    mlp_ratio=2.66667 * 2,
    num_classes=0,
    no_embed_class=True,
    mlp_layer=SwiGLUPacked,
    act_layer=torch.nn.SiLU,
    reg_tokens=8,
    dynamic_img_size=True,  # 224 외 해상도 입력도 허용(positional embedding 보간) — UNI와 동일 관례
)


def _build_backbone(pretrained: bool) -> nn.Module:
    return timm.create_model(UNI2H_HF_ID, pretrained=pretrained, **_TIMM_KWARGS)


class UNI2hEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, pretrained: bool = True, with_backbone: bool = True):
        """
        Args:
            with_backbone: False면 backbone을 생성하지 않는다 — 사전 추출된
                           pooled feature(utils/extract_features.py --backbone uni2 산출물)만
                           사용하는 모드에서 불필요한 GPU 메모리/로딩 시간을 아낀다.
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
            x: (N_patches, 3, H, W)
        Returns:
            features: (N_patches, embed_dim)
        """
        if self.backbone is None:
            raise RuntimeError("backbone이 없는 UNI2hEncoder(with_backbone=False)입니다 — "
                               "forward_pooled()으로 사전 추출된 feature를 전달하세요.")
        pooled = self.backbone(x)  # (N_patches, 1536) — ViT라 이미 pooled 출력(reg_tokens 제외)
        return self.proj(pooled)

    def forward_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled: (N_patches, 1536) - backbone까지 미리 계산해 캐싱해둔 feature
        Returns:
            features: (N_patches, embed_dim)
        """
        return self.proj(pooled)
