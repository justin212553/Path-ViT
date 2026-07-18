"""
ViT_M4A_FF — M4A_EX에서 Nystromformer의 FFN 서브레이어를 제거한 맛보기 ablation. train.py --M4A_FF.

attention 서브레이어(패치 간 정보를 섞음)와 FFN 서브레이어(섞인 결과를 패치 하나 단위로
비선형 변환)는 서로 독립적인 역할이다 — FFN은 다른 패치를 전혀 참조하지 않으므로, FFN을
빼도 attention이 만든 "공간 컨텍스트가 반영된 패치 표현" 자체는 그대로 남고 그 이후의
비선형 다듬기 단계만 없어진다. M4A_EX(co-attention pooling)는 그대로 두고 이 부분만 바꿔서,
"공간 컨텍스트 반영 vs 그걸 더 다듬는 표현력"이 결과에 얼마나 기여하는지 분리해서 본다.
"""
from .vit_m4a import ViT_M4A
from .vit_encoder import ViTEncoder
from config import ModelConfig


class ViT_M4A_FF(ViT_M4A):
    def __init__(
        self,
        cfg: ModelConfig,
        age_mean: float,
        age_std: float,
        rna_input_dim: int,
        precomputed: bool = True,
        backbone: str = "resnet50",
        num_heads: int = 4,
        use_staging: bool = False,
        stage_stats: dict[str, tuple[float, float]] | None = None,
    ):
        super().__init__(cfg, age_mean, age_std, rna_input_dim, precomputed, backbone, num_heads,
                          use_staging=use_staging, stage_stats=stage_stats)
        # ViT_M1.__init__이 만든 self.vit(use_ffn=True)를 FFN 없는 버전으로 교체한다.
        self.vit = ViTEncoder(
            cfg.embed_dim, cfg.num_heads, cfg.num_transformer_layers, cfg.dropout,
            use_grad_checkpoint=cfg.grad_checkpoint, num_landmarks=cfg.num_landmarks,
            use_ffn=False,
        )
