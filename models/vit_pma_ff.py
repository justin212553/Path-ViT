"""
ViT_PMA_FF — PMA_EX에서 Nystromformer의 FFN 서브레이어를 제거한 맛보기 ablation. train.py --PMA_FF.

8번 항목(M4A_FF)과 동일한 논리를 PMA에 적용한다 - attention 서브레이어(패치 간 정보를 섞음)와
FFN 서브레이어(섞인 결과를 패치 하나 단위로 비선형 변환)는 독립적인 역할이라, FFN을 빼도
attention이 만든 공간 컨텍스트 자체는 유지된다. M4A_FF는 FFN 제거 단독으로는 null result였지만
(findings_backlog.md 8번 항목), PMA_EX_SS_AUX(다성분 pooling + 패치 드롭아웃 + RNA aux, 9번 항목)
기준에서도 같은지 마지막으로 확인한다.
"""
from .vit_pma import ViT_PMA
from .vit_encoder import ViTEncoder
from config import ModelConfig


class ViT_PMA_FF(ViT_PMA):
    def __init__(
        self,
        cfg: ModelConfig,
        age_mean: float,
        age_std: float,
        rna_input_dim: int,
        precomputed: bool = True,
        backbone: str = "resnet50",
        num_heads: int = 2,
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
