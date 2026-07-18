"""
ViT_M2_FF — M2(WSI+Clinical)에 RNA를 ViTEncoder의 FFN 서브레이어 직전 FiLM(가산 bias)으로만
개입시키는 맛보기 ablation. train.py --M2_FF.

[설계 의도] RNA가 최종 결합(risk_head 직전 concat)에 직접 노출되지 않는다 — M4/M4A/PM4/PMA는
전부 [z_wsi, z_clinical, z_rna]를 concat해서, RNA 신호가 (1) WSI 표현을 조절하는 경로와
(2) risk_head에 직접 들어가는 경로 두 군데로 들어간다. 이 모델은 (2)를 없애고 오직 (1)만
남겨서 — "RNA가 WSI 표현을 바꾸는 것"만으로 신호가 살아남는지를 순수하게 본다.

[풀링] attn_pool(ABMIL)을 쓰지 않고 mean pooling으로 patch 토큰을 집계한다 — RNA-FiLM이
이미 각 패치 토큰 자체를 조건화했으니, 그 위에 또 학습되는 attention 게이트를 얹지 않고
가장 단순한 방식으로 집계해 "패치 표현 조건화 자체"의 순수 기여를 본다.

[combine] ViT_M2.combine_with_clinical()을 그대로 재사용 — RNA 없이 [z_wsi, z_clinical] (2D,)만
concat한다. train.py::_patient_risk가 model.rna_encoder 존재 여부로 z_rna를 계산해 forward()에
rna_context로 넘겨주는 배선은 기존과 동일하게 재사용되지만(encode_rna 필요), 최종 결합
분기는 hasattr(model, "combine_with_clinical_rna")가 아니라 hasattr(model, "combine_with_clinical")
로 잡히도록 이 모델은 combine_with_clinical_rna를 정의하지 않는다(train.py 쪽 분기 수정 참조).
"""
import torch

from .vit_m2 import ViT_M2
from .vit_encoder import ViTEncoder
from .rna_encoder import RNAEncoder
from config import ModelConfig


class ViT_M2_FF(ViT_M2):
    def __init__(
        self,
        cfg: ModelConfig,
        age_mean: float,
        age_std: float,
        rna_input_dim: int,
        precomputed: bool = True,
        backbone: str = "resnet50",
        use_staging: bool = False,
        stage_stats: dict[str, tuple[float, float]] | None = None,
    ):
        super().__init__(cfg, age_mean, age_std, precomputed, backbone,
                          use_staging=use_staging, stage_stats=stage_stats)
        self.rna_encoder = RNAEncoder(rna_input_dim, cfg.embed_dim, dropout=cfg.dropout)
        # ViT_M1.__init__이 만든 self.vit을, FFN 직전 RNA FiLM을 받는 버전으로 교체한다.
        self.vit = ViTEncoder(
            cfg.embed_dim, cfg.num_heads, cfg.num_transformer_layers, cfg.dropout,
            use_grad_checkpoint=cfg.grad_checkpoint, num_landmarks=cfg.num_landmarks,
            use_ffn=True, context_dim=cfg.embed_dim,
        )
        # attn_pool(ABMIL)은 안 쓴다(mean pooling으로 대체) — 상속받은 모듈이라 그대로 남아있지만
        # forward()에서 호출하지 않으므로 학습에 영향 없다.

    def encode_rna(self, rna: torch.Tensor) -> torch.Tensor:
        return self.rna_encoder(rna.unsqueeze(0)).squeeze(0)

    def forward(
        self,
        coords: torch.Tensor,
        patch_paths=None,
        features: torch.Tensor | None = None,
        transform=None,
        chunk_size: int | None = None,
        rna_context: torch.Tensor | None = None,
    ) -> dict:
        patch_tokens = self._patch_tokens(coords, patch_paths, features, transform, chunk_size)
        ctx_tokens = self.vit(patch_tokens, coords, context=rna_context)  # (N, D) — FFN 직전 RNA FiLM 적용됨
        wsi_embed = ctx_tokens.mean(dim=0)  # (D,) — ABMIL 대신 mean pooling
        attn_weights = torch.full(
            (ctx_tokens.shape[0],), 1.0 / ctx_tokens.shape[0], device=ctx_tokens.device
        )
        return {"embed": wsi_embed, "attn_weights": attn_weights}
