"""
ViT_PMA_Bilinear — ViT_PMA(다성분 pooling + 4관점 co-attention, BRCA에서 M7 대비 +0.0535로
검증된 구조)는 그대로 두고, WSI-RNA를 합치는 마지막 단계만 concat에서 BilinearFusion
(Kronecker product, models/bilinear_fusion.py, PORPOISE 원본)으로 바꾼 조합.

[배경] 2026-08-31: PAAD(N=152)에서는 attention-guided WSI pooling(M4/M4A/PMA/MCAT 계열)이
전부 co-attention entropy 붕괴로 막혔고, 그걸 완전히 우회한 PORPOISE(RNA 무관 plain
gated-ABMIL + Kronecker fusion)가 그 계열을 확실히 이겼다. 그런데 BRCA(N=1058) 스케일에서
같은 PORPOISE(seed84)를 돌려보니 기존 PMA(co-attention 기반, seed42)보다 오히려 낮게
나왔다(0.6759 vs 0.7155) — PAAD와 정반대 방향. "표본이 커지면 co-attention이 patch/관점을
실제로 잘 고르기 시작해서 유리해지는 것 아니냐"는 가설과, "PORPOISE의 강점(Kronecker fusion)
자체는 표본 규모와 무관하게 유효할 수 있다"는 가설을 분리하기 위해, 이 둘을 합친 하이브리드로
직접 검증한다 — WSI pooling은 PMA 그대로(co-attention이 유리하다면 그 이점을 그대로 가져오고),
결합만 Kronecker로 바꿔서 PORPOISE의 결합 방식이 추가로 도움이 되는지 본다.

[ViT_PMA 대비 변경 지점]
  - combine_mode: 항상 "cox_add"로 고정(clinical은 별도 Cox 가산항, PORPOISE와 동일 관례) —
    Kronecker product는 두 벡터 사이 상호작용이라 z_wsi/z_rna/z_clinical 세 개를 한 번에
    넣을 자연스러운 방법이 없어서, PORPOISE가 이미 검증한 "clinical은 밖으로 뺀다" 선택을
    그대로 따른다.
  - combine_with_clinical_rna: co-attention으로 z_wsi(4관점 중 RNA가 골라 가중합, 또는
    use_coattn=False면 단순 평균)를 만드는 부분은 ViT_PMA와 완전히 동일하게 재현하고,
    그 뒤 [z_wsi, z_clinical, z_rna] concat 대신 BilinearFusion(z_wsi, z_rna)로 대체.
  - risk_head: 입력이 concat 폭(2~3*embed_dim)이 아니라 fusion.mmhid 기준이라 다시 만든다.
"""
import torch
import torch.nn as nn

from .vit_pma import ViT_PMA
from .vit_m4a import CoAttentionPooling
from .bilinear_fusion import BilinearFusion
from .clinical_encoder import STAGE_FIELDS, _STAGE_BUFFER_NAMES
from config import ModelConfig


class ViT_PMA_Bilinear(ViT_PMA):
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
        use_margin: bool = False,
        margin_stats: tuple[float, float] | None = None,
        use_age_sex: bool = True,
        drop_component: str | None = None,
        top_frac: float = 0.1,
        skip_patch_vit: bool = False,
        use_wsi_extra_mlp: bool = False,
        use_coattn: bool = True,
        fusion_gate: bool = True,
        fusion_dropout: float = 0.25,
    ):
        super().__init__(
            cfg, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
            precomputed=precomputed, backbone=backbone, num_heads=num_heads,
            use_staging=use_staging, stage_stats=stage_stats,
            use_margin=use_margin, margin_stats=margin_stats, use_age_sex=use_age_sex,
            combine_mode="cox_add", use_clinical=True,
            drop_component=drop_component, top_frac=top_frac,
            skip_patch_vit=skip_patch_vit, use_wsi_extra_mlp=use_wsi_extra_mlp,
            use_coattn=use_coattn,
        )
        rna_dim = cfg.embed_dim  # ViT_PMA는 rna_dim 기본값이 cfg.embed_dim(64) — fusion 양쪽 폭을 맞춘다
        self.fusion = BilinearFusion(cfg.embed_dim, rna_dim, mmhid=cfg.embed_dim,
                                      gate=fusion_gate, dropout=fusion_dropout)

        # ViT_PMA.__init__이 만든 risk_head는 [z_wsi, z_rna] concat(2*embed_dim) 기준이라
        # 안 맞는다 — fusion 출력(mmhid) 기준으로 새로 만든다.
        spatial_feat_dim = (2 if self.use_spatial_autocorr else 0) + (1 if self.use_attn_dispersion else 0)
        risk_stats_dim = 10 if use_wsi_extra_mlp else 0  # use_tile_risk_head 미노출(이 변형엔 안 씀)
        risk_input_dim = cfg.embed_dim + spatial_feat_dim + risk_stats_dim
        self.risk_head = nn.Sequential(
            nn.LayerNorm(risk_input_dim),
            nn.Linear(risk_input_dim, 1),
        )

    def combine_with_clinical_rna(
        self,
        patient_embed: torch.Tensor,  # (4, D) — 환자 단위로 평균 풀링된 4개 관점
        age_years: torch.Tensor,
        sex_idx: torch.Tensor,
        z_rna: torch.Tensor,
        stage_ord: dict[str, torch.Tensor] | None = None,
        margin_ord: torch.Tensor | None = None,
        spatial_feat: torch.Tensor | None = None,
        risk_stats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """z_wsi를 만드는 부분(co-attention 또는 단순 평균)은 ViT_PMA.combine_with_clinical_rna와
        완전히 동일 — 그 뒤 concat 대신 BilinearFusion(Kronecker product)으로 z_wsi/z_rna를
        합친다. clinical은 combine_mode="cox_add" 고정이라 여기 안 섞이고 train.py가
        _clinical_embed()로 별도 계산해 최종 스칼라에 Cox 가산항으로 더한다."""
        if self.use_coattn:
            z_wsi, _ = self.component_coattn(patient_embed, z_rna)  # (D,)
        else:
            z_wsi = patient_embed.mean(dim=0)  # (D,)
        fused = self.fusion(z_wsi, z_rna)  # (mmhid,)
        if spatial_feat is not None:
            fused = torch.cat([fused, spatial_feat], dim=-1)
        if risk_stats is not None:
            fused = torch.cat([fused, risk_stats], dim=-1)
        return fused
