"""
ViT_PORPOISE — PORPOISE(Chen et al. 2022, Cancer Cell) 스타일 bilinear/Kronecker product fusion.
train.py --PORPOISE. 2026-08-31 3단계 계획의 Phase 3.

[왜 Phase 3인가]
Phase 1(MCAT 스타일 multi-pathway co-attention, models/vit_mcat.py)까지 포함해 M4/M4A/PMA/MCAT
전부 "RNA(또는 pathway 토큰)가 WSI patch 전체에 co-attention/RNA-guided attention으로 중요
patch를 찾아낸다"는 같은 계열의 구조였다. findings_backlog.md 최상위 발견(2026-08-31)에서
이 계열 전체가 같은 벽에 부딪힌다는 게 확인됐다 — query를 1개(M4A)에서 8개(MCAT)로 늘려도
co-attention entropy가 여전히 0.9998(거의 완전 uniform)로 붕괴하고, 이게 gradient 부족
때문도 아니었다(scripts/diagnose_mcat_gradients.py, attn_pool gradient가 30 epoch 내내
안정적으로 도달함에도 붕괴 재발). 즉 "이 WSI feature 공간엔 patch 단위로 구별해 Cox loss를
낮출 신호가 거의 없다"는 게 attention 계열 구조를 아무리 바꿔도 반복 재현되는 근본 벽이다.

PORPOISE는 이 벽을 정면 돌파하는 대신 **비켜간다** — WSI 풀링(attn_pool)이 RNA를 전혀
참조하지 않는 평범한 gated-ABMIL(models/vit_m1.py::AttentionPooling, context 없음)로 patch를
집계하고, WSI-RNA 상호작용은 풀링 *이후* z_wsi(D,)와 z_rna(D,) 사이의 명시적 pairwise
interaction(Kronecker/outer product, models/bilinear_fusion.py::BilinearFusion)으로 포착한다.
"어느 patch가 중요한지 attention이 찾아내야 한다"는 전제 자체가 없다 — patch 판별력이 약해도
(findings_backlog.md의 결론대로) z_wsi 자체(단순 통계적 요약)와 z_rna의 pairwise 조합이 risk에
기여할 여지는 남아 있다는 발상.

[ViT_M4 대비 구조 변경 지점]
  - attn_pool: RNA-guided(context_dim 있음) → 평범한 gated-ABMIL(context_dim 없음, RNA 무시)
  - combine_with_clinical_rna: [z_wsi ‖ z_rna] concat → BilinearFusion(z_wsi, z_rna)
  - combine_mode: 반드시 "cox_add"로 고정(clinical은 항상 별도 Cox 가산항, ViT_M4/PMA와 동일
    관례) — PORPOISE 원 논문도 clinical/grade 등 부가 정보를 genomic 벡터에 섞기보다 별도
    처리하는 경우가 흔해, 이 프로젝트의 기존 raw-feature-direct cox_add 관례를 그대로 재사용.
  - risk_head: 입력이 (2~3)*embed_dim(concat)이 아니라 fusion.mmhid(기본 embed_dim)이므로
    ViT_M4가 만든 risk_head를 폐기하고 다시 만든다.
"""
import torch
import torch.nn as nn

from .vit_m1 import AttentionPooling
from .vit_m4 import ViT_M4
from .vit_m4a import CoAttentionPooling
from .bilinear_fusion import BilinearFusion
from config import ModelConfig


class MeanPooling(nn.Module):
    """무파라미터 mean-pooling — AttentionPooling(gated-ABMIL)과 동일한 (wsi_embed, attn_weights)
    반환 규약을 지켜 attn_pool 자리에 그대로 꽂아 쓸 수 있다.

    2026-08-31 배경: scripts/diagnose_porpoise_reliance.py로 확인한 plain gated-ABMIL의 patch
    attention entropy가 0.999(거의 완전 uniform)였다 — RNA 간섭을 없애도 patch를 못 골랐다.
    그런데 이 결과가 PAAD(N≈90)뿐 아니라 BRCA(N≈1058, findings_backlog.md 2026-07-22 heatmap
    확인: co-attention 4-관점 가중치 0.24~0.27로 균등)에서도 재현됐다 — 표본을 8배 늘려도
    attention이 "선택"하는 역할을 한 번도 못 해봤다는 뜻이다. 즉 지금 attn_pool은 사실상
    비싼(파라미터 있는) mean-pool을 흉내내고 있을 뿐이므로, 아예 진짜 mean-pool로 바꿔도
    성능이 같아야 한다는 가설을 직접 검증한다 — 같으면 attention 모듈 자체가 이 태스크에
    불필요하다는 실증이 되고, 파라미터도 줄어 이 표본 규모(91명)의 과적합 위험도 낮아진다.

    attn_weights를 균등분포(1/N)로 반환하는 이유: --attn-dispersion(models/spatial_features.py::
    attention_dispersion(coords, attn_weights))이 이 값을 그대로 쓰는데, 균등 가중치를 넣으면
    "전체 patch의 공간적 퍼짐"이라는 여전히 의미 있는 기하학적 통계가 나온다(0으로 채우거나
    None을 주면 이 항이 깨짐) — dispersion ablation(2026-08-31)에서 dispersion 자체는 PORPOISE
    성능에 크게 기여한다고 확인됐으므로, mean-pool로 바꿔도 이 경로는 그대로 살려둔다.
    """

    def forward(self, tokens: torch.Tensor, context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        n = tokens.shape[0]
        wsi_embed = tokens.mean(dim=0)
        attn_weights = tokens.new_full((n,), 1.0 / n)
        return wsi_embed, attn_weights


class ViT_PORPOISE(ViT_M4):
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
        use_margin: bool = False,
        margin_stats: tuple[float, float] | None = None,
        use_age_sex: bool = True,
        use_attn_dispersion: bool = False,
        skip_patch_vit: bool = False,
        fusion_gate: bool = True,
        fusion_dropout: float = 0.25,
        use_meanpool: bool = False,
        use_coattn: bool = False,
        num_heads: int = 4,
        attn_temperature: float = 1.0,
    ):
        if use_meanpool and use_coattn:
            raise ValueError("use_meanpool과 use_coattn은 동시에 켤 수 없습니다(attn_pool 자리가 하나뿐).")
        super().__init__(cfg, age_mean, age_std, rna_input_dim, precomputed, backbone,
                          use_staging=use_staging, stage_stats=stage_stats,
                          use_margin=use_margin, margin_stats=margin_stats, use_age_sex=use_age_sex,
                          combine_mode="cox_add", use_attn_dispersion=use_attn_dispersion,
                          skip_patch_vit=skip_patch_vit, use_clinical=True)

        # ViT_M4가 만든 RNA-guided attn_pool(context_dim=embed_dim)을 평범한 gated-ABMIL로
        # 교체 — PORPOISE는 원래 WSI 풀링 단계에서 RNA를 전혀 참조하지 않는다.
        #   use_meanpool=True: 그 gated-ABMIL마저 무파라미터 MeanPooling으로 바꾼다(위 클래스
        #     docstring 참조 — attention이 patch를 못 고른다는 게 이미 확인됐으니, "학습되는
        #     균등 근사"를 진짜 균등으로 바꿔도 성능이 같은지 직접 검증하는 ablation).
        #   use_coattn=True: 2026-08-31 "나이스트롬이 patch 간 차이를 뭉개서 그 뒤의 RNA
        #     co-attention이 구별을 못 한 것 아니냐" 가설(스크립트: m4a_skip_patch_vit_pilot)을
        #     BilinearFusion과 결합해서도 검증하는 조합 — models/vit_m4a.py::CoAttentionPooling
        #     (M4A와 동일 RNA-query cross-attention)을 그대로 재사용, skip_patch_vit=True와
        #     같이 켜면 "나이스트롬 없는 RNA co-attention + Kronecker fusion"이 된다. M4A는
        #     concat fusion이라 이 조합(co-attention + Kronecker)은 M4A 단독 실험과 별개다.
        if use_meanpool:
            self.attn_pool = MeanPooling()
        elif use_coattn:
            self.attn_pool = CoAttentionPooling(cfg.embed_dim, num_heads=num_heads, dropout=cfg.dropout)
        else:
            # attn_temperature: score/T로 softmax 이전에 나눈다(models/vit_m1.py::AttentionPooling
            # 참조) — 2026-08-31 재학습-없는 post-hoc sharpening은 역효과였어서(diagnose 결과),
            # 학습 자체를 이 값을 알고 하게 만드는 ablation(train.py --porpoise-attn-temperature).
            self.attn_pool = AttentionPooling(cfg.embed_dim, temperature=attn_temperature)

        self.fusion = BilinearFusion(cfg.embed_dim, cfg.embed_dim, mmhid=cfg.embed_dim,
                                      gate=fusion_gate, dropout=fusion_dropout)

        # ViT_M4.__init__이 만든 risk_head는 [z_wsi, z_rna] concat(2*embed_dim) 기준이라 여기선
        # 안 맞는다 — fusion 출력(mmhid) 기준으로 새로 만든다.
        spatial_feat_dim = 1 if use_attn_dispersion else 0
        risk_input_dim = cfg.embed_dim + spatial_feat_dim
        self.risk_head = nn.Sequential(
            nn.LayerNorm(risk_input_dim),
            nn.Linear(risk_input_dim, 1),
        )

    def combine_with_clinical_rna(
        self,
        patient_embed: torch.Tensor,
        age_years: torch.Tensor,
        sex_idx: torch.Tensor,
        z_rna: torch.Tensor,
        stage_ord: dict[str, torch.Tensor] | None = None,
        margin_ord: torch.Tensor | None = None,
        spatial_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """clinical은 ViT_M4와 동일하게 여기 안 섞이고(combine_mode="cox_add" 고정) train.py가
        _clinical_embed()로 별도 계산해 최종 스칼라에 Cox 가산항으로 더한다. 여기서는 WSI-RNA만
        BilinearFusion(Kronecker product)으로 융합한다."""
        fused = self.fusion(patient_embed, z_rna)  # (mmhid,)
        if spatial_feat is not None:
            fused = torch.cat([fused, spatial_feat], dim=-1)
        return fused
