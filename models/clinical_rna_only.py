"""
ClinicalRNAOnly — M7, Clinical(age/sex)+RNA-seq 결합, WSI 없음. train_light.py --M7.

2026-07-19: RNA 인코더를 레퍼런스(Leeyoungsup/pancreatic_cancer_pathology)
scripts/models/tabular_survival.py 사양(RNAEncoderExtend, G->256->256, LayerNorm+Dropout
입출력 정규화)으로 교체 — GitHub 원문(모델 구조·RNA-seq 전처리·gene selection·split 프로토콜
전부)을 직접 확인해 이식했다. dim만 64로 낮춰본 ablation도 해봤지만 개선이 없어서(오히려
악화, findings_backlog.md 13번 항목) 원래 레퍼런스 폭(256)으로 되돌리고, 대신 학습 레시피
(epochs=100+early stopping patience=20, --dataset both)까지 전부 맞춰서 재검증한다.
Clinical은 레퍼런스와 동일하게 RNA(256)에 맞추지 않고 16차원 그대로 둔다(레퍼런스도
[rnaseq_embed_dim=256, clinical_embed_dim=16]로 비대칭). age/sex 정보량 자체는 기존과
동일(레퍼런스 clinical_dim=3=age+sex_onehot도 age/sex뿐, stage/grade 미사용 - GitHub
M4_Train.ipynb 주석으로 확인됨).

2026-07-21: risk_head를 레퍼런스(tabular_survival.py::ClinicalRNASeqSurvivalModel) 사양
(LayerNorm→Dropout(0.4)→Linear(272→128)→GELU→Dropout(0.4)→Linear(128→1))으로 교체해
M7_EX(기본 레시피)로 `--external` 재검증했으나 negative result(external C 0.634→0.533,
findings_backlog.md 13번 항목 참조) — 원래의 단순 선형(LayerNorm→Linear) 형태로 되돌렸다.

2026-07-21(후속, RNA 전처리 버그 수정 이후 재검증): 은닉층 없이 Dropout(0.4)만 추가한
최소 버전(레퍼런스 M4 사양)을 RNA 수정 후 M7_EX로 재검증했으나 또 negative — train_c_index가
30 epoch 만에 0.9865까지 치솟는데 val_c_index는 0.50대에 붙박이(교과서적 과적합), lifelines가
"collinearity or complete separation" 경고를 반복 — 학습 곡선 자체가 불안정/퇴화한 것으로
판단해 즉시 원복(사용자 판단, 결과 확정 전 중단). RNA를 고쳐도 risk head 직전 Dropout은
여전히 해롭다는 결론 유지 — 원래의 단순 선형(LayerNorm→Linear) 형태로 되돌렸다.
"""
import torch
import torch.nn as nn

from .clinical_encoder import ClinicalEncoder
from .rna_encoder_extend import RNAEncoderExtend
from config import ModelConfig

RNA_EMBED_DIM = 256
CLINICAL_EMBED_DIM = 16


class ClinicalRNAOnly(nn.Module):
    def __init__(self, cfg: ModelConfig, age_mean: float, age_std: float, rna_input_dim: int):
        super().__init__()
        self.clinical_encoder = ClinicalEncoder(CLINICAL_EMBED_DIM, age_mean, age_std)
        self.rna_encoder = RNAEncoderExtend(rna_input_dim, embed_dim=RNA_EMBED_DIM, hidden_dim=RNA_EMBED_DIM, dropout=0.25)
        self.risk_head = nn.Sequential(
            nn.LayerNorm(RNA_EMBED_DIM + CLINICAL_EMBED_DIM),
            nn.Linear(RNA_EMBED_DIM + CLINICAL_EMBED_DIM, 1),
        )

    def forward(self, age_years: torch.Tensor, sex_idx: torch.Tensor, rna: torch.Tensor) -> torch.Tensor:
        """
        Args:
            age_years: () — 환자 나이(연 단위) 스칼라 텐서
            sex_idx:   () — encode_sex() 인덱스 스칼라 텐서 (0=male, 1=female)
            rna:       (G,) — 코호트 내부 z-score 정규화된 유전자 발현 벡터
        Returns:
            risk: (1,)
        """
        z_c = self.clinical_encoder(age_years.unsqueeze(0), sex_idx.unsqueeze(0)).squeeze(0)  # (D,)
        z_r = self.rna_encoder(rna.unsqueeze(0)).squeeze(0)                                    # (D,)
        fused = torch.cat([z_c, z_r], dim=-1)                                                  # (2D,)
        return self.risk_head(fused.unsqueeze(0)).view(1)
