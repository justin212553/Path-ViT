"""
StagePredictionHead — WSI 표현에서 T-stage/grade를 예측하는 보조과제(auxiliary task) 헤드.
train.py --stage-aux-weight.

models/rna_predictor.py::RNAPredictionHead와 같은 설계 원칙: 예측값 자체는 risk_head에
노출되지 않고 버려진다 - 오직 그래디언트만 WSI 인코더로 흘려보내 "형태학적으로 병기/등급과
상관된 특징"을 뽑도록 정규화하는 게 목적이다(RNA-free meanpool_embed 입력, ViT 직후 mean
pooling이라 attn_pool의 RNA/clinical 개입과 무관하게 항상 RNA-free/clinical-free).

[N/M-stage가 아니라 T-stage/grade인 이유] 우리 WSI 데이터는 원발암(primary tumor) 슬라이드만
있고 임파선/원격전이 슬라이드가 없다(TCGA barcode sample-type 확인 완료) - N/M-stage는 WSI
자체에서 판단할 근거가 없다. 반면 T-stage(종양 크기/침습 범위)와 grade(분화도)는 원발암
조직 형태 자체에서 판단 가능한, 문헌에서도 흔한 WSI-MIL 타깃이다.

[T/grade 결측 처리] TX/GX("판정 불가")나 NaN(진짜 결측)인 환자는 그 필드의 loss 항목에서
제외한다(둘 다 <1~4건/코호트로 드묾). 두 필드 다 미상인 환자는 이 보조 loss 자체가 None이
된다(train.py::_patient_risk가 이 경우를 건너뛴다).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .clinical_encoder import STAGE_FIELDS


class StagePredictionHead(nn.Module):
    def __init__(self, embed_dim: int, stage_stats: dict[str, tuple[float, float]],
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        t_mean, t_std = stage_stats["ajcc_t"]
        g_mean, g_std = stage_stats["tumor_grade"]
        self.register_buffer("t_mean", torch.tensor(t_mean, dtype=torch.float32))
        self.register_buffer("t_std", torch.tensor(t_std, dtype=torch.float32))
        self.register_buffer("g_mean", torch.tensor(g_mean, dtype=torch.float32))
        self.register_buffer("g_std", torch.tensor(g_std, dtype=torch.float32))
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # [t_stage_z_pred, grade_z_pred]
        )

    def forward(self, wsi_meanpool_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wsi_meanpool_embed: (D,) — RNA-free/clinical-free mean-pooled WSI 표현(환자 단위)
        Returns:
            pred: (2,) — [t_stage_z, grade_z] 예측
        """
        return self.mlp(wsi_meanpool_embed)

    def loss(self, wsi_meanpool_embed: torch.Tensor,
              t_ord: torch.Tensor, grade_ord: torch.Tensor) -> torch.Tensor | None:
        """
        Args:
            wsi_meanpool_embed: (D,)
            t_ord, grade_ord: () 스칼라 long 텐서 — encode_stage_value() 순서형 정수,
                               "미상"은 -1(data/dataset.py 규약).
        Returns:
            MSE loss(known 필드만 평균), 둘 다 미상이면 None.
        """
        pred = self.mlp(wsi_meanpool_embed)  # (2,)
        targets, preds = [], []
        if t_ord.item() >= 0:
            targets.append((t_ord.float() - self.t_mean) / self.t_std)
            preds.append(pred[0])
        if grade_ord.item() >= 0:
            targets.append((grade_ord.float() - self.g_mean) / self.g_std)
            preds.append(pred[1])
        if not targets:
            return None
        return F.mse_loss(torch.stack(preds), torch.stack(targets))
