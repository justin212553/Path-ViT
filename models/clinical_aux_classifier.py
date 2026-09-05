"""
ClinicalAuxClassifier — models/stage_predictor.py::StagePredictionHead와 같은 설계 원칙(WSI만의
RNA-free/clinical-free meanpool_embed을 입력으로 받아, 예측값 자체는 버리고 그래디언트만 WSI
인코더로 흘려보내는 보조과제)을 3개 병리학적 축의 **분류** 문제로 확장한다.

사용자 요청(2026-09-04): "CPTAC에 이미 있는 staging/신경침윤(PNI)/면역침윤 3가지를 classification
하는 aux를 만들어서 CPTAC로 돌리자" — StagePredictionHead는 T-stage/grade를 z-score 회귀로
풀었지만, 여기서는 3개 다 분류(cross-entropy)로 통일한다.

  stage:  AJCC T-stage(Tis~T4, models/clinical_encoder.py::_STAGE_ORDINAL_MAPS["ajcc_t"], 5클래스)
  pni:    신경주위침윤 유무(CPTAC 전용, data/cptac_pni_immune_aux.csv, cBioPortal paad_cptac_2021
          PERINEURAL_INVASION "Present"=1/"Not identified"=0, 2클래스)
  immune: 면역세포침윤 비율(CPTAC 전용, 같은 소스, 코호트 median split 이진화, 2클래스)

TCGA는 pni/immune 라벨이 없다(GDC/cBioPortal/원논문 세 군데 확인 완료, 2026-09-04) — stage만
두 코호트 다 있다. 라벨 없는 필드/환자는 -1(기존 STAGE_FIELDS/mutation 규약과 동일)로 표시해
loss에서 그 항만 제외한다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_T_CLASSES = 5  # Tis/T1/T2/T3/T4


class ClinicalAuxClassifier(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.stage_head = nn.Linear(hidden_dim, N_T_CLASSES)
        self.pni_head = nn.Linear(hidden_dim, 2)
        self.immune_head = nn.Linear(hidden_dim, 2)

    def forward(self, wsi_meanpool_embed: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(wsi_meanpool_embed)
        return {"stage": self.stage_head(h), "pni": self.pni_head(h), "immune": self.immune_head(h)}

    def loss(self, wsi_meanpool_embed: torch.Tensor, stage_ord: torch.Tensor,
              pni_ord: torch.Tensor, immune_ord: torch.Tensor) -> torch.Tensor | None:
        """
        Args:
            wsi_meanpool_embed: (D,)
            stage_ord, pni_ord, immune_ord: () 스칼라 long 텐서, "미상"은 -1.
        Returns:
            사용 가능한 태스크들의 평균 cross-entropy, 전부 미상이면 None.
        """
        logits = self.forward(wsi_meanpool_embed)
        losses = []
        if stage_ord.item() >= 0:
            losses.append(F.cross_entropy(logits["stage"].unsqueeze(0), stage_ord.unsqueeze(0)))
        if pni_ord.item() >= 0:
            losses.append(F.cross_entropy(logits["pni"].unsqueeze(0), pni_ord.unsqueeze(0)))
        if immune_ord.item() >= 0:
            losses.append(F.cross_entropy(logits["immune"].unsqueeze(0), immune_ord.unsqueeze(0)))
        if not losses:
            return None
        return torch.stack(losses).mean()
