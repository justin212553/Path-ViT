"""
HDP_Pretrain 전용 — frozen UNI2-h feature(1536차원) -> patch 단위 종양 함량 스칼라(0~1)
회귀 head. scripts/train_hdp_pretrain_head.py에서 PanNuke pancreas subset(핵 단위 라벨,
neoplastic area fraction)으로 학습되고, scripts/apply_hdp_pretrain_head.py가 이 학습된
가중치를 우리 코호트(TCGA-PAAD/CPTAC-PDA)의 이미 추출된 uni2native feature에 그대로
적용한다(원본 이미지 재처리 없음).
"""
import torch.nn as nn


class TumorContentHead(nn.Module):
    def __init__(self, in_dim: int = 1536, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 1), nn.Sigmoid(),
            )
        else:
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x).squeeze(-1)
