"""
RNAPredictionHead - WSI 표현에서 RNA 발현을 예측하는 보조과제(auxiliary task) 헤드.
train.py --rna-aux-weight.

[배경] 지금까지 시도한 전부(M4/M4A/M4B/PM4/PMA/M2_FF)는 "추론 시점에 WSI와 RNA를 어떻게
결합할까"였다 - 두 표현이 이미 각자 어느 정도 완성돼 있다고 가정하고 결합 방식만 바꾼
ablation들. 근데 실제 병목은 WSI 브랜치가 생존이라는 극도로 약한 신호(환자당 라벨 1개,
censoring으로 더 약해짐)만으로 62만 파라미터를 학습하다 과적합하는 것이었다(model_zoo.md,
findings_backlog.md 참조). 이 헤드는 결합 방식이 아니라 "학습 신호의 빈곤함" 자체를
겨냥한다 - RNA 발현(환자당 1500차원)을 보조 라벨로 써서 WSI 인코더가 형태학적으로
RNA와 상관된 특징을 뽑도록 정규화한다(HE2RNA, Schmauch et al. 2020과 같은 방향).

[RNA-누수 방지] WSI 표현이 이미 RNA로 조건화된 뒤(M4A의 co-attention pooling, PMA의
component co-attention 등)에 그 RNA를 다시 예측하게 하면 순환 논리라 의미가 없다.
그래서 이 헤드는 각 모델의 attn_pool(RNA가 개입하는 지점)이 아니라, ViT를 지난 직후의
patch 토큰을 RNA와 무관하게 mean pooling한 별도 표현("meanpool_embed", 각 forward()가
반환)에 붙는다 - 어떤 pooling/fusion 방식을 쓰든 항상 RNA-free하다.
"""
import torch.nn as nn


class RNAPredictionHead(nn.Module):
    def __init__(self, embed_dim: int, rna_output_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, rna_output_dim),
        )

    def forward(self, wsi_meanpool_embed):
        """
        Args:
            wsi_meanpool_embed: (D,) - RNA-free mean-pooled WSI 표현(환자 단위)
        Returns:
            rna_pred: (G,) - 예측된 z-score 정규화 유전자 발현
        """
        return self.mlp(wsi_meanpool_embed)
