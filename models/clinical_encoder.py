"""
ClinicalEncoder — 임상 정보(age, sex) MLP 인코더

data/clinical_{tcga,cptac}.csv 의 age_years, sex 컬럼을 WSI 임베딩과 late-fusion할 수 있는
D차원 벡터로 변환한다. patch_vit_fusion.py의 ClusterHistogramBranch(Path B)와 같은 역할 —
서로 다른 모달리티(연속형 age, 범주형 sex)를 하나의 임베딩으로 합쳐 risk_head 입력에
concat할 수 있게 한다.

[입력 전처리]
  age : (age_years - mean) / std 로 z-score 정규화. mean/std는 학습 코호트 내부에서 계산해
        buffer로 고정한다 — extract_rna_clinical.py가 RNA feature에 적용한 "데이터셋 내부
        z-score 정규화" 관례와 동일하게, age_stats_from_csv()로 학습 코호트 clinical.csv에서
        직접 계산해 생성자에 전달한다.
  sex : male=0, female=1 이진 인코딩. 두 코호트(clinical_tcga.csv, clinical_cptac.csv) 모두
        male/female만 존재함을 확인했으므로 그 외 값은 지원하지 않는다.
"""
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

SEX_TO_IDX = {"male": 0, "female": 1}


def age_stats_from_csv(csv_path: str | Path) -> tuple[float, float]:
    """clinical_{tcga,cptac}.csv의 age_years 평균/표준편차를 계산한다(z-score 정규화용)."""
    age_years = pd.read_csv(csv_path)["age_years"].astype(float)
    return float(age_years.mean()), float(age_years.std(ddof=0))


def encode_sex(sex: pd.Series | list[str]) -> torch.Tensor:
    """sex 컬럼(male/female 문자열)을 이진 인덱스(long) 텐서로 변환한다."""
    return torch.tensor([SEX_TO_IDX[s] for s in sex], dtype=torch.long)


class ClinicalEncoder(nn.Module):
    """
    age/sex (2,) → 임베딩 (D,) 두 층 MLP.

    [학습 범위]
    age_mean/age_std : 고정 — 학습 코호트에서 사전 계산된 정규화 통계
    mlp              : 학습 — age_z, sex_bin을 risk 예측에 유용한 임베딩으로 변환
    """

    def __init__(self, embed_dim: int, age_mean: float, age_std: float, hidden_dim: int = 64):
        super().__init__()
        self.register_buffer("age_mean", torch.tensor(age_mean, dtype=torch.float32))
        self.register_buffer("age_std", torch.tensor(age_std, dtype=torch.float32))

        # 입력 (age_z, sex_bin) (2,) → 임베딩 (D,): 두 층 MLP로 비선형 변환
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, age_years: torch.Tensor, sex_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            age_years : (N,) float — 원본 나이(연 단위)
            sex_idx   : (N,) long  — encode_sex()로 만든 0(male)/1(female) 인덱스
        Returns:
            z_clinical: (N, D) — 임상 정보 임베딩
        """
        age_z = (age_years.float() - self.age_mean) / self.age_std
        x = torch.stack([age_z, sex_idx.float()], dim=-1)  # (N, 2)
        return self.mlp(x)
