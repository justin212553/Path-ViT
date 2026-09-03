"""
HDP_Cluster — models/hdp.py::HDP(비지도 k-means 군집의 40차원 결정론적 통계)에, 2026-09-01
초반에 리스크가 크다는 이유로 뺐던 두 컴포넌트를 다시 넣은 버전(train_hdp_cluster.py 전용).
사용자 결정 — 안전한 부분(HDP)만으로는 M7을 못 넘었으니, 152개 생존 라벨로 새 컴포넌트를
학습시키는 리스크를 감수하고서라도 침윤전선/성숙도까지 마저 시도해본다.

  - GrowthPatternCNN(침윤전선/성장 패턴, 원래 계획 3-2): patch를 (K, H, W) 공간 맵으로
    재조합(K=군집 soft weight 채널, H/W=슬라이드 grid)한 뒤 작은 CNN + global average pool로
    고정 차원 벡터를 뽑는다. "이 군집이 슬라이드 전체에 어떤 형태로 분포하는가"를 거시적으로
    본다.
  - MaturityMLP(성숙도, 원래 계획 3-4): patch feature(1536)+soft 군집 벡터(K)를 이어붙여
    작은 MLP로 스칼라를 내고, mean pooling(학습되는 attention 아님 — 이 세션 내내 attention/
    MIL이 반복 실패한 패턴을 다시 밟지 않기 위한 의도적 선택)으로 환자 단위 스칼라를 만든다.

둘 다 152개 생존 라벨로 end-to-end 학습된다(CNN/MLP 파라미터는 다른 pretext task로 미리
학습된 게 아님) — HDP(안전한 4*K차원 결정론적 통계, 학습 파라미터 0개)와는 리스크 범주가
다르다는 걸 명심할 것.
"""
import torch
import torch.nn as nn

from .hdp import HDP


class GrowthPatternCNN(nn.Module):
    def __init__(self, k: int, out_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(k, 16, kernel_size=3, padding=1), nn.GroupNorm(4, 16), nn.ReLU(inplace=True),
            nn.Conv2d(16, hidden_dim, kernel_size=3, padding=1), nn.GroupNorm(4, hidden_dim), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, occupancy_map: torch.Tensor) -> torch.Tensor:
        """occupancy_map: (1, K, H, W) — 슬라이드 하나의 군집 soft-weight 공간 맵.
        Returns: (out_dim,)"""
        x = self.conv(occupancy_map).flatten(1)  # (1, hidden_dim)
        return self.proj(x).squeeze(0)  # (out_dim,)


class MaturityMLP(nn.Module):
    def __init__(self, feat_dim: int, k: int, hidden_dim: int = 64, out_dim: int = 1):
        """out_dim=1(기본, cox_add 관례) — 학습 후에도 스칼라 하나로 patch를 뭉갬.
        2026-09-02: out_dim>1(concat 모드)이면 patch마다 벡터를 내고 그 벡터들을 mean-pool해
        환자 단위 (out_dim,) "성숙도 임베딩"을 만든다 — RNA/growth처럼 조기 스칼라 압축 없이
        마지막 공유 risk_head까지 표현력을 유지한다."""
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim + k),
            nn.Linear(feat_dim + k, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, features: torch.Tensor, soft_weights: torch.Tensor) -> torch.Tensor:
        """features: (N, feat_dim), soft_weights: (N, K). Returns: (out_dim,)(mean pooling)."""
        x = torch.cat([features, soft_weights], dim=-1)  # (N, feat_dim+K)
        per_patch = self.net(x)  # (N, out_dim)
        return per_patch.mean(dim=0)  # (out_dim,)


class HDPCluster(HDP):
    def __init__(self, *args, k: int, feat_dim: int = 1536, growth_dim: int = 8,
                 maturity_embed_dim: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.growth_cnn = GrowthPatternCNN(k, out_dim=growth_dim)

        if self.combine_mode == "cox_add":
            self.maturity_mlp = MaturityMLP(feat_dim, k, out_dim=1)
            self.growth_linear = nn.Linear(growth_dim, 1, bias=False)
            self.maturity_linear = nn.Linear(1, 1, bias=False)
            nn.init.zeros_(self.growth_linear.weight)
            nn.init.zeros_(self.maturity_linear.weight)
        else:
            # 2026-09-02: concat 모드 — growth_cnn은 이미 (growth_dim,) 임베딩을 내므로 그대로
            # 쓰고, maturity_mlp도 스칼라 대신 (maturity_embed_dim,) 임베딩을 내게 바꾼다.
            # HDP.__init__(combine_mode="concat")가 만들어둔 self.risk_head_concat(rna+clinical+
            # hist용, base_fused_dim 기준)은 growth/maturity를 못 담으니 여기서 더 큰 걸로
            # 새로 만들어 덮어쓴다 — HDP가 standalone으로 쓰일 때(HDPCluster 없이)와 파라미터가
            # 안 섞이게, __init__ 순서상 이 시점 이후로는 self.risk_head_concat이 이 새 버전만
            # 유효하다(부모가 만든 작은 버전은 옵티마이저 생성 전에 교체되므로 학습에 안 걸림).
            self.maturity_mlp = MaturityMLP(feat_dim, k, out_dim=maturity_embed_dim)
            fused_dim = self.base_fused_dim + growth_dim + maturity_embed_dim
            self.risk_head_concat = nn.Sequential(
                nn.LayerNorm(fused_dim), nn.Linear(fused_dim, self.risk_hidden_dim), nn.GELU(),
                nn.Linear(self.risk_hidden_dim, 1),
            )

    def forward_wsi_extra(self, growth_vec: torch.Tensor, maturity_scalar: torch.Tensor) -> torch.Tensor:
        """cox_add 전용 — HDP.forward()의 합에 추가할 침윤전선+성숙도 risk 항.
        Args:
            growth_vec:      (growth_dim,) — train_hdp_cluster.py가 슬라이드별 GrowthPatternCNN
                              출력을 환자 단위로 평균해 넘긴다(멀티 슬라이드 대응, mean pooling).
            maturity_scalar: (1,) — 환자의 전체 patch(슬라이드 무관 풀링)에 대한 MaturityMLP 출력.
        Returns: risk (1,)
        """
        risk_growth = self.growth_linear(growth_vec.unsqueeze(0)).view(1)
        risk_maturity = self.maturity_linear(maturity_scalar.view(1, 1)).view(1)
        return risk_growth + risk_maturity

    def forward(self, age_years, sex_idx, rna, cluster_hist, growth_vec, maturity_scalar,
                margin_ord=None, stage_ord=None, return_components: bool = False):
        if self.combine_mode == "concat":
            if return_components:
                raise NotImplementedError("return_components는 combine_mode='cox_add'에서만 지원합니다.")
            base_fused = self._embed_base(age_years, sex_idx, rna, cluster_hist, margin_ord, stage_ord)
            fused = torch.cat([base_fused, growth_vec, maturity_scalar.view(-1)], dim=-1)
            return self.risk_head_concat(fused.unsqueeze(0)).view(1)

        if return_components:
            base = super().forward(age_years, sex_idx, rna, cluster_hist,
                                    margin_ord=margin_ord, stage_ord=stage_ord, return_components=True)
            risk_growth = self.growth_linear(growth_vec.unsqueeze(0)).view(1)
            risk_maturity = self.maturity_linear(maturity_scalar.view(1, 1)).view(1)
            return {**base, "growth": risk_growth, "maturity": risk_maturity}
        base_risk = super().forward(age_years, sex_idx, rna, cluster_hist,
                                     margin_ord=margin_ord, stage_ord=stage_ord)
        extra_risk = self.forward_wsi_extra(growth_vec, maturity_scalar)
        return base_risk + extra_risk
