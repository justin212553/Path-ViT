"""
ViT Encoder with 2D Spatial Position Embedding
- 림프절 내부 해부학적 구조 보존을 위한 2D 공간적 위치 인코딩
- Nystrom 근사 self-attention으로 Global Spatial Context 연산 (O(N) 복잡도, 대규모 패치 수 대응)
- 출력: 공간 문맥이 반영된 패치 토큰 (N, D)
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from nystrom_attention import NystromAttention

from models.spatial_residual import RelativePositionBias, knn_edges


class SpatialPositionEmbedding(nn.Module):
    """
    WSI 패치의 (row, col) 좌표를 sinusoidal 인코딩으로 변환.

    학습 파라미터 없음 — 좌표값 자체를 결정론적으로 인코딩.
    embed_dim은 4의 배수여야 한다 (row·col 각각 sin/cos 절반씩 할당).
    """

    def __init__(self, embed_dim: int = 512, temperature: float = 10000.0):
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
        self.embed_dim  = embed_dim
        self.temperature = temperature

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N_patches, 2) int64 — 각 패치의 (row, col) 그리드 좌표
        Returns:
            pos_embed: (N_patches, embed_dim)
        """
        quarter = self.embed_dim // 4
        freq = self.temperature ** (
            torch.arange(quarter, device=coords.device, dtype=torch.float32) / quarter
        )                                               # (D/4,)  주파수 분모
        row = coords[:, 0:1].float() / freq             # (N, D/4)
        col = coords[:, 1:2].float() / freq             # (N, D/4)
        return torch.cat(
            [row.sin(), row.cos(), col.sin(), col.cos()], dim=-1
        )                                               # (N, D)


class _FullSelfAttention(nn.Module):
    """NystromAttention과 같은 (x)->x 시그니처의 표준(O(N^2)) self-attention 래퍼.

    2026-07-22: 슬라이드당 패치 수가 실측상 num_landmarks(128)보다 작은 경우가 절반 이상이라
    (findings_backlog.md), Nystrom 근사가 오히려 패딩 토큰을 landmark에 섞어 넣는 역효과를
    냈을 수 있다는 의심으로 추가한 대안 — 이 프로젝트 규모(N<=544)에서는 O(N^2)이 GPU에
    전혀 부담이 없어(544^2≈30만) 근사 없이 정확한 attention을 그대로 쓸 수 있다.
    """

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.mha(x, x, x, need_weights=False)
        return out


class RelativeBiasFullAttention(nn.Module):
    """전체(O(N^2)) self-attention + 상대offset(Δrow, Δcol) bias(Swin류, models/spatial_residual.py
    ::RelativePositionBias 재사용) — 절대좌표 sin/cos 임베딩(SpatialPositionEmbedding)을 patch_tokens에
    더하는 대신, attention logit 자체에 "두 패치가 얼마나/어느 방향으로 떨어져 있는가"를 직접 주입한다.

    2026-07-23: 얼린 Stage1 뒤에 별도 잔차 branch로 공간 정보를 붙이는 ResTopoMIL류 설계
    (models/spatial_residual.py)가 PAAD·BRCA 둘 다에서 실패한 뒤 시도하는 대안 — late-fusion
    잔차가 아니라 WSI branch 안에서 RNA/Clinical과 함께 end-to-end로 처음부터 학습시킨다.
    상대offset이라 절대좌표 임베딩과 달리 슬라이드 크기/위치와 무관하게 같은 의미를 갖는다.
    """

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        assert dim % heads == 0, "dim은 heads로 나누어떨어져야 합니다"
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.rel_bias = RelativePositionBias(num_heads=heads)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """x: (1, N, D), coords: (N, 2) -> (1, N, D)"""
        n = x.shape[1]
        q = self.q_proj(x).view(n, self.heads, self.head_dim)
        k = self.k_proj(x).view(n, self.heads, self.head_dim)
        v = self.v_proj(x).view(n, self.heads, self.head_dim)

        rel = (coords.unsqueeze(1) - coords.unsqueeze(0)).float()          # (N, N, 2) j-i offset
        bias = self.rel_bias(rel.view(-1, 2)).view(n, n, self.heads)       # (N, N, H)

        logits = torch.einsum("qhd,khd->qkh", q, k) / (self.head_dim ** 0.5) + bias  # (N, N, H)
        attn = logits.softmax(dim=1)                                       # key(dim=1) 기준 정규화
        attn = self.attn_drop(attn)

        out = torch.einsum("qkh,khd->qhd", attn, v).reshape(n, -1)
        out = self.out_drop(self.out_proj(out))
        return out.unsqueeze(0)                                            # (1, N, D)


class KNNBiasAttention(nn.Module):
    """RelativeBiasFullAttention의 희소(sparse) 버전 — 모든 패치 쌍(O(N^2)) 대신 각 패치의
    kNN 이웃 k개에만 attention한다(엣지 수 E=N*k, N에 선형). models/spatial_residual.py::
    knn_edges/RelativePositionBias 재사용(scatter-softmax 방식은 KNNGraphAttentionLayer와 동일).

    2026-07-23: BRCA(슬라이드당 패치 수 중앙값 10,309, 최대 67,268 — data/brca_slide_manifest.csv)
    로 RelativeBiasFullAttention을 그대로 돌리자 즉시 CUDA OOM(중앙값 규모 슬라이드에서 attention
    logit/bias 텐서가 N^2으로 터짐, findings_backlog.md) — PAAD(N<=544)에서만 성립하던 "O(N^2)도
    부담 없다"는 전제가 BRCA에는 아예 적용되지 않는다. kNN 그래프 구성(torch.cdist, O(N^2)이지만
    hidden_dim 없이 거리값 하나뿐이라 가볍다) 자체는 이미 spatial_residual.py의 잔차 branch 실험에서
    BRCA 규모로 문제없이 검증됨 — attention/bias 계산만 엣지 단위(E)로 제한하면 된다.
    """

    def __init__(self, dim: int, heads: int, dropout: float, k: int = 8, edge_dropout: float = 0.2):
        super().__init__()
        assert dim % heads == 0, "dim은 heads로 나누어떨어져야 합니다"
        self.heads = heads
        self.head_dim = dim // heads
        self.k = k
        self.edge_dropout = edge_dropout
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.rel_bias = RelativePositionBias(num_heads=heads)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """x: (1, N, D), coords: (N, 2) -> (1, N, D)"""
        n = x.shape[1]
        x0 = x[0]                                        # (N, D)
        edge_index = knn_edges(coords, self.k)
        if edge_index.shape[1] == 0:                      # 패치 1개뿐 — 이웃 없음
            return x
        if self.training and self.edge_dropout > 0:
            keep = torch.rand(edge_index.shape[1], device=x.device) >= self.edge_dropout
            if keep.any():
                edge_index = edge_index[:, keep]
        src, dst = edge_index[0], edge_index[1]            # src=쿼리(i), dst=이웃(j)

        q = self.q_proj(x0).view(n, self.heads, self.head_dim)
        k = self.k_proj(x0).view(n, self.heads, self.head_dim)
        v = self.v_proj(x0).view(n, self.heads, self.head_dim)

        rel = (coords[dst] - coords[src]).float()          # (E, 2)
        bias = self.rel_bias(rel)                           # (E, H)

        logits = (q[src] * k[dst]).sum(-1) / (self.head_dim ** 0.5) + bias  # (E, H)
        logits = logits - logits.max()
        exp = logits.exp()

        denom = torch.zeros(n, self.heads, device=x.device, dtype=exp.dtype)
        denom.index_add_(0, src, exp)
        attn = exp / (denom[src] + 1e-8)                    # (E, H) — src(쿼리)별 softmax
        attn = self.attn_drop(attn)

        weighted_v = attn.unsqueeze(-1) * v[dst]            # (E, H, Dh)
        out = torch.zeros(n, self.heads, self.head_dim, device=x.device, dtype=x.dtype)
        out.index_add_(0, src, weighted_v.to(x.dtype))
        out = out.reshape(n, -1)
        out = self.out_drop(self.out_proj(out))
        return out.unsqueeze(0)                             # (1, N, D)


class KNNMeanAggregation(nn.Module):
    """KNNBiasAttention/KNNFixedBiasAttention과 같은 kNN 그래프를 쓰지만, attention(softmax
    가중치, q/k projection)을 아예 없애고 이웃 k개를 단순 평균(GraphSAGE mean-aggregator 스타일)
    해서 자기 자신과 concat 후 선형변환한다 — "학습되는 attention 파라미터 자체가 작은 코호트
    에서 과적합 유인"이라는 KNNFixedBiasAttention의 문제의식을 극단까지 밀어붙인 버전. bias조차
    없고 순수 평균이라 학습 파라미터가 결합 MLP 하나뿐이다.

    2026-09-05: "self-attention 말고 패치끼리 정보를 주고받는 다른 방법"(사용자 질문) 중 하나 —
    RelativeBiasFullAttention이 A30(24GB)에서도 OOM났던 것과 달리, 이웃 집계는 attention
    logit/softmax 없이 인덱스 합산(index_add_)만 쓰므로 메모리가 O(E)(엣지 수)로 훨씬 가볍다.
    """

    def __init__(self, dim: int, k: int = 8, edge_dropout: float = 0.2, dropout: float = 0.0):
        super().__init__()
        self.k = k
        self.edge_dropout = edge_dropout
        self.combine = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """x: (1, N, D), coords: (N, 2) -> (1, N, D)"""
        n = x.shape[1]
        x0 = x[0]                                        # (N, D)
        edge_index = knn_edges(coords, self.k)
        if edge_index.shape[1] == 0:                      # 패치 1개뿐 — 이웃 없음
            return x
        if self.training and self.edge_dropout > 0:
            keep = torch.rand(edge_index.shape[1], device=x.device) >= self.edge_dropout
            if keep.any():
                edge_index = edge_index[:, keep]
        src, dst = edge_index[0], edge_index[1]            # src=쿼리(i), dst=이웃(j)

        neighbor_sum = torch.zeros_like(x0)
        neighbor_sum.index_add_(0, src, x0[dst])
        neighbor_count = torch.zeros(n, device=x.device, dtype=x0.dtype)
        neighbor_count.index_add_(0, src, torch.ones(edge_index.shape[1], device=x.device, dtype=x0.dtype))
        neighbor_mean = neighbor_sum / neighbor_count.clamp_min(1.0).unsqueeze(-1)  # (N, D)

        out = self.combine(torch.cat([x0, neighbor_mean], dim=-1))  # (N, D)
        return out.unsqueeze(0)                             # (1, N, D)


class HierarchicalClusterAttention(nn.Module):
    """패치(N개, 수천~수만일 수 있음)를 K개(기본 16) 학습 가능한 클러스터 프로토타입에 먼저
    소프트 배정해 슈퍼토큰으로 압축한 뒤(N -> K), 그 K개끼리만 dense full self-attention을
    돌리고(K가 작아 O(K^2)이 사실상 공짜), 같은 배정 가중치로 그 결과를 다시 각 패치에
    broadcast한다 — Slot Attention/Perceiver의 병목(bottleneck) attention과 같은 발상.
    RelativeBiasFullAttention(dense, 전체 패치끼리 O(N^2))이 메모리를 못 감당하는 문제를,
    "패치 수천 개가 서로 직접 다 보는" 대신 "미리 K개 대표로 압축한 뒤 그 대표들끼리만 다 보고,
    그 결과를 다시 나눠준다"로 우회한다 — 원래 의도("패치들이 서로 정보를 교환해 거시적 구조를
    합성")를 압축된 병목에서 수행하는 셈.

    사전학습된 클러스터(예: data/cluster_centroids_uni2native.pt, K=10, raw 1536차원)는 재사용
    하지 않는다 — 그 centroid는 이 레이어가 보는 embed_dim(64) 투영 공간과 차원이 달라 그대로
    못 쓰고, 애초에 생존 예측과 무관한 기준(단순 K-means)으로 뽑힌 군집이라 이 태스크에 맞는
    군집을 처음부터 end-to-end로 학습시키는 쪽이 더 원칙적이다.

    2026-09-05: "self-attention 말고 패치끼리 정보를 주고받는 다른 방법"(사용자 질문) 중 하나.
    """

    def __init__(self, dim: int, heads: int, dropout: float, n_clusters: int = 16):
        super().__init__()
        self.n_clusters = n_clusters
        self.prototypes = nn.Parameter(torch.randn(n_clusters, dim) * 0.02)
        self.cluster_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1, N, D) -> (1, N, D)"""
        x0 = x[0]                                                          # (N, D)
        sim = x0 @ self.prototypes.T / (x0.shape[-1] ** 0.5)               # (N, K)
        assign = sim.softmax(dim=-1)                                       # (N, K) — 패치별 소프트 배정
        cluster_weight_sum = assign.sum(dim=0).clamp_min(1e-6)             # (K,)
        super_tokens = (assign.T @ x0) / cluster_weight_sum.unsqueeze(-1)  # (K, D) — 패치 -> 슈퍼토큰
        mixed, _ = self.cluster_attn(
            super_tokens.unsqueeze(0), super_tokens.unsqueeze(0), super_tokens.unsqueeze(0), need_weights=False
        )                                                                  # (1, K, D) — 슈퍼토큰끼리 full attn
        out = assign @ mixed[0]                                            # (N, D) — 슈퍼토큰 -> 패치로 broadcast
        return self.out_proj(out).unsqueeze(0)                            # (1, N, D)


class HybridLocalGlobalAttention(nn.Module):
    """local(KNNBiasAttention) + global(NystromAttention)를 같은 레이어에서 계산해 더한다.

    2026-07-23: PAAD(pre-augment)에서 kNN-bias-attention 단독은 기존 Nystrom+절대좌표 baseline보다
    낮았다(internal 0.6309->0.6094, external 0.6289->0.5880) — "국소만"으로 대체하는 대신, 기존
    Nystrom(전역, landmark 근사)은 그대로 두고 kNN(국소, 상대offset bias)을 같은 레이어에 병렬로
    얹어 더하는 쪽을 시도한다("Combining GNN and Mamba..." 논문의 local+global 병렬 조합과 같은 발상).
    global 쪽(Nystrom)은 좌표 정보를 직접 안 받으므로, 이 모드에서는 절대좌표 임베딩
    (SpatialPositionEmbedding, use_spatial_embed)을 계속 켜둔 채로 쓰는 게 자연스럽다.
    """

    def __init__(self, dim: int, heads: int, dropout: float, num_landmarks: int, k: int = 8, edge_dropout: float = 0.2):
        super().__init__()
        self.local = KNNBiasAttention(dim, heads, dropout, k=k, edge_dropout=edge_dropout)
        self.global_attn = NystromAttention(
            dim=dim, dim_head=dim // heads, heads=heads, num_landmarks=num_landmarks,
            pinv_iterations=6, residual=True, dropout=dropout,
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return self.local(x, coords) + self.global_attn(x)


class KNNFixedBiasAttention(nn.Module):
    """KNNBiasAttention과 동일하나, 학습되는 RelativePositionBias(MLP) 대신 거리 감쇠 커널을 쓴다
    — bias = -||Δcoords|| / tau. learnable_tau=False(기본)면 tau는 학습되지 않는 스칼라
    하이퍼파라미터(idea #3), learnable_tau=True면 head별로 학습되는 스칼라 1개씩(PSA-MIL
    "learnable distance-decayed prior"의 경량 버전 — posterior∝likelihood×prior를 log공간에서
    풀면 logits=QK+log(prior)가 되는데, 이미 이 클래스의 "logits+bias" 구조가 정확히 그 형태다.
    학습 파라미터가 head 개수(보통 2)뿐이라 RelativePositionBias MLP(82개)보다 훨씬 작다).

    2026-07-23: kNN/hybrid attention이 PAAD(pre-augment)에서 baseline을 못 넘은 뒤, "새로 학습되는
    attention 파라미터 자체가 작은 코호트에서 과적합 유인"이라는 가설을 검증하는 3번째 대안 —
    q/k/v/out projection(패치 내용 기반 attention에 필요)은 그대로 두되, 공간 정보를 주입하는
    bias 항목의 학습량만 조절한다(0개=고정 tau, head개=learnable_tau, 82개=RelativePositionBias MLP).
    """

    def __init__(
        self, dim: int, heads: int, dropout: float, k: int = 8, edge_dropout: float = 0.2,
        tau: float = 50.0, learnable_tau: bool = False,
    ):
        super().__init__()
        assert dim % heads == 0, "dim은 heads로 나누어떨어져야 합니다"
        self.heads = heads
        self.head_dim = dim // heads
        self.k = k
        self.edge_dropout = edge_dropout
        self.learnable_tau = learnable_tau
        if learnable_tau:
            # log_tau로 파라미터화 — tau=exp(log_tau)>0 보장(감쇠 스케일이 음수/0이 되면 안 됨).
            self.log_tau = nn.Parameter(torch.full((heads,), float(torch.log(torch.tensor(tau)))))
        else:
            self.register_buffer("tau_fixed", torch.tensor(tau), persistent=False)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """x: (1, N, D), coords: (N, 2) -> (1, N, D)"""
        n = x.shape[1]
        x0 = x[0]                                        # (N, D)
        edge_index = knn_edges(coords, self.k)
        if edge_index.shape[1] == 0:                      # 패치 1개뿐 — 이웃 없음
            return x
        if self.training and self.edge_dropout > 0:
            keep = torch.rand(edge_index.shape[1], device=x.device) >= self.edge_dropout
            if keep.any():
                edge_index = edge_index[:, keep]
        src, dst = edge_index[0], edge_index[1]            # src=쿼리(i), dst=이웃(j)

        q = self.q_proj(x0).view(n, self.heads, self.head_dim)
        k = self.k_proj(x0).view(n, self.heads, self.head_dim)
        v = self.v_proj(x0).view(n, self.heads, self.head_dim)

        rel = (coords[dst] - coords[src]).float()          # (E, 2)
        dist = rel.norm(dim=-1)                             # (E,)
        if self.learnable_tau:
            tau = self.log_tau.exp()                        # (H,) — head별 학습되는 감쇠 스케일
            bias = -dist.unsqueeze(-1) / tau.unsqueeze(0)    # (E, H)
        else:
            bias = (-dist / self.tau_fixed).unsqueeze(-1).expand(-1, self.heads)  # (E, H)

        logits = (q[src] * k[dst]).sum(-1) / (self.head_dim ** 0.5) + bias  # (E, H)
        logits = logits - logits.max()
        exp = logits.exp()

        denom = torch.zeros(n, self.heads, device=x.device, dtype=exp.dtype)
        denom.index_add_(0, src, exp)
        attn = exp / (denom[src] + 1e-8)                    # (E, H) — src(쿼리)별 softmax
        attn = self.attn_drop(attn)

        weighted_v = attn.unsqueeze(-1) * v[dst]            # (E, H, Dh)
        out = torch.zeros(n, self.heads, self.head_dim, device=x.device, dtype=x.dtype)
        out.index_add_(0, src, weighted_v.to(x.dtype))
        out = out.reshape(n, -1)
        out = self.out_drop(self.out_proj(out))
        return out.unsqueeze(0)                             # (1, N, D)


class NystromEncoderLayer(nn.Module):
    """
    Pre-LN Transformer 블록, self-attention을 Nystrom 근사(기본) 또는 표준 O(N^2)
    attention(use_nystrom=False)으로 구성한다. 전체 softmax attention(O(N^2)) 대신 landmark
    기반 근사(O(N))를 쓰면 패치 수가 매우 큰 WSI(수만 패치)에서도 attention 연산이 가능해지지만,
    이 프로젝트의 실제 패치 수 규모(수십~수백)에서는 O(N^2)도 부담 없어 use_nystrom=False로
    근사 없는 정확한 attention을 시험해볼 수 있다.

    [use_ffn=False, --M4A_FF 맛보기 ablation] FFN 서브레이어를 통째로 제거한다.
    attention은 패치 간 정보를 "섞는" 역할, FFN은 그렇게 섞인 결과를 패치 하나 단위로
    비선형 변환하는 역할로 서로 독립적이다(FFN은 다른 패치를 전혀 참조하지 않음) — 그래서
    FFN을 빼도 공간 컨텍스트(attention이 만든) 자체는 그대로 남고, "그 결과를 더 다듬는"
    단계만 없어진다.

    [context_dim, --M2_FF 맛보기 ablation] FFN 입력 직전에 외부 컨텍스트(z_rna)를 가산
    bias로 더한다(vit_m1.py::AttentionPooling의 FiLM 방식과 동일한 관례) — "attention이
    이미 공간 문맥을 반영한 패치 표현"을 RNA 기준으로 한 번 더 조건화한 뒤 FFN으로 다듬는
    지점. use_ffn=True일 때만 의미가 있다(FFN이 없으면 조건화할 대상이 없음).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        num_landmarks: int,
        use_ffn: bool = True,
        context_dim: int | None = None,
        use_nystrom: bool = True,
        use_rel_bias_attn: bool = False,
        use_knn_bias_attn: bool = False,
        use_hybrid_attn: bool = False,
        use_knn_fixed_bias_attn: bool = False,
        use_knn_mean_agg: bool = False,
        use_cluster_attn: bool = False,
        n_clusters: int = 16,
        fix_nystrom_landmarks: bool = False,
        knn_k: int = 8,
        knn_edge_dropout: float = 0.2,
        knn_bias_tau: float = 50.0,
        knn_bias_learnable_tau: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        # use_rel_bias_attn/use_knn_bias_attn/use_hybrid_attn/use_knn_fixed_bias_attn/
        # use_knn_mean_agg/use_cluster_attn이 use_nystrom보다 우선한다(모델/ViT_M1.__init__이
        # 배타적 조합을 강제) — dense(전체 O(N^2), PAAD처럼 패치 수가 작을 때), sparse(kNN,
        # BRCA처럼 패치 수가 많아 dense가 OOM나는 경우), hybrid(kNN 국소 + Nystrom 전역을
        # 병렬로 더함), knn_fixed_bias(kNN이되 relative bias가 학습 없는 고정 거리 커널),
        # knn_mean_agg(kNN이되 attention 자체를 없애고 단순 평균), cluster_attn(패치를 K개
        # 학습형 슈퍼토큰으로 압축해 그 안에서만 dense attention) 중 하나.
        self._fix_nystrom_landmarks = fix_nystrom_landmarks
        self._configured_num_landmarks = num_landmarks
        if use_rel_bias_attn:
            self.attn = RelativeBiasFullAttention(embed_dim, num_heads, dropout)
        elif use_hybrid_attn:
            self.attn = HybridLocalGlobalAttention(
                embed_dim, num_heads, dropout, num_landmarks, k=knn_k, edge_dropout=knn_edge_dropout,
            )
        elif use_knn_fixed_bias_attn:
            self.attn = KNNFixedBiasAttention(
                embed_dim, num_heads, dropout, k=knn_k, edge_dropout=knn_edge_dropout, tau=knn_bias_tau,
                learnable_tau=knn_bias_learnable_tau,
            )
        elif use_knn_bias_attn:
            self.attn = KNNBiasAttention(embed_dim, num_heads, dropout, k=knn_k, edge_dropout=knn_edge_dropout)
        elif use_knn_mean_agg:
            self.attn = KNNMeanAggregation(embed_dim, k=knn_k, edge_dropout=knn_edge_dropout, dropout=dropout)
        elif use_cluster_attn:
            self.attn = HierarchicalClusterAttention(embed_dim, num_heads, dropout, n_clusters=n_clusters)
        elif use_nystrom:
            self.attn = NystromAttention(
                dim=embed_dim,
                dim_head=embed_dim // num_heads,
                heads=num_heads,
                num_landmarks=num_landmarks,
                pinv_iterations=6,
                residual=True,
                dropout=dropout,
            )
        else:
            self.attn = _FullSelfAttention(embed_dim, num_heads, dropout)
        self.needs_coords = (use_rel_bias_attn or use_knn_bias_attn or use_hybrid_attn
                              or use_knn_fixed_bias_attn or use_knn_mean_agg)
        self.dropout1 = nn.Dropout(dropout)

        self.use_ffn = use_ffn
        self.context_proj: nn.Linear | None = None
        if use_ffn:
            self.norm2 = nn.LayerNorm(embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, embed_dim),
                nn.Dropout(dropout),
            )
            if context_dim is not None:
                self.context_proj = nn.Linear(context_dim, embed_dim, bias=False)

    def forward(
        self, x: torch.Tensor, coords: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._fix_nystrom_landmarks and isinstance(self.attn, NystromAttention):
            # 2026-09-05: findings_backlog.md에서 이미 지적된 버그 — nystrom_attention 라이브러리는
            # n<num_landmarks면 F.pad로 (num_landmarks-n)개 zero 토큰을 채워 넣어(nystrom_attention.py
            # forward의 `remainder = n % m` 분기), 슬라이드 절반 이상(패치 수 중앙값 67 < landmark
            # 128)에서 landmark의 상당수가 진짜 패치가 아니라 0벡터가 된다. self.attn.num_landmarks는
            # forward마다 다시 읽히는 평범한 int라, 매 호출 직전에 실제 패치 수로 clamp하면 라이브러리
            # 코드를 건드리지 않고도 이 특정 호출에서만 안전하게 고칠 수 있다.
            self.attn.num_landmarks = min(self._configured_num_landmarks, x.shape[1])
        if self.needs_coords:
            attn_out = self.attn(self.norm1(x), coords)
        else:
            attn_out = self.attn(self.norm1(x))
        x = x + self.dropout1(attn_out)
        if self.use_ffn:
            h = self.norm2(x)
            if self.context_proj is not None and context is not None:
                h = h + self.context_proj(context)  # (1, N, D) + (D,) broadcast
            x = x + self.ffn(h)
        return x


class ViTEncoder(nn.Module):
    """
    패치 토큰 + 공간 임베딩 → Nystrom Transformer Encoder → 문맥화된 패치 토큰
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_grad_checkpoint: bool = True,
        num_landmarks: int = 128,
        use_ffn: bool = True,
        context_dim: int | None = None,
        use_nystrom: bool = True,
        use_spatial_embed: bool = True,
        use_rel_bias_attn: bool = False,
        use_knn_bias_attn: bool = False,
        use_hybrid_attn: bool = False,
        use_knn_fixed_bias_attn: bool = False,
        use_knn_mean_agg: bool = False,
        use_cluster_attn: bool = False,
        n_clusters: int = 16,
        fix_nystrom_landmarks: bool = False,
        knn_k: int = 8,
        knn_edge_dropout: float = 0.2,
        knn_bias_tau: float = 50.0,
        knn_bias_learnable_tau: bool = False,
        use_tumor_type_embed: bool = False,
    ):
        super().__init__()
        self.pos_embedding = SpatialPositionEmbedding(embed_dim)
        self.use_grad_checkpoint = use_grad_checkpoint
        # 2026-08-13: --tumor-type-embed — pos_embedding과 완전히 같은 패턴(ViT 입력 직전
        # patch_tokens에 가산)으로, 슬라이드가 종양/정상/미상(data/dataset.py::_slide_tumor_status)
        # 인지를 학습 가능한 임베딩으로 조건화한다. pos_embedding은 결정론적(좌표의 sinusoidal
        # 함수)인 반면 이건 룩업 테이블이라 학습된다는 점만 다르다.
        self.use_tumor_type_embed = use_tumor_type_embed
        if use_tumor_type_embed:
            self.tumor_type_embed = nn.Embedding(3, embed_dim)  # 0=tumor,1=normal,2=unknown
        # 2026-07-22: attention이 이미 uniform으로 붕괴해 있고(diagnose_wsi_reliance.py) 나이스트롬
        # 근사가 landmark를 뭉뚱그리는 상황에서, 좌표 임베딩이 실제로 최종 예측에 기여하는지
        # 직접 확인하는 ablation(findings_backlog.md, --no-spatial-embed) — False면
        # pos_embedding을 아예 계산에서 제외한다(patch_tokens를 그대로 사용).
        self.use_spatial_embed = use_spatial_embed

        self.layers = nn.ModuleList([
            NystromEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                num_landmarks=num_landmarks,
                use_ffn=use_ffn,
                context_dim=context_dim,
                use_nystrom=use_nystrom,
                use_rel_bias_attn=use_rel_bias_attn,
                use_knn_bias_attn=use_knn_bias_attn,
                use_hybrid_attn=use_hybrid_attn,
                use_knn_fixed_bias_attn=use_knn_fixed_bias_attn,
                use_knn_mean_agg=use_knn_mean_agg,
                use_cluster_attn=use_cluster_attn,
                n_clusters=n_clusters,
                fix_nystrom_landmarks=fix_nystrom_landmarks,
                knn_bias_learnable_tau=knn_bias_learnable_tau,
                knn_k=knn_k,
                knn_edge_dropout=knn_edge_dropout,
                knn_bias_tau=knn_bias_tau,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, patch_tokens: torch.Tensor, coords: torch.Tensor, context: torch.Tensor | None = None,
        tumor_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            patch_tokens: (N_patches, embed_dim) - CNN에서 추출한 패치 특징
            coords:       (N_patches, 2)          - 각 패치의 (row, col) 좌표
            context:      (embed_dim,) 선택 — 각 레이어의 FFN 직전에 가산 bias로 주입할
                          외부 컨텍스트(z_rna). --M2_FF 맛보기 ablation 전용, 기본 None이면
                          기존 모델들과 완전히 동일하게 동작한다.
            tumor_type:   () 스칼라 텐서 선택 — 이 슬라이드의 종양/정상/미상 라벨(0/1/2,
                          data/dataset.py::_slide_tumor_status). use_tumor_type_embed=True일
                          때만 쓰이며, 슬라이드 전체 패치에 동일한 임베딩을 더한다.
        Returns:
            out_tokens: (N_patches, embed_dim) - 공간 문맥이 반영된 패치 토큰
        """
        if self.use_spatial_embed:
            pos = self.pos_embedding(coords)       # (N, D)
            x = patch_tokens + pos
        else:
            x = patch_tokens
        if self.use_tumor_type_embed and tumor_type is not None:
            x = x + self.tumor_type_embed(tumor_type)  # (D,) -> 전체 패치에 broadcast
        x = x.unsqueeze(0)                          # (1, N, D)

        use_ckpt = self.use_grad_checkpoint and self.training
        for layer in self.layers:
            if use_ckpt:
                x = cp.checkpoint(layer, x, coords, context, use_reentrant=False)
            else:
                x = layer(x, coords, context)

        return self.norm(x[0])                     # (N, D)
