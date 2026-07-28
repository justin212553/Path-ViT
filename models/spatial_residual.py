"""
공간정보 잔차(residual) branch — kNN 그래프 기반. 기본은 attention 없는 단순 이웃평균
(SimpleGCNLayer, "simple GNN"), q/k/v attention+relative bias 버전(KNNGraphAttentionLayer)도
비교용으로 남겨뒀다(layer_type="attention").

[배경] SpatialPositionEmbedding(models/vit_encoder.py, 절대좌표 sin/cos)은 BRCA(WSI 신호가
실재하는 환경)에서도 null이었다(findings_backlog.md). "Spatial Blindness in Whole-Slide
Multiple Instance Learning"(arXiv:2605.17449)에 따르면 이건 아키텍처 문제가 아니라 학습
역학(optimization dynamics) 문제일 수 있다 — 약한 슬라이드 단위 지도(censored survival)
하에서는 밀도 높은 외형 통계(우리의 MultiComponentPooling)가 먼저 학습돼 공간적 관계를
배울 gradient가 거의 안 남는다. 이 논문의 ResTopoMIL 설계를 이식한다: 기존 모델(Stage1,
PMA_EX_SS_AUX_AUG)을 통째로 얼리고, 이 branch를 "잔차"로만 별도 학습시킨다
(scripts/train_spatial_residual.py). RNA/Clinical과는 섞이지 않는다 — 최종 결합은 스칼라
risk 값끼리 더하기(combined logits)뿐이라 임베딩 차원을 맞출 필요가 없다.

절대좌표 대신 두 패치의 *상대* offset(Δrow, Δcol)만 쓰므로, 슬라이드가 달라도 같은 의미를
가지고(일반화), --shuffle-patches/patch_keep_frac로 패치 배열 순서가 섞여도 매 forward마다
coords로 그래프를 새로 구성하니 전혀 영향을 받지 않는다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn_edges(coords: torch.Tensor, k: int) -> torch.Tensor:
    """coords(N,2)에서 각 노드의 최근접 이웃 k개로 edge_index(2, N*k)를 만든다.

    배열 순서를 전혀 안 쓰고 coords 값만으로 매번 새로 계산 — 패치 셔플/서브샘플과 독립적.
    """
    n = coords.shape[0]
    k = min(k, n - 1) if n > 1 else 0
    if k <= 0:
        return torch.zeros((2, 0), dtype=torch.long, device=coords.device)
    dist = torch.cdist(coords.float(), coords.float())  # (N, N)
    dist.fill_diagonal_(float("inf"))
    _, nn_idx = torch.topk(dist, k, dim=1, largest=False)  # (N, k) — 노드별 최근접 k개
    src = torch.arange(n, device=coords.device).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = nn_idx.reshape(-1)
    return torch.stack([src, dst], dim=0)  # (2, N*k)


class SimpleGCNLayer(nn.Module):
    """attention 없이 kNN 이웃의 평균만 집계하는 표준 GCN/GraphSAGE류 — "Spatial Blindness
    in Whole-Slide MIL"(arXiv:2605.17449)이 명시한 "The architecture is simple by design"를
    그대로 따른다. 2026-07-23 실측: q/k/v attention + relative bias를 얹은 KNNGraphAttentionLayer
    버전은 dropout을 넣어도 안 넣어도 Stage1 단독보다 test/external이 낮게 나왔다
    (findings_backlog.md) — task(91명, censored 생존)에 비해 과한 설계였을 가능성이 있어,
    파라미터를 훨씬 줄인 이 버전으로 재시도한다. 좌표는 여전히 "누가 이웃인가"(kNN 엣지
    구성)에만 쓰고 attention 가중치 계산에는 관여하지 않는다 — 상대 offset bias도 뺐다.
    """

    def __init__(self, dim: int, k: int = 8, dropout: float = 0.1, edge_dropout: float = 0.2):
        super().__init__()
        self.k = k
        self.edge_dropout = edge_dropout
        self.lin_self  = nn.Linear(dim, dim)
        self.lin_neigh = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """x: (N, D), coords: (N, 2) -> (N, D) — residual 연결 포함."""
        n = x.shape[0]
        edge_index = knn_edges(coords, self.k)
        if edge_index.shape[1] == 0:
            return x
        if self.training and self.edge_dropout > 0:
            keep = torch.rand(edge_index.shape[1], device=x.device) >= self.edge_dropout
            if keep.any():
                edge_index = edge_index[:, keep]
        src, dst = edge_index[0], edge_index[1]

        neigh_sum = torch.zeros(n, x.shape[1], device=x.device, dtype=x.dtype)
        neigh_sum.index_add_(0, src, x[dst])
        deg = torch.zeros(n, device=x.device, dtype=x.dtype)
        deg.index_add_(0, src, torch.ones_like(src, dtype=x.dtype))
        neigh_mean = neigh_sum / deg.clamp(min=1).unsqueeze(-1)  # 이웃 없는 노드는 0으로 유지

        out = F.relu(self.lin_self(x) + self.lin_neigh(neigh_mean))
        out = self.drop(out)
        return self.norm(x + out)


class RelativePositionBias(nn.Module):
    """두 패치의 상대 offset(Δrow, Δcol)을 attention score에 더할 bias로 변환(Swin류 발상).

    절대좌표가 아니라 상대 offset이라 슬라이드가 달라도, 좌표를 셔플해도(엣지 자체는 coords로
    재계산되므로) 같은 offset은 항상 같은 의미를 가진다.
    """

    def __init__(self, hidden_dim: int = 16, num_heads: int = 1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, rel_offset: torch.Tensor) -> torch.Tensor:
        """rel_offset: (E, 2) -> (E, num_heads)"""
        return self.mlp(rel_offset.float())


class KNNGraphAttentionLayer(nn.Module):
    """kNN 그래프 위 단일 GAT류 attention 레이어 + relative position bias.

    엣지 (i,j)는 매 forward마다 coords로 새로 계산되는 i의 최근접 이웃 j — 배열 순서 무관.
    """

    def __init__(
        self, dim: int, k: int = 8, num_heads: int = 1,
        dropout: float = 0.1, edge_dropout: float = 0.2,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim은 num_heads로 나누어떨어져야 합니다"
        self.k = k
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.edge_dropout = edge_dropout
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.rel_bias = RelativePositionBias(num_heads=num_heads)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """x: (N, D), coords: (N, 2) -> (N, D) — residual 연결 포함.

        [정규화, 2026-07-23] 91명 train으로 200 epoch를 돌리자 val_c_index가 초반 반짝(epoch
        6) 이후 완전히 무너지는 과적합이 실측 확인됨(findings_backlog.md) — patch_tokens가
        고정 캐시라 매 epoch 완전히 똑같은 그래프를 반복해서 보는 게 원인 중 하나로 보인다.
        edge_dropout(학습 중에만 kNN 엣지 일부를 무작위로 제거)으로 매 forward마다 다른
        그래프를 보게 강제하고, attn_drop/out_drop으로 일반적인 feature dropout도 더한다.
        """
        n = x.shape[0]
        edge_index = knn_edges(coords, self.k)
        if edge_index.shape[1] == 0:  # 패치가 1개뿐이면 이웃이 없음 — 그대로 통과
            return x
        if self.training and self.edge_dropout > 0:
            keep = torch.rand(edge_index.shape[1], device=x.device) >= self.edge_dropout
            if keep.any():  # 전부 드롭되면(운 나쁜 경우) 원래 엣지를 그대로 둔다
                edge_index = edge_index[:, keep]
        src, dst = edge_index[0], edge_index[1]  # src=쿼리(i), dst=이웃(j)

        q = self.q_proj(x).view(n, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(n, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(n, self.num_heads, self.head_dim)

        rel = (coords[dst] - coords[src]).float()  # (E, 2) — "j가 i에서 얼마나 떨어져 있나"
        bias = self.rel_bias(rel)  # (E, H)

        logits = (q[src] * k[dst]).sum(-1) / (self.head_dim ** 0.5)  # (E, H)
        logits = logits + bias
        logits = logits - logits.max()  # 안정화(softmax 이전 overflow 방지)
        exp = logits.exp()

        # 노드(src)별 softmax — scatter 방식(엣지 수가 노드마다 동일(=k)하지 않을 수 있음)
        denom = torch.zeros(n, self.num_heads, device=x.device, dtype=exp.dtype)
        denom.index_add_(0, src, exp)
        attn = exp / (denom[src] + 1e-8)  # (E, H)
        attn = self.attn_drop(attn)

        weighted_v = attn.unsqueeze(-1) * v[dst]  # (E, H, Dh)
        out = torch.zeros(n, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        out.index_add_(0, src, weighted_v.to(x.dtype))
        out = out.reshape(n, -1)
        out = self.out_drop(self.out_proj(out))
        return self.norm(x + out)


class SpatialResidualBranch(nn.Module):
    """patch_tokens(얼려둔 Stage1 CNN 출력) + coords -> risk correction 스칼라.

    Stage1의 risk_head/RNA/Clinical과 완전히 독립 — 결합은 scripts/train_spatial_residual.py에서
    "스칼라끼리 더하기"로만 이뤄진다(combined logits, ResTopoMIL과 동일).
    """

    def __init__(
        self, patch_dim: int, hidden_dim: int = 32, k: int = 8, num_layers: int = 2,
        dropout: float = 0.1, edge_dropout: float = 0.2, layer_type: str = "gcn",
    ):
        """layer_type: "gcn"(기본, SimpleGCNLayer — attention 없는 단순 이웃평균, 논문의
        "simple GNN" 재현) 또는 "attention"(KNNGraphAttentionLayer — q/k/v+relative bias,
        첫 시도에서 dropout 유무 둘 다 Stage1 단독보다 낮게 나온 무거운 버전, 비교용으로 남김).
        """
        super().__init__()
        self.in_proj = nn.Linear(patch_dim, hidden_dim)
        self.in_drop = nn.Dropout(dropout)
        layer_cls = {"gcn": SimpleGCNLayer, "attention": KNNGraphAttentionLayer}[layer_type]
        self.layers = nn.ModuleList([
            layer_cls(hidden_dim, k=k, dropout=dropout, edge_dropout=edge_dropout)
            for _ in range(num_layers)
        ])
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def encode(self, patch_tokens: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """슬라이드 1장 -> 그래프 attention을 거쳐 mean-pool한 (hidden_dim,) 표현.

        risk 스칼라 이전 단계 — shuffle_margin_loss가 바로 이 표현에 대해 계산된다.
        """
        x = self.in_drop(self.in_proj(patch_tokens))
        for layer in self.layers:
            x = layer(x, coords)
        return x.mean(dim=0)  # (hidden_dim,)

    def forward(self, patch_tokens: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        pooled = self.head_drop(self.encode(patch_tokens, coords))
        return self.head(pooled).view(())  # 스칼라


def shuffle_margin_loss(
    branch: SpatialResidualBranch, patch_tokens: torch.Tensor, coords: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """좌표만 무작위로 섞은 버전과 pooled 표현이 margin 이상 달라지도록 강제한다(코사인 유사도).

    patch_tokens(구성)는 그대로 두고 coords(배치)만 섞음 — branch가 좌표를 무시한 채 "또 다른
    외형 통계"로 퇴화하는 걸 막는 ResTopoMIL의 structure-aware texture 제약과 동일한 발상.
    """
    real = branch.encode(patch_tokens, coords)
    perm = torch.randperm(coords.shape[0], device=coords.device)
    shuffled = branch.encode(patch_tokens, coords[perm])
    cos_sim = F.cosine_similarity(real.unsqueeze(0), shuffled.unsqueeze(0)).squeeze(0)
    return F.relu(cos_sim - (1 - margin))  # 실제/셔플 표현이 너무 비슷하면(cos_sim 큼) 벌점
