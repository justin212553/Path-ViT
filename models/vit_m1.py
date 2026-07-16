"""
ViT_M1 — WSI 단위 MIL 모델 (ViT + ABMIL), train.py의 --M1(기본값) 플래그로 선택되는 모델

패치 → CNN → 공간 임베딩 ViT(self-attention) → attention pooling → WSI 임베딩.
OS(overall survival) risk score 예측(Cox Proportional Hazards)을 위한 표현을 만든다.

환자 1명이 슬라이드를 여러 장 보유할 수 있어(WSISurvivalDataset) risk_head는 슬라이드
단위 forward가 아니라 환자 단위로 임베딩을 풀링한 뒤 별도로 적용해야 한다
(train.py::_patient_risk, eval.py::evaluate_survival 참조).

Forward 출력:
    embed        : (D,)          — WSI 임베딩 (risk_head 적용 전)
    attn_weights : (N_patches,)  — 패치별 attention 가중치 (시각화용)
"""
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from .cnn_encoder import CNNEncoder
from .uni_encoder import UNIEncoder
from .vit_encoder import ViTEncoder
from config import ModelConfig

# tile encoder(backbone) 선택 레지스트리 — CNNEncoder/UNIEncoder 둘 다 forward/forward_pooled/
# .backbone 인터페이스가 동일해 여기서만 바꾸면 나머지 코드는 그대로 재사용된다.
TILE_ENCODER_REGISTRY = {
    "resnet50": CNNEncoder,
    "uni":      UNIEncoder,
}


class AttentionPooling(nn.Module):
    """
    Gated attention pooling (Ilse et al., 2018 ABMIL).

    [존재 이유]
    ViT를 지난 뒤 N개의 패치 토큰이 남는다. 이 토큰들은 WSI 내 각 위치의 표현이지만,
    최종 분류는 WSI 단위 단일 벡터를 요구한다.
    ABMIL은 "어떤 패치가 WSI 라벨 결정에 중요한가"를 학습 가능한 attention 가중치로
    결정해 N개 토큰을 1개 WSI 임베딩으로 집계한다.

    [Cluster Query Token으로 전환 시 이 모듈이 제거되는 이유]
    Cluster Query Token 방식에서는 K개의 쿼리 토큰이 ViT 내부 attention을 통해
    이미 유형별 집계를 완료한다. 즉 "N → 1 집계" 문제가 "K개 유형별 표현 → 히스토그램
    가중합"으로 대체되므로, 별도의 ABMIL 단계가 불필요해진다.

    [구조]
    attn_v: tanh 게이트  — 패치 표현의 방향성 포착
    attn_u: sigmoid 게이트 — 패치 표현의 크기/활성 포착
    두 게이트의 element-wise 곱 → attn_w로 스칼라 점수 산출 (gated attention)
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attn_v = nn.Linear(embed_dim, hidden_dim)   # tanh 게이트
        self.attn_u = nn.Linear(embed_dim, hidden_dim)   # sigmoid 게이트
        self.attn_w = nn.Linear(hidden_dim, 1)           # 스칼라 점수

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (N, D) — ViT를 지난 패치 토큰. N은 WSI마다 다름.
        Returns:
            wsi_embed:    (D,) — attention 가중합으로 집계된 WSI 임베딩
            attn_weights: (N,) — 패치별 attention 가중치 (합=1, 시각화·해석용)
        """
        # gated attention: tanh × sigmoid → 두 게이트가 방향성과 크기를 동시에 제어
        gate = torch.tanh(self.attn_v(tokens)) * torch.sigmoid(self.attn_u(tokens))  # (N, H)

        # 각 패치의 중요도 점수 → softmax로 확률 분포화 (합=1 보장)
        scores = self.attn_w(gate).squeeze(-1)        # (N,)
        attn_weights = torch.softmax(scores, dim=0)   # (N,)

        # 중요도 가중합으로 N개 패치 토큰을 단일 WSI 임베딩으로 집계
        wsi_embed = (attn_weights.unsqueeze(-1) * tokens).sum(dim=0)  # (D,)
        return wsi_embed, attn_weights


class ViT_M1(nn.Module):
    def __init__(self, cfg: ModelConfig, precomputed: bool = True, backbone: str = "resnet50"):
        """
        Args:
            precomputed: True면 tile encoder backbone을 생성하지 않는다 — 항상 사전 추출된
                         pooled feature(features 인자)만 입력으로 받는 모드.
                         False면 patch_paths로 이미지를 직접 디코딩/forward한다.
            backbone:    "resnet50"(기본, CNNEncoder=ResNet50 Lunit SwAV, 2048-dim) 또는
                         "uni"(UNIEncoder=UNI ViT-L/16, 1024-dim, 224 리사이즈).
                         data/extract_features.py --backbone과 값을 맞춰야 캐싱된 feature
                         차원이 일치한다. attribute 이름은 backbone이 uni여도 관례상 self.cnn을
                         유지한다(train.py의 model.cnn.backbone 참조 전부와 호환).
        """
        super().__init__()
        self.precomputed = precomputed
        self.backbone_name = backbone
        encoder_cls = TILE_ENCODER_REGISTRY[backbone]
        self.cnn = encoder_cls(cfg.embed_dim, with_backbone=not precomputed)
        self.vit = ViTEncoder(cfg.embed_dim, cfg.num_heads,
                              cfg.num_transformer_layers, cfg.dropout,
                              use_grad_checkpoint=cfg.grad_checkpoint,
                              num_landmarks=cfg.num_landmarks)
        self.attn_pool = AttentionPooling(cfg.embed_dim)

        self.risk_head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, 1),
        )

    def _patch_tokens(
        self,
        coords: torch.Tensor,
        patch_paths: list[Path] | None = None,
        features: torch.Tensor | None = None,
        transform=None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """
        CNN을 통과시켜 (N_patches, embed_dim) 패치 토큰을 만든다.

        Args:
            coords:      (N_patches, 2) — device 참조용(patch_paths/features 자체엔 device 정보 없음)
            patch_paths: N개 패치 이미지 파일 경로 (precomputed=False 모드) — 이미지 디코딩을
                         chunk_size 단위로 지연 로딩해 한 번에 메모리에 올리는 패치 수를 제한한다
                         (패치 수에 cap이 없는 대형 WSI에서 host RAM OOM 방지)
            features:    (N_patches, 2048) 사전 추출된 backbone+pool feature (precomputed=True 모드)
            transform:   패치 이미지 → 텐서 변환. patch_paths 모드에서만 사용
            chunk_size:  CNN을 이 크기 단위로 나눠 실행. None이면 한 번에 실행. patch_paths 모드에서만 사용
        """
        device = coords.device

        if features is not None:
            return self.cnn.forward_pooled(features.to(device, non_blocking=True))

        chunk_size = chunk_size or len(patch_paths)
        return torch.cat([
            self.cnn(
                torch.stack([
                    transform(Image.open(p).convert("RGB"))
                    for p in patch_paths[i : i + chunk_size]
                ]).to(device, non_blocking=True)
            )
            for i in range(0, len(patch_paths), chunk_size)
        ])

    def forward(
        self,
        coords: torch.Tensor,
        patch_paths: list[Path] | None = None,
        features: torch.Tensor | None = None,
        transform=None,
        chunk_size: int | None = None,
    ) -> dict:
        """
        risk_head를 적용하기 전, WSI 1장을 attention-pooled 임베딩 1개로 집계한다.
        환자 1명이 슬라이드를 여러 장 보유하는 경우(WSISurvivalDataset) 슬라이드별로
        이 메서드를 호출한 뒤 임베딩을 환자 단위로 풀링하고 나서 risk_head를 적용해야 한다.

        Returns:
            embed:        (D,) — WSI 임베딩
            attn_weights: (N_patches,)
        """
        patch_tokens = self._patch_tokens(coords, patch_paths, features, transform, chunk_size)
        ctx_tokens   = self.vit(patch_tokens, coords)          # (N, D)
        wsi_embed, attn_weights = self.attn_pool(ctx_tokens)   # (D,), (N,)
        return {"embed": wsi_embed, "attn_weights": attn_weights}
