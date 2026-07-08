"""
패치 jpg/png → CNN(backbone+pool) feature 사전 추출 스크립트

train.py는 model.cnn.backbone.requires_grad_(False)로 CNN backbone을 항상 고정해
학습한다(BN도 eval로 고정). 즉 같은 패치에 대한 backbone 출력은 epoch마다 동일하므로,
매 epoch JPEG 디코딩 + CNN forward를 반복하는 대신 패치당 한 번만 계산해 캐싱한다.
(CNNEncoder.proj는 학습 대상이라 캐싱하지 않고 train.py에서 매번 forward한다.)

출력:
    <patches_root>/<slide_id>/features.pt   (N_patches, feature_dim) float32 tensor
    행 순서 = data.patch_dataset.list_patch_paths()와 동일한 정렬 순서

사용법:
    python -m data.extract_features   (또는 python data/extract_features.py 직접 실행도 가능)
"""
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent  # Path-ViT 프로젝트 루트
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DataConfig
from data.patch_dataset import FEATURES_FILENAME, PATCH_TRANSFORM, list_patch_paths
from models.cnn_encoder import CNNEncoder
from utils import load_env, send_slack

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64


def _build_encoder() -> CNNEncoder:
    encoder = CNNEncoder(embed_dim=1, with_backbone=True).to(DEVICE)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


@torch.no_grad()
def _extract_node(encoder: CNNEncoder, patch_paths: list[Path]) -> torch.Tensor:
    chunks = []
    for i in range(0, len(patch_paths), BATCH_SIZE):
        batch = torch.stack([
            PATCH_TRANSFORM(Image.open(p).convert("RGB"))
            for p in patch_paths[i : i + BATCH_SIZE]
        ]).to(DEVICE, non_blocking=True)
        feat_map = encoder.backbone(batch)
        pooled   = encoder.pool(feat_map)   # (B, BACKBONE_DIM)
        chunks.append(pooled.cpu())
    return torch.cat(chunks)


def extract_features_for_root(patches_root: Path, encoder: CNNEncoder | None = None, notify: bool = True) -> int:
    """patches_root 바로 아래의 각 디렉터리(슬라이드/노드 1개당 1폴더)에 features.pt를 생성한다.
    이미 features.pt가 있는 디렉터리는 skip한다.

    다른 전처리 파이프라인(예: data/preprocess_cptac.py)이 타일링 직후 같은 프로세스 안에서
    바로 이어 호출할 수 있도록 만든 진입점 — encoder를 넘기면 재사용하고, 안 넘기면 새로 로드한다.

    Returns: 새로 추출한 디렉터리 수
    """
    start_time = datetime.now()
    owns_encoder = encoder is None
    if owns_encoder:
        encoder = _build_encoder()

    node_dirs = sorted(d for d in patches_root.iterdir() if d.is_dir())

    try:
        from tqdm import tqdm
        node_dirs = tqdm(node_dirs, desc="Extracting CNN features", unit="node")
    except ImportError:
        pass

    done = 0
    for node_dir in node_dirs:
        out_path = node_dir / FEATURES_FILENAME
        if out_path.exists():
            continue

        patch_paths = list_patch_paths(node_dir)
        if not patch_paths:
            continue

        features = _extract_node(encoder, patch_paths)
        torch.save(features, out_path)
        done += 1

    elapsed = datetime.now() - start_time
    h, rem  = divmod(int(elapsed.total_seconds()), 3600)
    m, s    = divmod(rem, 60)
    print(f"완료: {done}개 노드 → {patches_root}/<slide_id>/{FEATURES_FILENAME}")
    if notify:
        send_slack(
            f":white_check_mark: *Feature 추출 완료*\n"
            f"> 저장 위치: `{patches_root}/<slide_id>/{FEATURES_FILENAME}`\n"
            f"> 처리 노드: *{done}개*\n"
            f"> 소요 시간: {h}h {m}m {s}s"
        )
    return done


def main():
    cfg = DataConfig()
    extract_features_for_root(Path(cfg.patches_root))


if __name__ == "__main__":
    load_env()
    try:
        main()
    except Exception as e:
        send_slack(f":x: *Feature 추출 에러*\n```{type(e).__name__}: {e}```")
        raise
