"""
패치 jpg/png → ResNet50(backbone+pool) feature 사전 추출 스크립트

train.py는 model.cnn.backbone.requires_grad_(False)로 ResNet50 backbone을 항상 고정해
학습한다(BN도 eval로 고정). 즉 같은 패치에 대한 backbone 출력은 epoch마다 동일하므로,
매 epoch JPEG 디코딩 + ResNet50 forward를 반복하는 대신 패치당 한 번만 계산해 캐싱한다.
(CNNEncoder.proj는 학습 대상이라 캐싱하지 않고 train.py에서 매번 forward한다.)

출력:
    <patches_root>/<slide_id>/features.pt   (N_patches, 2048) float32 tensor
    행 순서 = data.patch_dataset.list_patch_paths()와 동일한 정렬 순서

사용법:
    python -m data.extract_features
"""
from pathlib import Path

import torch
from PIL import Image

from config import DataConfig
from data.patch_dataset import FEATURES_FILENAME, PATCH_TRANSFORM, list_patch_paths
from models.cnn_encoder import CNNEncoder

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
        pooled   = encoder.pool(feat_map)   # (B, 2048)
        chunks.append(pooled.cpu())
    return torch.cat(chunks)


def main():
    cfg = DataConfig()
    patches_root = Path(cfg.patches_root)

    encoder   = _build_encoder()
    node_dirs = sorted(d for d in patches_root.iterdir() if d.is_dir())

    try:
        from tqdm import tqdm
        node_dirs = tqdm(node_dirs, desc="Extracting ResNet50 features", unit="node")
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

    print(f"완료: {done}개 노드 → {patches_root}/<slide_id>/{FEATURES_FILENAME}")


if __name__ == "__main__":
    main()
