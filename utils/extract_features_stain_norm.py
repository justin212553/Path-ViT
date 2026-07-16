"""
패치 jpg/png → Macenko stain normalization → frozen ResNet50(Lunit SwAV) feature 사전 추출.

utils/extract_features.py와 산출물이 다른 별도 스크립트다: 기존 features.pt(원본 패치 feature)는
그대로 두고, 정규화된 버전을 features_norm.pt로 새로 저장한다(롤백 가능, 기존 학습 파이프라인에
영향 없음).

배경: check_domain_shift.py로 raw CNN feature만으로 TCGA/CPTAC 기관을 구분하는 도메인 분류기가
AUC=0.78을 기록해, feature 공간에 기관/스캐너 배치 효과가 강하게 남아 있음을 확인했다. Macenko
stain normalization이 이 배치 효과(주로 염색 색상 편차에서 기인)를 줄이는지 검증하기 위해
전체 코호트를 정규화 후 재추출한다.

target(정규화 기준 이미지)은 TCGA 코호트를 슬라이드명 정렬 후 첫 번째 슬라이드의 첫 번째 패치로
고정한다 — 재현성을 위해 전체 실행에서 항상 동일한 target을 사용한다.

Macenko normalize()는 이미지 1장 단위로만 동작해(배치 미지원) 정규화 자체는 패치별로 순차
수행하고, 정규화된 이미지를 모아 backbone forward만 배치로 처리한다.

출력:
    <patches_root>/<slide_id>/features_norm.pt   (N_patches, 2048) float32
    행 순서 = data.patch_utils.list_patch_paths()와 동일한 정렬 순서

사용법:
    python -m utils.extract_features_stain_norm                    # 기본: cptac
    python -m utils.extract_features_stain_norm --dataset tcga
    python -m utils.extract_features_stain_norm --dataset tcga --resume-only   # 이미 있는 산출물 skip(기본 동작)
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import torchstain
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DataConfig
from data.dataset import PATCHES_ROOT_ATTRS
from data.patch_utils import FEATURES_NORM_FILENAME, PATCH_TRANSFORM, list_patch_paths
from models.cnn_encoder import CNNEncoder
from utils import load_env, send_slack

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8  # extract_features.py의 resnet50 배치 크기와 동일(1024px 원본 입력 기준 실측치)

STAIN_T = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 255)])


def _build_encoder() -> CNNEncoder:
    encoder = CNNEncoder(embed_dim=1, with_backbone=True).to(DEVICE)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def _build_normalizer(target_path: Path):
    # Macenko normalize()는 실측상 GPU(cuda)에서 커널 launch 오버헤드로 CPU보다 140배 이상
    # 느리다(30장 기준 0.12 img/s vs 16.9 img/s) — 정규화는 항상 CPU 텐서로 수행하고,
    # backbone forward만 DEVICE(cuda)로 옮긴다.
    target_img = Image.open(target_path).convert("RGB")
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
    normalizer.fit(STAIN_T(target_img))
    return normalizer


def _normalize_image(normalizer, path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    try:
        norm, _, _ = normalizer.normalize(I=STAIN_T(img), stains=True)  # (H,W,3), [0,255], CPU
        return Image.fromarray(norm.clamp(0, 255).byte().numpy())
    except Exception:
        # Macenko가 염색 성분을 못 찾는 극소수 패치(배경 위주 등)는 원본을 그대로 사용
        return img


@torch.no_grad()
def _extract_node(encoder, normalizer, patch_paths: list[Path]) -> torch.Tensor:
    chunks = []
    for i in range(0, len(patch_paths), BATCH_SIZE):
        batch_paths = patch_paths[i : i + BATCH_SIZE]
        batch = torch.stack([
            PATCH_TRANSFORM(_normalize_image(normalizer, p)) for p in batch_paths
        ]).to(DEVICE, non_blocking=True)
        raw = encoder.backbone(batch)
        pooled = encoder.pool(raw)
        chunks.append(pooled.cpu())
    return torch.cat(chunks)


def extract_normalized_features_for_root(
    patches_root: Path, normalizer, encoder=None, notify: bool = True,
) -> int:
    start_time = datetime.now()
    owns_encoder = encoder is None
    if owns_encoder:
        encoder = _build_encoder()

    node_dirs = sorted(d for d in patches_root.iterdir() if d.is_dir())

    try:
        from tqdm import tqdm
        node_dirs = tqdm(node_dirs, desc="Extracting stain-normalized features", unit="node")
    except ImportError:
        pass

    done = 0
    for node_dir in node_dirs:
        out_path = node_dir / FEATURES_NORM_FILENAME
        if out_path.exists():
            continue

        patch_paths = list_patch_paths(node_dir)
        if not patch_paths:
            continue

        features = _extract_node(encoder, normalizer, patch_paths)
        torch.save(features, out_path)
        done += 1

    elapsed = datetime.now() - start_time
    h, rem  = divmod(int(elapsed.total_seconds()), 3600)
    m, s    = divmod(rem, 60)
    print(f"완료: {done}개 노드 → {patches_root}/<slide_id>/{FEATURES_NORM_FILENAME}")
    if notify:
        send_slack(
            f":white_check_mark: *Stain-normalized feature 추출 완료*\n"
            f"> 저장 위치: `{patches_root}/<slide_id>/{FEATURES_NORM_FILENAME}`\n"
            f"> 처리 노드: *{done}개*\n"
            f"> 소요 시간: {h}h {m}m {s}s"
        )
    return done


def main():
    parser = argparse.ArgumentParser(description="패치 jpg/png → Macenko 정규화 → ResNet50 feature 사전 추출")
    parser.add_argument("--dataset", type=str, default="cptac", choices=["tcga", "cptac", "both"])
    args = parser.parse_args()

    cfg = DataConfig()

    tcga_root = Path(getattr(cfg, PATCHES_ROOT_ATTRS["tcga"])) / "tiles"
    target_path = list_patch_paths(sorted(tcga_root.iterdir())[0])[0]
    print(f"Macenko target (고정): {target_path}")
    normalizer = _build_normalizer(target_path)
    encoder = _build_encoder()

    datasets = ["tcga", "cptac"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        patches_root = Path(getattr(cfg, PATCHES_ROOT_ATTRS[ds])) / "tiles"
        print(f"\n=== {ds} ===")
        extract_normalized_features_for_root(patches_root, normalizer, encoder=encoder)


if __name__ == "__main__":
    load_env()
    try:
        main()
    except Exception as e:
        send_slack(f":x: *Stain-normalized feature 추출 에러*\n```{type(e).__name__}: {e}```")
        raise
