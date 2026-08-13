"""
패치 jpg/png → frozen tile encoder(backbone) feature 사전 추출 스크립트

train.py는 backbone을 항상 고정해 학습한다(BN/LayerScale 등도 eval로 고정). 즉 같은
패치에 대한 backbone 출력은 epoch마다 동일하므로, 매 epoch JPEG 디코딩 + backbone forward를
반복하는 대신 패치당 한 번만 계산해 캐싱한다. (proj는 학습 대상이라 캐싱하지 않고
train.py에서 매번 forward한다.)

--backbone resnet50 (기본): models/cnn_encoder.py::CNNEncoder(ResNet50 Lunit SwAV, 2048-dim)
--backbone uni        : models/uni_encoder.py::UNIEncoder(UNI ViT-L/16, 1024-dim, 224 리사이즈)
두 backbone은 산출물 파일명이 달라(features.pt vs features_uni.pt) 서로 덮어쓰지 않는다.

출력:
    <patches_root>/<slide_id>/{features.pt|features_uni.pt}   (N_patches, feature_dim) float32
    행 순서 = data.patch_utils.list_patch_paths()와 동일한 정렬 순서

사용법:
    python -m utils.extract_features                            # 기본: cptac, resnet50
    python -m utils.extract_features --dataset tcga --backbone uni
"""
import argparse
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
from data.dataset import PATCHES_ROOT_ATTRS
from data.patch_utils import (
    FEATURES_FILENAME, FEATURES_UNI_FILENAME, FEATURES_UNI2_FILENAME,
    PATCH_TRANSFORM, UNI_PATCH_TRANSFORM, UNI2_NATIVE_PATCH_TRANSFORM, list_patch_paths,
)
from models.cnn_encoder import CNNEncoder
from models.uni_encoder import UNIEncoder
from models.uni2_encoder import UNI2hEncoder
from utils import load_env, send_slack

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# backbone별 배치 크기 — 실측 최적치. ResNet50은 1024px 원본 입력이라 무겁고(8: peak_mem 2GB,
# 24 img/s), UNI는 224px로 리사이즈해서 들어가는 대신 ViT-L이라 파라미터가 훨씬 커서 별도로
# 튜닝이 필요하다(README/실측치로 조정 예정, 우선 보수적인 기본값).
BACKBONE_REGISTRY = {
    "resnet50": {
        "encoder_cls":  CNNEncoder,
        "transform":    PATCH_TRANSFORM,
        "out_filename": FEATURES_FILENAME,
        "batch_size":   8,
    },
    "uni": {
        "encoder_cls":  UNIEncoder,
        "transform":    UNI_PATCH_TRANSFORM,
        "out_filename": FEATURES_UNI_FILENAME,
        "batch_size":   32,
    },
    "uni2": {
        # UNI2-h(ViT-H/14, 6.81억 파라미터, models/uni2_encoder.py) — UNI(ViT-L/16)보다
        # 파라미터 2배+토큰 수도 더 많아(512px 입력 기준 1024개 -> ~1296개) 배치를 훨씬
        # 보수적으로 잡는다. 패치 원본 해상도(1024px @ 1.0MPP)-UNI 학습 해상도 간극 문제는
        # UNI와 동일해 같은 UNI_PATCH_TRANSFORM(512 리사이즈)을 재사용한다 — dynamic_img_size=True
        # 라 224가 아닌 입력도 positional embedding 보간으로 그대로 받는다.
        # 첫 로그에서 GPU 메모리/처리 속도 보고 필요시 조정.
        "encoder_cls":  UNI2hEncoder,
        "transform":    UNI_PATCH_TRANSFORM,
        "out_filename": FEATURES_UNI2_FILENAME,
        "batch_size":   8,
    },
    "uni2native": {
        # 2026-08-12: UNI2-h 공식 스펙(256px@20x, ~0.5MPP) 그대로 재타일링한 타일
        # (data/preprocess.py --target-mpp 0.5 --tile-size 256, --patches-root로 별도
        # 디렉토리에 뽑음)에서 feature를 추출할 때 쓴다. 위 "uni2"는 1024px@1.0MPP 원본을
        # 512로 억지 리사이즈하지만, 여기 입력은 이미 256px로 딱 맞게 뽑혀 있어 리사이즈가
        # 항등연산이어야 한다(UNI2_NATIVE_PATCH_TRANSFORM 참조). out_filename은 별도
        # 디렉토리 트리에 쓰여 기존 "uni2" 산출물과 안 겹치므로 그대로 FEATURES_UNI2_FILENAME
        # 재사용 — scripts/reconcile_uni2native_features.py가 이걸 기존 patches 트리로
        # features_uni2native.pt(+coords_uni2native.pt)라는 이름으로 복사해온다.
        "encoder_cls":  UNI2hEncoder,
        "transform":    UNI2_NATIVE_PATCH_TRANSFORM,
        "out_filename": FEATURES_UNI2_FILENAME,
        "batch_size":   32,
    },
}


def _build_encoder(backbone: str):
    encoder = BACKBONE_REGISTRY[backbone]["encoder_cls"](embed_dim=1, with_backbone=True).to(DEVICE)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


@torch.no_grad()
def _extract_node(encoder, patch_paths: list[Path], transform, batch_size: int) -> torch.Tensor:
    chunks = []
    for i in range(0, len(patch_paths), batch_size):
        batch = torch.stack([
            transform(Image.open(p).convert("RGB"))
            for p in patch_paths[i : i + batch_size]
        ]).to(DEVICE, non_blocking=True)
        # ResNet50(CNNEncoder)은 feature map을 반환해 별도 pool이 필요하고,
        # UNI(UNIEncoder)는 ViT라 backbone 출력이 이미 pooled (B, feature_dim)이다.
        raw = encoder.backbone(batch)
        pooled = encoder.pool(raw) if hasattr(encoder, "pool") else raw
        chunks.append(pooled.cpu())
    return torch.cat(chunks)


def extract_features_for_root(
    patches_root: Path, backbone: str = "resnet50", encoder=None, notify: bool = True,
) -> int:
    """patches_root 바로 아래의 각 디렉터리(슬라이드/노드 1개당 1폴더)에 feature 파일을 생성한다.
    이미 산출물이 있는 디렉터리는 skip한다.

    다른 전처리 파이프라인(예: data/preprocess_cptac.py)이 타일링 직후 같은 프로세스 안에서
    바로 이어 호출할 수 있도록 만든 진입점 — encoder를 넘기면 재사용하고, 안 넘기면 새로 로드한다.

    Returns: 새로 추출한 디렉터리 수
    """
    spec = BACKBONE_REGISTRY[backbone]
    out_filename = spec["out_filename"]
    transform    = spec["transform"]
    batch_size   = spec["batch_size"]

    start_time = datetime.now()
    owns_encoder = encoder is None
    if owns_encoder:
        encoder = _build_encoder(backbone)

    node_dirs = sorted(d for d in patches_root.iterdir() if d.is_dir())

    try:
        from tqdm import tqdm
        node_dirs = tqdm(node_dirs, desc=f"Extracting {backbone} features", unit="node")
    except ImportError:
        pass

    done = 0
    for node_dir in node_dirs:
        out_path = node_dir / out_filename
        if out_path.exists():
            continue

        patch_paths = list_patch_paths(node_dir)
        if not patch_paths:
            continue

        features = _extract_node(encoder, patch_paths, transform, batch_size)
        torch.save(features, out_path)
        done += 1

    elapsed = datetime.now() - start_time
    h, rem  = divmod(int(elapsed.total_seconds()), 3600)
    m, s    = divmod(rem, 60)
    print(f"완료: {done}개 노드 → {patches_root}/<slide_id>/{out_filename}")
    if notify:
        send_slack(
            f":white_check_mark: *Feature 추출 완료* ({backbone})\n"
            f"> 저장 위치: `{patches_root}/<slide_id>/{out_filename}`\n"
            f"> 처리 노드: *{done}개*\n"
            f"> 소요 시간: {h}h {m}m {s}s"
        )
    return done


def main():
    parser = argparse.ArgumentParser(description="패치 jpg/png → frozen tile encoder feature 사전 추출")
    parser.add_argument("--dataset", type=str, default="cptac", choices=["tcga", "cptac"],
                        help="처리할 코호트 (기본: cptac). config.DataConfig()의 "
                             "patches_root_{tcga,cptac}/tiles/ 를 대상으로 한다.")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=list(BACKBONE_REGISTRY),
                        help="사용할 frozen tile encoder (기본: resnet50=Lunit SwAV). "
                             "uni/uni2는 각각 HuggingFace gated repo(MahmoodLab/UNI, MahmoodLab/UNI2-h) "
                             "접근 승인 + .env의 HF_TOKEN이 필요하다(같은 토큰 재사용 가능). "
                             "uni2native는 --patches-root로 별도 재타일링 트리(256px@0.5MPP, "
                             "data/preprocess.py --target-mpp 0.5 --tile-size 256)를 가리켜야 한다.")
    parser.add_argument("--patches-root", type=str, default=None,
                        help="2026-08-12: 주어지면 config.DataConfig()의 기본 patches_root_{dataset} "
                             "대신 이 경로 아래 tiles/를 대상으로 한다 — UNI2-h 공식 스펙(256px@20x) "
                             "재타일링 결과물처럼 기존 patches 트리와 다른 별도 디렉토리에서 추출할 때 "
                             "쓴다(예: --patches-root data/patches_tcga_uni2native).")
    args = parser.parse_args()

    cfg = DataConfig()
    patches_root = Path(args.patches_root) if args.patches_root else Path(getattr(cfg, PATCHES_ROOT_ATTRS[args.dataset]))
    extract_features_for_root(patches_root / "tiles", backbone=args.backbone)


if __name__ == "__main__":
    load_env()
    try:
        main()
    except Exception as e:
        send_slack(f":x: *Feature 추출 에러*\n```{type(e).__name__}: {e}```")
        raise
