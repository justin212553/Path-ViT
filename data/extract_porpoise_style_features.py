"""
PORPOISE 원본 스펙(CLAM 툴박스 관례) feature 재현 — 256x256 @ 20x(~0.5um/px) 패치를 ImageNet
사전학습 ResNet50의 layer3까지만 통과시킨 뒤 global average pool한 1024차원 벡터로 인코딩한다
(CLAM의 잘 알려진 "truncated ResNet50" 구성 — README.md 3절, PORPOISE 논문의 원본 방식).

2026-09-05 확인: sota/PORPOISE/inputs/tcga_paad_20x_features/pt_files/의 기존 .pt 파일은
PORPOISE 원본이 아니라 저희 UNI2 feature를 그쪽 학습 코드 입력 형식에 맞춰 재포장한 것이었다
(shape이 1536차원 — UNI2-h와 동일, 1024가 아님). "PORPOISE가 우리와 다른 전처리로 신호를
뽑아낸 게 아닐까"를 직접 검증하려면 진짜 원본 스펙(ImageNet ResNet50 truncated, 1024차원)으로
다시 추출해야 한다.

패치 위치는 새로 tissue segmentation을 하지 않고, 이미 있는 UNI2-h 공식 feature 추출
(data/uni2h_official_features/{tcga,cptac}/*.h5)의 coords를 그대로 재사용한다 — 이미 조직
영역으로 검증된 위치이고, "같은 패치 위치, 다른 backbone"으로 통제해야 backbone 차이만 순수하게
비교할 수 있다. coords는 level-0 픽셀, 512x512 stride(scripts/extract_cluster_exemplars.py와
동일 관례) — 512x512로 읽어 256x256으로 리사이즈하면 실효 20x(0.5um/px)가 된다.

사용법(HPC — SVS 원본 필요):
    python -m data.extract_porpoise_style_features --datasets tcga,cptac
    python -m data.extract_porpoise_style_features --datasets tcga --case-ids TCGA-2J-AAB1
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import openslide
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FEATURES_ROOT = _ROOT / "data" / "uni2h_official_features"  # coords 출처(패치 위치 재사용)
WSI_ROOTS = {"tcga": _ROOT / "data" / "tcga_paad_wsi", "cptac": _ROOT / "data" / "cptac_pda_wsi"}
OUT_ROOT = _ROOT / "data" / "porpoise_style_features"
READ_SIZE_LEVEL0 = 512  # coords stride와 동일
DISPLAY_SIZE = 256  # CLAM/PORPOISE 공식 patch 크기(20x)

_IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize((DISPLAY_SIZE, DISPLAY_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class TruncatedResNet50(nn.Module):
    """CLAM의 "truncated ResNet50" 관례 — layer3까지만 통과시킨 뒤 global average pool
    (1024채널, torchvision resnet50 layer3 출력 채널 수와 동일). ImageNet 지도학습
    사전학습 그대로(병리 도메인 특화 사전학습 없음) — PORPOISE 논문/공식 repo README 그대로."""

    def __init__(self):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        net = torchvision.models.resnet50(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        self.pool = nn.AdaptiveAvgPool2d(1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return x  # (B, 1024)


def _find_wsi_path(dataset: str, slide_id: str) -> Path | None:
    root = WSI_ROOTS[dataset]
    cand = root / f"{slide_id}.svs"
    if cand.exists():
        return cand
    matches = list(root.glob(f"{slide_id}*.svs"))
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default="tcga,cptac")
    parser.add_argument("--case-ids", type=str, default=None, help="쉼표구분, 주어지면 이 케이스만(디버그용).")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TruncatedResNet50().to(device).eval()
    print(f"TruncatedResNet50(ImageNet, layer3 truncated, 1024-dim) 로드 완료, device={device}")

    case_filter = set(args.case_ids.split(",")) if args.case_ids else None
    datasets = args.datasets.split(",")

    for ds in datasets:
        ds_dir = FEATURES_ROOT / ds
        out_dir = OUT_ROOT / ds / "pt_files"
        out_dir.mkdir(parents=True, exist_ok=True)
        h5_paths = sorted(ds_dir.glob("*.h5"))
        print(f"[{ds}] {len(h5_paths)}개 슬라이드")

        for i, h5_path in enumerate(h5_paths):
            slide_id = h5_path.stem
            if case_filter and not any(slide_id.startswith(c) for c in case_filter):
                continue
            out_path = out_dir / f"{slide_id}.pt"
            if out_path.exists():
                continue

            with h5py.File(h5_path, "r") as f:
                coords = f["coords"][0]  # (N, 2) level-0 픽셀

            wsi_path = _find_wsi_path(ds, slide_id)
            if wsi_path is None:
                print(f"  [경고] WSI 원본을 못 찾음: {ds}/{slide_id}")
                continue
            slide = openslide.OpenSlide(str(wsi_path))

            feats = []
            batch_imgs = []
            for x, y in coords:
                region = slide.read_region((int(x), int(y)), 0, (READ_SIZE_LEVEL0, READ_SIZE_LEVEL0)).convert("RGB")
                batch_imgs.append(_IMAGENET_TRANSFORM(region))
                if len(batch_imgs) == args.batch_size:
                    batch = torch.stack(batch_imgs).to(device)
                    feats.append(model(batch).cpu())
                    batch_imgs = []
            if batch_imgs:
                batch = torch.stack(batch_imgs).to(device)
                feats.append(model(batch).cpu())
            slide.close()

            out = torch.cat(feats, dim=0)  # (N, 1024)
            torch.save(out, out_path)
            print(f"  [{i+1}/{len(h5_paths)}] {slide_id}: {out.shape} -> {out_path}")

    print(f"\n완료 — {OUT_ROOT}/{{tcga,cptac}}/pt_files/ 확인.")


if __name__ == "__main__":
    main()
