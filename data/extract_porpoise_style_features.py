"""
PORPOISE 원본 스펙(CLAM 툴박스 관례) feature 재현 — 256x256 @ 20x(~0.5um/px) 패치를 ImageNet
사전학습 ResNet50의 layer3까지만 통과시킨 뒤 global average pool한 1024차원 벡터로 인코딩한다
(CLAM의 잘 알려진 "truncated ResNet50" 구성 — PMC 원문 확인: "a ResNet50 model pretrained on
ImageNet is used as an encoder to convert each 256x256 patch into 1024-dimensional feature
vector, via spatial average pooling after the 3rd residual block").

2026-09-05(3차 수정): 패치 자체는 새로 만들 필요가 없었다 — uni2native 리타일링 단계
(sbatch/preprocess_uni2native_retile_array_hpc.sh, data/patches_tcga_uni2native/tiles/<slide_id>/
*.jpg)가 이미 256px@0.5MPP(=PORPOISE 논문 스펙과 사실상 동일 해상도)로 tissue segmentation까지
끝낸 jpg를 만들어 뒀다. 1차/2차 버전에서 재구현한 Otsu tissue segmentation은 중복 작업이었음
— 그 jpg 픽셀 자체는 인코더와 무관하게 재사용 가능하고, 재사용 불가능한 건 오직
features_uni2native.pt(UNI2-h 임베딩, 1536차원)뿐이다. 이건 이미 다른 아키텍처를 통과시킨
"결과물"이라 PORPOISE의 ResNet50 임베딩과 호환되지 않는다 — 그래서 이 스크립트가 다시 해야 하는
일은 "같은 jpg를 다른 인코더에 통과시키는 GPU forward pass" 하나뿐이다(SVS/openslide 접근 없음).

사용법(HPC — uni2native 리타일링이 이미 끝나 있어야 함):
    python -m data.extract_porpoise_style_features
    python -m data.extract_porpoise_style_features --slide-ids TCGA-2J-AAB1-01Z-00-DX1....
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CSV_PATH = _ROOT / "porpoise" / "datasets_csv" / "tcga_paad_all_clean.csv.zip"
PATCHES_ROOT = _ROOT / "data" / "patches_tcga_uni2native" / "tiles"
OUT_DIR = _ROOT / "data" / "porpoise_style_features" / "tcga" / "pt_files"
DISPLAY_SIZE = 256

_IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize((DISPLAY_SIZE, DISPLAY_SIZE)),  # 이미 256px라 사실상 no-op, 안전장치용
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class TruncatedResNet50(nn.Module):
    """CLAM의 "truncated ResNet50" 관례 — layer3까지만 통과시킨 뒤 global average pool
    (1024채널). ImageNet 지도학습 사전학습 그대로(병리 도메인 특화 사전학습 없음) — PORPOISE
    논문/공식 repo 그대로."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slide-ids", type=str, default=None, help="쉼표구분, 주어지면 이 슬라이드만(디버그용).")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TruncatedResNet50().to(device).eval()
    print(f"TruncatedResNet50(ImageNet, layer3 truncated, 1024-dim) 로드 완료, device={device}")

    df = pd.read_csv(CSV_PATH, compression="zip")
    slide_ids = df["slide_id"].tolist()
    if args.slide_ids:
        wanted = set(args.slide_ids.split(","))
        slide_ids = [s for s in slide_ids if s in wanted]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"PORPOISE 공식 CSV 기준 슬라이드 {len(slide_ids)}개 처리 (패치 출처: {PATCHES_ROOT})")

    for i, slide_id in enumerate(slide_ids):
        out_path = OUT_DIR / f"{slide_id}.pt"
        if out_path.exists():
            continue
        tile_dir = PATCHES_ROOT / slide_id
        jpg_paths = sorted(tile_dir.glob("*.jpg")) if tile_dir.is_dir() else []
        if not jpg_paths:
            print(f"  [경고] uni2native 타일 없음: {slide_id} ({tile_dir})")
            continue

        feats = []
        batch_imgs = []
        for jpg_path in jpg_paths:
            img = Image.open(jpg_path).convert("RGB")
            batch_imgs.append(_IMAGENET_TRANSFORM(img))
            if len(batch_imgs) == args.batch_size:
                batch = torch.stack(batch_imgs).to(device)
                feats.append(model(batch).cpu())
                batch_imgs = []
        if batch_imgs:
            batch = torch.stack(batch_imgs).to(device)
            feats.append(model(batch).cpu())

        out = torch.cat(feats, dim=0)  # (N, 1024)
        torch.save(out, out_path)
        print(f"  [{i+1}/{len(slide_ids)}] {slide_id}: {out.shape} (타일 {len(jpg_paths)}개) -> {out_path}")

    print(f"\n완료 — {OUT_DIR} 확인.")


if __name__ == "__main__":
    main()
