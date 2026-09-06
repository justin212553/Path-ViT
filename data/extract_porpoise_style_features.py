"""
PORPOISE 원본 스펙(CLAM 툴박스 관례) feature 재현 — 256x256 @ 20x(~0.5um/px) 패치를 ImageNet
사전학습 ResNet50의 layer3까지만 통과시킨 뒤 global average pool한 1024차원 벡터로 인코딩한다
(CLAM의 잘 알려진 "truncated ResNet50" 구성 — PMC 원문 확인: "a ResNet50 model pretrained on
ImageNet is used as an encoder to convert each 256x256 patch into 1024-dimensional feature
vector, via spatial average pooling after the 3rd residual block").

2026-09-05(2차 수정): 1차 버전은 패치 위치를 UNI2-h 공식 추출(data/uni2h_official_features/
tcga/*.h5)의 coords에서 재사용했는데, 그 컬렉션이 203개 슬라이드(전부 DX=진단용/영구절편)뿐이라
PORPOISE 공식 CSV(porpoise/datasets_csv/tcga_paad_all_clean.csv.zip)가 실제로 쓰는 377개
슬라이드(DX+TS/BS=냉동절편·생검 섞임)의 절반 이상을 못 커버했다 — HPC 실행이 없는 파일 때문에
크래시. 이번엔 UNI2-h에 기대지 않고, PORPOISE가 실제로 필요로 하는 슬라이드 목록(CSV)을 직접
기준으로 삼아 **독립적으로 tissue segmentation**을 해서 전체 377개를 커버한다(raw SVS는
data/tcga_paad_wsi/에 377개 전부 로컬 확인 완료).

Tissue segmentation은 CLAM 관례(Otsu thresholding on HSV saturation channel)를 cv2/skimage
없이 numpy+scipy만으로 재구현 — 이 환경엔 둘 다 없음(2026-09-05 확인).

사용법(HPC — SVS 원본 필요):
    python -m data.extract_porpoise_style_features
    python -m data.extract_porpoise_style_features --slide-ids TCGA-2J-AAB1-01Z-00-DX1....
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from scipy import ndimage
from torchvision import transforms

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CSV_PATH = _ROOT / "porpoise" / "datasets_csv" / "tcga_paad_all_clean.csv.zip"
WSI_ROOT = _ROOT / "data" / "tcga_paad_wsi"
OUT_DIR = _ROOT / "data" / "porpoise_style_features" / "tcga" / "pt_files"
PATCH_SIZE_LEVEL0 = 512   # UNI2-h와 동일 stride 관례 유지(512 읽어 256으로 리사이즈 = 실효 20x)
DISPLAY_SIZE = 256
THUMBNAIL_MAX_DIM = 2048
TISSUE_FRACTION_THRESHOLD = 0.10  # 패치 영역 내 조직 비율 최소 기준(CLAM 기본값과 유사한 수준)

_IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize((DISPLAY_SIZE, DISPLAY_SIZE)),
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


def _otsu_threshold(gray: np.ndarray) -> int:
    """cv2/skimage 없이 표준 Otsu 알고리즘 직접 구현(between-class variance 최대화)."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_bg = 0.0
    w_bg = 0.0
    best_var, best_t = -1.0, 0
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_total - sum_bg) / w_fg
        var_between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var, best_t = var_between, t
    return best_t


def _tissue_mask(slide: openslide.OpenSlide) -> tuple[np.ndarray, float]:
    """CLAM 관례 — HSV saturation 채널 + Otsu thresholding으로 조직 영역 이진 마스크를
    thumbnail 해상도에서 계산한다. 흰 배경(낮은 채도)과 조직(높은 채도)을 가르는 표준 방식.

    Returns:
        mask: (H_thumb, W_thumb) bool, True=조직
        scale: level-0 픽셀 좌표 -> thumbnail 픽셀 좌표로 줄이는 배율(thumb_size / level0_size)
    """
    w0, h0 = slide.level_dimensions[0]
    scale = THUMBNAIL_MAX_DIM / max(w0, h0)
    thumb = slide.get_thumbnail((int(w0 * scale), int(h0 * scale))).convert("RGB")
    hsv = np.array(thumb.convert("HSV"))
    sat = hsv[:, :, 1]
    sat_blurred = ndimage.median_filter(sat, size=7)
    thresh = _otsu_threshold(sat_blurred)
    mask = sat_blurred > thresh
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
    mask = ndimage.binary_fill_holes(mask)
    # 아주 작은 파편(노이즈) 제거 — 전체 조직 영역의 1% 미만인 연결요소는 배경으로 간주
    labeled, n = ndimage.label(mask)
    if n > 0:
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        min_size = mask.size * 0.001
        for i, s in enumerate(sizes, start=1):
            if s < min_size:
                mask[labeled == i] = False
    return mask, scale


def _generate_patch_coords(slide: openslide.OpenSlide, mask: np.ndarray, scale: float) -> np.ndarray:
    """level-0 픽셀 기준 512px 격자 좌표 중, 마스크 상 조직 비율이 기준 이상인 것만 남긴다."""
    w0, h0 = slide.level_dimensions[0]
    xs = np.arange(0, w0 - PATCH_SIZE_LEVEL0 + 1, PATCH_SIZE_LEVEL0)
    ys = np.arange(0, h0 - PATCH_SIZE_LEVEL0 + 1, PATCH_SIZE_LEVEL0)
    mh, mw = mask.shape
    patch_size_mask = max(1, round(PATCH_SIZE_LEVEL0 * scale))
    coords = []
    for y in ys:
        my = int(y * scale)
        my_end = min(mh, my + patch_size_mask)
        if my >= mh:
            continue
        for x in xs:
            mx = int(x * scale)
            mx_end = min(mw, mx + patch_size_mask)
            if mx >= mw:
                continue
            region = mask[my:my_end, mx:mx_end]
            if region.size == 0:
                continue
            if region.mean() >= TISSUE_FRACTION_THRESHOLD:
                coords.append((x, y))
    return np.array(coords, dtype=np.int64)


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
    print(f"PORPOISE 공식 CSV 기준 슬라이드 {len(slide_ids)}개 처리")

    for i, slide_id in enumerate(slide_ids):
        out_path = OUT_DIR / f"{slide_id}.pt"
        if out_path.exists():
            continue
        wsi_path = WSI_ROOT / f"{slide_id}.svs"
        if not wsi_path.exists():
            print(f"  [경고] SVS 없음: {slide_id}")
            continue

        slide = openslide.OpenSlide(str(wsi_path))
        mask, scale = _tissue_mask(slide)
        coords = _generate_patch_coords(slide, mask, scale)
        if len(coords) == 0:
            print(f"  [경고] 조직 패치 0개(마스크 실패 가능성): {slide_id}")
            slide.close()
            continue

        feats = []
        batch_imgs = []
        for x, y in coords:
            region = slide.read_region((int(x), int(y)), 0, (PATCH_SIZE_LEVEL0, PATCH_SIZE_LEVEL0)).convert("RGB")
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
        print(f"  [{i+1}/{len(slide_ids)}] {slide_id}: {out.shape} (조직 패치 {len(coords)}개) -> {out_path}")

    print(f"\n완료 — {OUT_DIR} 확인.")


if __name__ == "__main__":
    main()
