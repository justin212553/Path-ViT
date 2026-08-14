"""
uni2native(우리 파이프라인, 256px@0.5MPP 재타일링) vs uni2official(MahmoodLab 공식 배포) 간,
같은 슬라이드에서 뽑힌 패치 수를 비교한다 — 둘 다 명목상 같은 스펙(256px@20x)인데 실제 tissue
segmentation 알고리즘 차이로 패치 수/커버 면적이 얼마나 다른지 확인하는 진단용.

scripts/reconcile_uni2native_features.py를 먼저 돌려야 한다(features_uni2native.pt가
data/patches_{tcga,cptac}/tiles/<slide>/ 아래 있어야 함).

사용법: python scripts/compare_native_vs_official_coverage.py
"""
import statistics
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main():
    for tag, root in [("tcga", _ROOT / "data" / "patches_tcga" / "tiles"),
                       ("cptac", _ROOT / "data" / "patches_cptac" / "tiles")]:
        native_slides = {p.parent.name for p in root.glob("*/features_uni2native.pt")}
        official_slides = {p.parent.name for p in root.glob("*/features_uni2official.pt")}
        shared = sorted(native_slides & official_slides)
        print(f"{tag}: native {len(native_slides)}개, official {len(official_slides)}개, "
              f"겹치는 슬라이드 {len(shared)}개")
        if not shared:
            continue

        ratios = []
        for slide_id in shared[:20]:  # 샘플 20개만
            n_native = torch.load(root / slide_id / "features_uni2native.pt", weights_only=True).shape[0]
            n_official = torch.load(root / slide_id / "features_uni2official.pt", weights_only=True).shape[0]
            ratio = n_native / n_official if n_official > 0 else float("nan")
            ratios.append(ratio)
            print(f"  {slide_id}: native={n_native}, official={n_official}, 비율(native/official)={ratio:.2f}")

        print(f"  -> 평균 비율(native/official) = {statistics.mean(ratios):.2f} (표본 {len(ratios)}개)")


if __name__ == "__main__":
    main()
