"""
패치 jpg/png → 타일 augmentation(seed 고정, 1회성) → frozen ResNet50(Lunit SwAV) feature 사전 추출.

utils/extract_features.py와 산출물이 다른 별도 스크립트다: 기존 features.pt(원본 패치 feature)는
그대로 두고, 증강된 버전을 features_aug.pt로 새로 저장한다(롤백 가능, 기존 학습 파이프라인에
영향 없음) — utils/extract_features_stain_norm.py와 정확히 같은 관례.

배경: 레퍼런스(Leeyoungsup/pancreatic_cancer_pathology)는 학습 시 타일마다 매 epoch 새로
RandomFlip/ColorJitter/GaussianBlur를 적용해 frozen backbone을 통과시킨다(data/patch_utils.py::
PATCH_TRANSFORM_AUGMENTED와 동일한 조합). 우리는 features.pt를 한 번만 추출해 캐싱해왔기 때문에
이 augmentation이 구조적으로 없었다. 매 epoch 실시간으로 재현하면(local 24 img/s 실측 기준)
런 1개당 20시간 안팎이라 비현실적이고, HPC도 할당량(AssocGrpBillingMinutes)에 걸렸다 — 그래서
"매 epoch 다른 view"는 포기하고, 무작위 augmentation을 슬라이드당 딱 1벌만 미리 뽑아 한 번만
추출해두는 절충안을 쓴다(train.py --tile-augment가 train split에서 이 산출물을 읽는다).
"증강 없음"보다는 낫지만 진짜 epoch별 augmentation의 정규화 효과에는 못 미친다는 한계를 분명히
인지하고 쓴다.

대상 슬라이드는 --split="all"(기본, 코호트 전체)로 **seed와 무관하게 한 번만** 뽑는다 — 우리가
쓰는 6:2:2 stratified split은 시드마다 어떤 case가 train/val/test에 들어가는지가 달라지므로
(seed 42/84/126 3시드가 표준 관례), 특정 시드의 train split만 좁혀서 뽑으면 다른 시드로 돌릴 때
그 시드의 train에 새로 들어온 슬라이드에 features_aug.pt가 없어 조용히 원본으로 폴백해버린다.
코호트 전체를 미리 한 번 뽑아두면 어떤 시드로 돌리든(train.py --seed) 그 시드의 train 슬라이드는
전부 이미 augmented feature가 준비돼 있다 — 결과적으로 3시드 전부 augmented로 돌리는 총비용도
시드별로 따로 뽑는 것보다 더 싸다(시드 3개 train 합집합 < 시드 3개 train 총합).

출력:
    <patches_root>/<slide_id>/features_aug.pt   (N_patches, 2048) float32
    행 순서 = data.patch_utils.list_patch_paths()와 동일한 정렬 순서

사용법:
    python -m utils.extract_features_augmented --dataset tcga --seed 42
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, literature_guided_gene_ids
from data.patch_utils import FEATURES_AUG_FILENAME, PATCH_TRANSFORM_AUGMENTED, list_patch_paths
from models.cnn_encoder import CNNEncoder
from utils import load_env, send_slack

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8  # extract_features.py의 resnet50 배치 크기와 동일(1024px 원본 입력 기준 실측치)


def _build_encoder() -> CNNEncoder:
    encoder = CNNEncoder(embed_dim=1, with_backbone=True).to(DEVICE)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


@torch.no_grad()
def _extract_node(encoder, patch_paths: list[Path]) -> torch.Tensor:
    chunks = []
    for i in range(0, len(patch_paths), BATCH_SIZE):
        batch = torch.stack([
            PATCH_TRANSFORM_AUGMENTED(Image.open(p).convert("RGB"))
            for p in patch_paths[i : i + BATCH_SIZE]
        ]).to(DEVICE, non_blocking=True)
        raw = encoder.backbone(batch)
        pooled = encoder.pool(raw)
        chunks.append(pooled.cpu())
    return torch.cat(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="tcga", choices=["tcga", "cptac"])
    parser.add_argument(
        "--split", type=str, default="all", choices=["train", "val", "test", "all"],
        help="기본 all(코호트 전체, seed 무관 — 권장). 특정 시드의 train만 좁히고 싶으면 "
             "train으로 바꾸고 --seed도 그 시드로 맞춰야 하지만, 다른 시드로 돌릴 때 커버리지가 "
             "안 맞을 수 있다(위 모듈 docstring 참조).",
    )
    parser.add_argument("--seed", type=int, default=42,
                         help="torch 전역 시드 — RandomHorizontalFlip 등의 확률적 선택을 재현 가능하게 "
                              "고정한다(--split train일 때는 6:2:2 split 자체도 이 시드를 따른다).")
    args = parser.parse_args()

    load_env()
    torch.manual_seed(args.seed)

    cfg = Config()
    cfg.data.seed = args.seed
    rna_gene_ids = literature_guided_gene_ids(1500)

    ds = WSISurvivalDataset(
        cfg.data, dataset=args.dataset, split=args.split,
        with_clinical=True, with_rna=True, rna_gene_ids=rna_gene_ids,
    )
    slide_ids = sorted(ds.items["slide_id"].unique())
    print(f"[{args.dataset}/{args.split}] 대상 슬라이드 {len(slide_ids)}개 (augmentation seed={args.seed})")

    encoder = _build_encoder()
    root = Path(getattr(cfg.data, f"patches_root_{args.dataset}")) / "tiles"

    start_time = datetime.now()
    try:
        from tqdm import tqdm
        slide_ids = tqdm(slide_ids, desc="augmented feature 추출", unit="slide")
    except ImportError:
        pass

    done, skipped = 0, 0
    for slide_id in slide_ids:
        slide_dir = root / slide_id
        out_path = slide_dir / FEATURES_AUG_FILENAME
        if out_path.exists():
            skipped += 1
            continue
        patch_paths = list_patch_paths(slide_dir)
        if not patch_paths:
            continue
        features = _extract_node(encoder, patch_paths)
        torch.save(features, out_path)
        done += 1

    elapsed = datetime.now() - start_time
    h, rem = divmod(int(elapsed.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    msg = (f"[{args.dataset}/{args.split}] 완료: {done}개 신규, {skipped}개 스킵(이미 존재) "
           f"-> {root}/<slide_id>/{FEATURES_AUG_FILENAME} — {h}h {m}m {s}s")
    print(msg)
    send_slack(f":white_check_mark: *augmented feature 추출 완료* {msg}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        load_env()
        send_slack(f":x: *augmented feature 추출 에러*\n```{type(e).__name__}: {e}```")
        raise
