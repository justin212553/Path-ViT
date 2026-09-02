"""
HDP_Pretrain 1단계 — PanNuke(pancreas subset)의 핵 단위 라벨(Neoplastic/Inflammatory/
Connective/Dead/Epithelial)을 patch 단위 "종양 함량 비율"로 집계해, frozen UNI2-h(공식 스펙)
feature 위에 이 비율을 예측하는 작은 회귀 head를 학습시킨다.

[배경, 2026-09-01] 지금까지(k-means 군집화)는 "이 patch가 종양인지"에 대한 라벨이 TCGA/CPTAC
어디에도 없어서 비지도 군집 통계로 우회했다 — 여러 아키텍처(co-attention/ABMIL/sharpening/
군집 통계 hard·soft)가 전부 M7과 통계적으로 구분 안 되는 결과를 냈다. 사용자가 "진짜 라벨로
학습시켜보자"고 결정 — PanNuke는 pancreas를 포함한 19개 조직 유형에 핵 단위 라벨이 있는
공개 데이터셋(연구 목적 CC-BY-NC-SA-4.0)이라, 이걸로 "patch 안에 종양세포가 얼마나 있는지"의
진짜 supervised proxy를 만들 수 있다.

[방법]
  1. PanNuke pancreas 이미지(256x256, 40x/0.25um px) 각각에서, 각 핵 instance mask의 면적을
     category별로 합산해 neoplastic_area_fraction = (Neoplastic 핵 면적 합) / (256*256)을 계산.
     이게 이 patch의 "종양 함량" 회귀 타겟(0~1)이다.
  2. 같은 이미지를 이 프로젝트가 실제로 쓰는 UNI2-h 공식 스펙 전처리(data/patch_utils.py::
     UNI2_NATIVE_PATCH_TRANSFORM, Resize(256)+ImageNet 정규화)로 이 프로젝트가 쓰는 것과 동일한
     UNI2-h backbone(models/uni2_encoder.py)에 통과시켜 1536차원 frozen feature를 뽑는다 —
     PanNuke는 40x/0.25um인데 우리 uni2native는 20x/0.5um라 완전히 같은 물리 해상도는 아니다
     (약 2배 차이, PanNuke 원본이 더 좁은 시야 — 이 mismatch는 알려진 한계로 남겨둔다).
  3. frozen feature -> Linear(1536, 1)(+ 필요시 작은 hidden) 회귀 head를 MSE로 학습(train/val
     split, patient 단위가 아니라 image 단위 — PanNuke엔 patient ID가 없음).
  4. 학습된 head 가중치를 저장 — 다음 단계(scripts/apply_hdp_pretrain_head.py)에서 이 head를
     TCGA-PAAD/CPTAC-PDA의 이미 추출된 uni2native feature(raw 이미지 재처리 없이) 위에 그대로
     적용해 우리 코호트 전체의 patch별 종양 함량 스칼라를 뽑는다.

사용법:
    python -m scripts.train_hdp_pretrain_head
    python -m scripts.train_hdp_pretrain_head --hidden-dim 64 --epochs 50
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils import load_env

PANNUKE_DIR = Path.home() / "AppData/Local/Temp/claude" / \
    "d--wonse-Documents-Job-urban-datalab-PATH-ViT/31b48df2-b2a3-4986-995d-b8fef7cad784/scratchpad"
PANNUKE_FOLDS = ["pannuke_fold1.parquet", "pannuke_fold2.parquet", "pannuke_fold3.parquet"]
PANCREAS_TISSUE_ID = 12
NEOPLASTIC_CATEGORY = 0
IMG_SIZE = 256

OUT_DIR = _ROOT / "models" / "checkpoint"
OUT_PATH = OUT_DIR / "hdp_pretrain_tumor_content_head.pt"


def _load_pancreas_rows() -> list[dict]:
    rows = []
    for fname in PANNUKE_FOLDS:
        path = PANNUKE_DIR / fname
        if not path.exists():
            print(f"  [경고] {path} 없음, 건너뜀")
            continue
        pf = pq.ParquetFile(path)
        for i in range(pf.num_row_groups):
            tbl = pf.read_row_group(i, columns=["image", "instances", "categories", "tissue"])
            batch = tbl.to_pylist()
            rows.extend([r for r in batch if r["tissue"] == PANCREAS_TISSUE_ID])
        print(f"  {fname}: 누적 pancreas {len(rows)}장")
    return rows


def _neoplastic_fraction(row: dict) -> float:
    total_neoplastic_px = 0
    for inst, cat in zip(row["instances"], row["categories"]):
        if cat != NEOPLASTIC_CATEGORY:
            continue
        mask = np.array(Image.open(io.BytesIO(inst["bytes"])))
        total_neoplastic_px += int((mask > 0).sum())
    return total_neoplastic_px / (IMG_SIZE * IMG_SIZE)


from models.tumor_content_head import TumorContentHead as RegressionHead


@torch.no_grad()
def _extract_features(images: list[Image.Image], device) -> torch.Tensor:
    from data.patch_utils import UNI2_NATIVE_PATCH_TRANSFORM
    from models.uni2_encoder import UNI2hEncoder

    encoder = UNI2hEncoder(embed_dim=1, with_backbone=True).to(device)
    encoder.eval()
    encoder.requires_grad_(False)

    chunks = []
    batch_size = 32
    for i in range(0, len(images), batch_size):
        batch = torch.stack([UNI2_NATIVE_PATCH_TRANSFORM(im) for im in images[i:i + batch_size]]).to(device)
        raw = encoder.backbone(batch)  # (B, 1536) — utils/extract_features.py와 동일 관례(raw pooled, .proj 안 씀)
        chunks.append(raw.cpu())
        print(f"    feature 추출 {min(i + batch_size, len(images))}/{len(images)}", end="\r")
    print()
    return torch.cat(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden-dim", type=int, default=None, help="None(기본)=단순 선형회귀, 지정하면 1-hidden MLP.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_env()
    import os
    fixed_cert = Path(os.environ.get("CONDA_PREFIX", "")) / "Library/ssl/cacert.pem"
    if fixed_cert.exists():
        os.environ["SSL_CERT_FILE"] = str(fixed_cert)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[1/4] PanNuke pancreas subset 로드")
    rows = _load_pancreas_rows()
    print(f"  총 {len(rows)}장")

    print("[2/4] neoplastic area fraction 라벨 계산")
    targets = np.array([_neoplastic_fraction(r) for r in rows], dtype=np.float32)
    print(f"  target 분포: mean={targets.mean():.4f} std={targets.std():.4f} "
          f"min={targets.min():.4f} max={targets.max():.4f} (0인 이미지 {int((targets == 0).sum())}장)")

    print("[3/4] UNI2-h feature 추출")
    images = [Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB") for r in rows]
    features = _extract_features(images, device)  # (N, 1536)

    print("[4/4] 회귀 head 학습")
    idx = np.arange(len(rows))
    train_idx, val_idx = train_test_split(idx, test_size=args.val_frac, random_state=args.seed)

    x_train, y_train = features[train_idx].to(device), torch.from_numpy(targets[train_idx]).to(device)
    x_val, y_val = features[val_idx].to(device), torch.from_numpy(targets[val_idx]).to(device)

    head = RegressionHead(in_dim=features.shape[1], hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        head.train()
        optimizer.zero_grad()
        pred = head(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()

        head.eval()
        with torch.no_grad():
            val_pred = head(x_val)
            val_loss = loss_fn(val_pred, y_val).item()
            val_corr = float(np.corrcoef(val_pred.cpu().numpy(), y_val.cpu().numpy())[0, 1])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d} | train_loss={loss.item():.4f} | val_loss={val_loss:.4f} | val_corr={val_corr:.4f}")

    print(f"\n최종 best_val_loss={best_val_loss:.4f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "hidden_dim": args.hidden_dim,
        "in_dim": features.shape[1],
        "best_val_loss": best_val_loss,
        "n_train": len(train_idx), "n_val": len(val_idx),
    }, OUT_PATH)
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
