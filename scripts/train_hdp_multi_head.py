"""
HDP_Pretrain 확장 — scripts/train_hdp_pretrain_head.py는 PanNuke pancreas subset(195장)의
핵 라벨 중 category 0(Neoplastic) 하나만 "종양 함량"으로 회귀했다. 사용자 질문(2026-09-04):
"패치 종양비율 말고 다른 정보(세포 밀도 등)로도 만들 수 있나?" — 같은 PanNuke 데이터에 이미
들어있는 나머지 정보를 그대로 재사용해 5개 타깃을 한 번에 회귀하는 head를 학습한다(새 데이터
불필요, feature 추출도 1번만).

PanNuke pancreas subset 실측 category id: {0,1,2,4} (3=Dead는 이 subset엔 아예 없음 — 정상,
공개 데이터셋 자체의 특성).
  neoplastic_frac:   category 0 면적 비율 (기존 종양 함량, cross-check용 재학습)
  inflammatory_frac: category 1 면적 비율 — 면역세포 침윤(TIL). PDAC 예후 인자로 문헌에 흔히
                      보고됨(면역원성 종양미세환경일수록 예후가 좋다는 방향).
  connective_frac:   category 2 면적 비율 — 결합조직/스트로마. PDAC은 desmoplastic stroma
                      (과다 섬유화 반응)로 유명해 스트로마 비율 자체가 독립적 예후 인자로
                      보고된 바 있음.
  epithelial_frac:   category 4 면적 비율 — 비종양 상피(정상 도관 등).
  cellularity_frac:  전체 카테고리 핵 면적 합 / patch 면적 — "이 patch가 얼마나 세포로
                      빽빽한가"(세포 밀도의 면적 기반 proxy, 카테고리 무관).
  nucleus_density:   instance 개수 / patch 면적, 학습셋 99th percentile로 나눠 [0,1] 근사 —
                      cellularity_frac과 다른 축(핵이 작고 빽빽 vs 크고 듬성)을 구분하기 위한
                      개수 기반 밀도.

train_hdp_pretrain_head.py와 동일하게 uni2native와 배율을 맞춘(resmatch) 버전으로 feature를
뽑는다 — 이 project가 실제 적용 대상으로 삼는 20x/0.5um에 가장 가깝게 보정된 쪽.

사용법:
    python -m scripts.train_hdp_multi_head
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
IMG_SIZE = 256
CATEGORY_NAMES = {0: "neoplastic_frac", 1: "inflammatory_frac", 2: "connective_frac", 4: "epithelial_frac"}
TARGET_NAMES = list(CATEGORY_NAMES.values()) + ["cellularity_frac", "nucleus_density"]
UNI2NATIVE_DOWNSAMPLE_FACTOR = 2
OUT_PATH = _ROOT / "data" / "hdp_multi_head.pt"


def _simulate_uni2native_resolution(img: Image.Image, factor: int = UNI2NATIVE_DOWNSAMPLE_FACTOR) -> Image.Image:
    w, h = img.size
    small = img.resize((w // factor, h // factor), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


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


def _compute_targets(row: dict) -> dict:
    frac_by_cat = {c: 0.0 for c in CATEGORY_NAMES}
    total_area = 0
    n_instances = len(row["instances"])
    for inst, cat in zip(row["instances"], row["categories"]):
        mask = np.array(Image.open(io.BytesIO(inst["bytes"])))
        area = int((mask > 0).sum())
        total_area += area
        if cat in frac_by_cat:
            frac_by_cat[cat] += area
    out = {CATEGORY_NAMES[c]: v / (IMG_SIZE * IMG_SIZE) for c, v in frac_by_cat.items()}
    out["cellularity_frac"] = total_area / (IMG_SIZE * IMG_SIZE)
    out["nucleus_density_raw"] = n_instances / (IMG_SIZE * IMG_SIZE)  # 나중에 percentile로 정규화
    return out


class MultiHead(nn.Module):
    """TumorContentHead와 동일 구조(LayerNorm+Linear[+hidden]+Sigmoid)를 다중 출력으로 일반화 —
    기존 models/tumor_content_head.py(단일 스칼라 전제로 다른 스크립트들이 이미 씀)는 안 건드리고
    이 실험 전용으로 로컬 정의."""

    def __init__(self, in_dim: int, n_targets: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, n_targets), nn.Sigmoid(),
            )
        else:
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, n_targets), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def _extract_features(images: list[Image.Image], device) -> torch.Tensor:
    from data.patch_utils import UNI2_NATIVE_PATCH_TRANSFORM
    from models.uni2_encoder import UNI2hEncoder

    encoder = UNI2hEncoder(embed_dim=1, with_backbone=True).to(device)
    encoder.eval()
    encoder.requires_grad_(False)

    images = [_simulate_uni2native_resolution(im) for im in images]
    chunks = []
    batch_size = 32
    for i in range(0, len(images), batch_size):
        batch = torch.stack([UNI2_NATIVE_PATCH_TRANSFORM(im) for im in images[i:i + batch_size]]).to(device)
        raw = encoder.backbone(batch)
        chunks.append(raw.cpu())
        print(f"    feature 추출 {min(i + batch_size, len(images))}/{len(images)}", end="\r")
    print()
    return torch.cat(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=150)
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

    print("[2/4] 5개 타깃 계산")
    target_dicts = [_compute_targets(r) for r in rows]
    nucleus_density_raw = np.array([d["nucleus_density_raw"] for d in target_dicts], dtype=np.float32)
    p99 = np.percentile(nucleus_density_raw, 99)
    nucleus_density = np.clip(nucleus_density_raw / max(p99, 1e-6), 0, 1)
    targets = np.stack([
        [d[name] for name in CATEGORY_NAMES.values()] + [d["cellularity_frac"]] for d in target_dicts
    ], axis=0)
    targets = np.concatenate([targets, nucleus_density[:, None]], axis=1).astype(np.float32)  # (N, 6)
    for j, name in enumerate(TARGET_NAMES):
        col = targets[:, j]
        print(f"  {name:20s} mean={col.mean():.4f} std={col.std():.4f} min={col.min():.4f} max={col.max():.4f}")

    print("[3/4] UNI2-h feature 추출 (uni2native 배율 보정)")
    images = [Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB") for r in rows]
    features = _extract_features(images, device)

    print("[4/4] 다중출력 회귀 head 학습")
    idx = np.arange(len(rows))
    train_idx, val_idx = train_test_split(idx, test_size=args.val_frac, random_state=args.seed)

    x_train, y_train = features[train_idx].to(device), torch.from_numpy(targets[train_idx]).to(device)
    x_val, y_val = features[val_idx].to(device), torch.from_numpy(targets[val_idx]).to(device)

    head = MultiHead(in_dim=features.shape[1], n_targets=len(TARGET_NAMES), hidden_dim=args.hidden_dim).to(device)
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
            val_corrs = [float(np.corrcoef(val_pred[:, j].cpu().numpy(), y_val[:, j].cpu().numpy())[0, 1])
                         for j in range(len(TARGET_NAMES))]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            best_val_corrs = val_corrs
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            corr_str = " ".join(f"{n}={c:.3f}" for n, c in zip(TARGET_NAMES, val_corrs))
            print(f"  epoch {epoch:3d} | train_loss={loss.item():.4f} | val_loss={val_loss:.4f} | {corr_str}")

    print(f"\n최종 best_val_loss={best_val_loss:.4f}")
    for n, c in zip(TARGET_NAMES, best_val_corrs):
        print(f"  best-checkpoint val_corr[{n}] = {c:.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state, "hidden_dim": args.hidden_dim, "in_dim": features.shape[1],
        "target_names": TARGET_NAMES, "nucleus_density_p99": float(p99),
        "best_val_loss": best_val_loss, "n_train": len(train_idx), "n_val": len(val_idx),
    }, OUT_PATH)
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
