"""
CAMELYON17 패치 레벨 평가 스크립트
- WSI 1장의 모든 패치를 transformer에 한 번에 넣어 per-patch 분류
- 체크포인트에 저장된 Youden's J threshold 사용
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from data.patch_dataset import CAMELYON17NodeDataset
from models import PatchViT
from utils.metrics import compute_patch_metrics


def _find_wsi_path(slide_id: str, wsi_eval_dir: str = "data/wsi_eval") -> Optional[Path]:
    """slide_id로부터 WSI .tif 경로 탐색 (patient_XXX_node_Y → wsi_eval/patient_XXX/)."""
    parts = slide_id.split("_node_")
    if len(parts) != 2:
        return None
    candidate = Path(wsi_eval_dir) / parts[0] / f"{slide_id}.tif"
    return candidate if candidate.exists() else None


def _load_wsi_thumbnail_crop(
    wsi_path: Path,
    orig_coords: np.ndarray,
    patch_size: int = 256,
    max_thumb: int = 2048,
) -> Optional[np.ndarray]:
    """WSI 썸네일에서 패치 영역(bounding box)만 잘라 반환."""
    try:
        import openslide
        slide = openslide.OpenSlide(str(wsi_path))
        wsi_w, wsi_h = slide.level_dimensions[0]

        scale = min(max_thumb / wsi_w, max_thumb / wsi_h, 1.0)
        thumb_w = max(1, int(wsi_w * scale))
        thumb_h = max(1, int(wsi_h * scale))
        thumbnail = np.array(slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB"))
        slide.close()

        r_min = int(orig_coords[:, 0].min())
        r_max = int(orig_coords[:, 0].max()) + 1
        c_min = int(orig_coords[:, 1].min())
        c_max = int(orig_coords[:, 1].max()) + 1

        py0 = max(0, int(r_min * patch_size * scale))
        py1 = min(thumb_h, int(r_max * patch_size * scale))
        px0 = max(0, int(c_min * patch_size * scale))
        px1 = min(thumb_w, int(c_max * patch_size * scale))

        crop = thumbnail[py0:py1, px0:px1]
        return crop if crop.size > 0 else None
    except Exception as e:
        print(f"  WSI 썸네일 로드 실패 ({wsi_path.name}): {e}")
        return None


def _encode_patches_chunked(
    cnn: nn.Module, patches: torch.Tensor, chunk_size: int, device: torch.device
) -> torch.Tensor:
    """CNN을 chunk_size 단위로 CPU→GPU 이동하며 실행 (대형 WSI OOM 방지)."""
    return torch.cat([
        cnn(patches[i : i + chunk_size].to(device, non_blocking=True))
        for i in range(0, patches.shape[0], chunk_size)
    ])


def evaluate_patch_level(
    checkpoint: str,
    cfg: Config | None = None,
    split: str = "eval",
    save_vis: bool = False,
    vis_dir: str = "heatmaps",
):
    if cfg is None:
        cfg = Config()
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    dataset = CAMELYON17NodeDataset(
        cfg.data, split=split,
        val_ratio=cfg.data.val_ratio, eval_ratio=cfg.data.eval_ratio,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = PatchViT(cfg.model).to(device)
    ckpt  = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    threshold = ckpt.get("threshold", 0.5)
    print(f"  (checkpoint threshold: {threshold:.4f})")
    model.eval()

    chunk_size = cfg.train.cnn_chunk_size
    all_scores, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            patches      = batch["patches"].squeeze(0)                              # (N, 3, H, W) — CPU 유지
            coords       = batch["coords"].squeeze(0).to(device, non_blocking=True) # (N, 2)
            patch_labels = batch["patch_labels"].squeeze(0).numpy()                 # (N,)
            slide_id     = batch["slide_id"][0]

            orig_coords_np = batch["coords"].squeeze(0).numpy().copy()  # 원본 좌표 (thumbnail 크롭용)
            coords[:, 0] -= coords[:, 0].min()
            coords[:, 1] -= coords[:, 1].min()
            norm_coords_np = coords.cpu().numpy()

            patch_tokens = _encode_patches_chunked(model.cnn, patches, chunk_size, device)
            ctx_tokens   = model.vit(patch_tokens, coords)
            patch_logits = model.classifier(ctx_tokens)

            scores = torch.softmax(patch_logits, dim=-1)[:, 1].float().cpu().numpy()
            preds  = (scores >= threshold).astype(np.int64)
            acc    = float((preds == patch_labels).mean())
            print(f"  {slide_id}: {int(patch_labels.sum())}/{len(patch_labels)} tumor patches  acc={acc:.3f}")

            all_scores.append(scores)
            all_labels.append(patch_labels)

            if save_vis:
                from utils.visualize import save_dual_overlay
                thumbnail = None
                wsi_path = _find_wsi_path(slide_id, cfg.data.wsi_eval_dir)
                if wsi_path is not None:
                    thumbnail = _load_wsi_thumbnail_crop(wsi_path, orig_coords_np)
                else:
                    print(f"  WSI 없음, 배경 없이 렌더링: {slide_id}")
                save_dual_overlay(
                    scores=scores,
                    coords=norm_coords_np,
                    patch_labels=patch_labels,
                    slide_id=slide_id,
                    thumbnail=thumbnail,
                    out_dir=vis_dir,
                )

    metrics = compute_patch_metrics(
        np.concatenate(all_scores),
        np.concatenate(all_labels),
    )
    print("\n=== Patch-Level Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


if __name__ == "__main__":
    from pathlib import Path
    checkpoint = str(Path(__file__).parent / "models" / "checkpoint" / "camelyon_best.pt")
    print(f"checkpoint: {checkpoint}")

    evaluate_patch_level(checkpoint, split="eval", save_vis=True, vis_dir="heatmap")
