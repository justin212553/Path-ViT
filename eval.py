"""
CAMELYON17 WSI(노드) 단위 MIL 평가 스크립트
- WSI 1장의 모든 패치를 한 번에 넣어 attention pooling 후 WSI 단위 분류
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


def evaluate_wsi_level(
    checkpoint: str,
    cfg: Config | None = None,
    split: str = "test",
    save_vis: bool = False,
    vis_dir: str = "heatmaps",
):
    if cfg is None:
        cfg = Config()
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    dataset = CAMELYON17NodeDataset(cfg.data, split=split, max_patches=cfg.data.max_patches)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

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
            patches  = batch["patches"].squeeze(0)                              # (N, 3, H, W) — CPU 유지
            coords   = batch["coords"].squeeze(0).to(device, non_blocking=True) # (N, 2)
            label    = int(batch["label"].item())
            slide_id = f"{batch['patient_id'][0]}_node_{int(batch['node'].item())}"

            orig_coords_np = batch["coords"].squeeze(0).numpy().copy()  # 원본 좌표 (thumbnail 크롭용)
            coords[:, 0] -= coords[:, 0].min()
            coords[:, 1] -= coords[:, 1].min()
            norm_coords_np = coords.cpu().numpy()

            patch_tokens          = _encode_patches_chunked(model.cnn, patches, chunk_size, device)
            ctx_tokens            = model.vit(patch_tokens, coords)
            wsi_embed, attn_weights = model.attn_pool(ctx_tokens)
            wsi_logits            = model.classifier(wsi_embed.unsqueeze(0))

            score = torch.softmax(wsi_logits, dim=-1)[0, 1].float().item()
            pred  = int(score >= threshold)
            print(f"  {slide_id}: GT={'N1+' if label else 'N0'}  score={score:.3f}  pred={'N1+' if pred else 'N0'}")

            all_scores.append(score)
            all_labels.append(label)

            if save_vis:
                from utils.visualize import save_heatmap
                save_heatmap(
                    heatmap=attn_weights.float().cpu().numpy(),
                    coords=batch["coords"].squeeze(0).numpy(),
                    slide_id=slide_id,
                    label=label,
                    score=score,
                    out_dir=vis_dir,
                )

    metrics = compute_patch_metrics(
        np.array(all_scores),
        np.array(all_labels),
    )
    print("\n=== WSI-Level Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "val"],
                        help="평가에 사용할 split (기본: test — held-out 전체 셋)")
    parser.add_argument("--vis", action="store_true",
                        help="attention 히트맵 시각화 저장")
    parser.add_argument("--vis-dir", type=str, default="heatmaps",
                        help="시각화 저장 디렉토리")
    args = parser.parse_args()

    evaluate_wsi_level(args.checkpoint, split=args.split, save_vis=args.vis, vis_dir=args.vis_dir)
