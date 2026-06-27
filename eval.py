"""
CAMELYON17 WSI(노드) 단위 MIL 평가 스크립트
- WSI 1장의 모든 패치를 한 번에 넣어 attention pooling 후 WSI 단위 분류
- threshold는 eval 데이터 자체에서 Youden's J로 매번 새로 계산 (utils/metrics.py)
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from data.patch_dataset import CAMELYON17NodeDataset
from models import PatchViT
from utils.metrics import compute_patch_metrics


def _identity_collate(batch: list) -> list:
    """batch_size=1 전제 — DataLoader가 환자 1명의 노드 리스트를 그대로 통과시키도록 함."""
    return batch[0]


def evaluate_wsi_level(
    checkpoint: str,
    cfg: Config | None = None,
    split: str = "val",
    save_vis: bool = False,
    vis_dir: str = "heatmaps",
):
    if cfg is None:
        cfg = Config()
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    dataset = CAMELYON17NodeDataset(cfg.data, split=split)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_identity_collate)

    model = PatchViT(cfg.model).to(device)
    ckpt  = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    chunk_size = cfg.train.cnn_chunk_size
    all_scores, all_labels = [], []

    with torch.no_grad():
        for patient_nodes in loader:
            for node in patient_nodes:
                patch_paths = node["patch_paths"]                       # N개 경로 — 이미지는 model 내부에서 지연 로딩
                coords      = node["coords"].to(device, non_blocking=True) # (N, 2) — 이미 0-기반 정규화됨
                label       = int(node["label"].item())
                slide_id    = f"{node['patient_id']}_node_{node['node']}"

                out          = model(patch_paths, coords, dataset.transform, chunk_size=chunk_size)
                attn_weights = out["attn_weights"]

                score = torch.softmax(out["wsi_logits"], dim=-1)[0, 1].float().item()
                print(f"  {slide_id}: GT={'N1+' if label else 'N0'}  score={score:.3f}")

                all_scores.append(score)
                all_labels.append(label)

                if save_vis:
                    from utils.visualize import save_heatmap
                    save_heatmap(
                        heatmap=attn_weights.float().cpu().numpy(),
                        coords=node["coords"].numpy(),
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
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val"],
                        help="평가에 사용할 split (기본: val)")
    parser.add_argument("--vis", action="store_true",
                        help="attention 히트맵 시각화 저장")
    parser.add_argument("--vis-dir", type=str, default="heatmaps",
                        help="시각화 저장 디렉토리")
    args = parser.parse_args()

    evaluate_wsi_level(args.checkpoint, split=args.split, save_vis=args.vis, vis_dir=args.vis_dir)
