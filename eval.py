"""
TCGA-PAAD / CPTAC-PDA WSI 생존(OS) 예측 평가 스크립트
- 환자(case) 단위: 보유한 모든 슬라이드 임베딩을 평균 풀링해 risk score 1개 산출
- 지표: concordance index (c-index, utils/metrics.py)
- cross-dataset 검증(train.py)에서 이미 매 epoch val 지표가 나오므로, 이 스크립트는 주로
  저장된 체크포인트에 대한 사후 상세 평가(환자별 risk 출력, --vis 히트맵)용으로 쓴다.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from data.dataset import WSISurvivalDataset
from data.fit_clusters import CENTROIDS_DIR
from models import LateFusionViT, PatchViT
from utils.metrics import compute_survival_metrics


def _identity_collate(batch: list) -> list:
    """batch_size=1 전제 — DataLoader가 환자 1명의 슬라이드 리스트를 그대로 통과시키도록 함."""
    return batch[0]


def evaluate_survival(
    checkpoint: str,
    cfg: Config | None = None,
    dataset: str = "cptac",
    save_vis: bool = False,
    vis_dir: str = "heatmaps",
    image_mode: bool = False,
    fusion: bool = False,
):
    if cfg is None:
        cfg = Config()
    cfg.data.precomputed = not image_mode
    if fusion and not cfg.data.precomputed:
        raise ValueError("--fusion은 precomputed(features.pt) 모드에서만 지원됩니다. --image와 함께 사용 불가.")
    device = torch.device(cfg.train.device)

    ds     = WSISurvivalDataset(cfg.data, dataset=dataset)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate)

    if fusion:
        centroids_path = Path(__file__).parent / CENTROIDS_DIR
        if not centroids_path.exists():
            raise FileNotFoundError(
                f"cluster_centroids.pt 없음: {centroids_path}\n"
                "  먼저 실행: python -m data.fit_clusters"
            )
        cluster_centroids = torch.load(centroids_path, map_location="cpu")
        model = LateFusionViT(cfg.model, cluster_centroids, precomputed=cfg.data.precomputed).to(device)
    else:
        model = PatchViT(cfg.model, precomputed=cfg.data.precomputed).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    chunk_size = cfg.train.cnn_chunk_size
    all_risks, all_times, all_events = [], [], []

    with torch.no_grad():
        for patient_slides in loader:
            if len(patient_slides) == 0:
                continue

            slide_embeds, slide_vis = [], []
            for slide in patient_slides:
                coords = slide["coords"].to(device, non_blocking=True)  # (N, 2) — 이미 0-기반 정규화됨

                if "features" in slide:
                    out = model(coords, features=slide["features"])
                else:
                    out = model(coords, patch_paths=slide["patch_paths"],
                                 transform=ds.transform, chunk_size=chunk_size)
                slide_embeds.append(out["embed"])
                if save_vis:
                    slide_vis.append((slide["slide_id"], out["attn_weights"], slide["coords"]))

            patient_embed = torch.stack(slide_embeds).mean(dim=0)
            risk = model.risk_head(patient_embed.unsqueeze(0)).view(1).float().item()

            case_id  = patient_slides[0]["case_id"]
            os_time  = float(patient_slides[0]["OS_time"].item())
            os_event = int(patient_slides[0]["OS_event"].item())
            print(f"  {case_id}: OS_time={os_time:.1f}  OS_event={os_event}  risk={risk:.3f}  "
                  f"n_slides={len(patient_slides)}")

            all_risks.append(risk)
            all_times.append(os_time)
            all_events.append(os_event)

            if save_vis:
                from utils.visualize import save_heatmap
                for slide_id, attn_weights, coords_cpu in slide_vis:
                    save_heatmap(
                        heatmap=attn_weights.float().cpu().numpy(),
                        coords=coords_cpu.numpy(),
                        slide_id=slide_id,
                        label=os_event,
                        score=risk,
                        out_dir=vis_dir,
                    )

    metrics = compute_survival_metrics(
        np.array(all_risks),
        np.array(all_times),
        np.array(all_events),
    )
    print("\n=== WSI Survival Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--dataset", type=str, default="cptac",
                        choices=["tcga", "cptac"],
                        help="평가에 사용할 데이터셋 코호트 전체 (기본: cptac) — "
                             "cross-dataset 평가이므로 보통 checkpoint를 학습할 때 쓰지 않은 쪽을 지정한다")
    parser.add_argument("--vis", action="store_true",
                        help="attention 히트맵 시각화 저장 (슬라이드 단위)")
    parser.add_argument("--vis-dir", type=str, default="heatmaps",
                        help="시각화 저장 디렉토리")
    parser.add_argument("--image", action="store_true",
                        help="패치 jpg/png를 매 forward마다 ResNet50으로 직접 인코딩 "
                             "(기본: data/extract_features.py로 사전 추출한 features.pt 사용)")
    parser.add_argument("--fusion", action="store_true",
                        help="LateFusionViT(ViT+ABMIL + Cluster Histogram) 체크포인트 평가. "
                             "data/fit_clusters.py로 생성한 cluster_centroids.pt 필요. "
                             "--image와 함께 사용 불가.")
    args = parser.parse_args()

    evaluate_survival(args.checkpoint, dataset=args.dataset,
                       save_vis=args.vis, vis_dir=args.vis_dir, image_mode=args.image,
                       fusion=args.fusion)
