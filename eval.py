"""
TCGA-PAAD / CPTAC-PDA WSI 생존(OS) 예측 평가 스크립트
- 환자(case) 단위: 보유한 모든 슬라이드 임베딩을 평균 풀링해 risk score 1개 산출
- 지표: c-index, HR(95% CI), log-rank p, time-dependent AUC (utils/metrics.py)
- train.py가 학습 종료 시 best checkpoint로 test set을 한 번 평가해주지만, 이 스크립트는
  임의의 checkpoint/split 조합에 대한 사후 상세 평가(환자별 risk 출력, --vis 히트맵)용으로 쓴다.
  time-dependent AUC의 censoring 분포 추정을 위해 같은 dataset의 train split도 함께 로드한다.
- --external-dataset을 주면, --dataset/--split(internal)과 별도로 학습에 전혀 쓰이지 않은
  코호트 전체(split="all")도 같은 checkpoint로 평가한다(internal/external 동시 수행).
  censoring 분포는 두 평가 모두 checkpoint의 학습 코호트(--dataset)의 train split 기준을 쓴다.
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
from utils.metrics import compute_survival_metrics, compute_time_dependent_auc


def _identity_collate(batch: list) -> list:
    """batch_size=1 전제 — DataLoader가 환자 1명의 슬라이드 리스트를 그대로 통과시키도록 함."""
    return batch[0]


def _times_events(ds: WSISurvivalDataset) -> tuple[np.ndarray, np.ndarray]:
    """모델 forward 없이 dataset에서 환자별 (OS_time, OS_event)만 뽑는다.

    time-dependent AUC의 censoring 분포(train split 기준) 추정에만 필요하므로,
    비싼 CNN/ViT forward 없이 라벨만 바로 읽는다.
    """
    times, events = [], []
    for case_id in ds.cases:
        row = ds.items[ds.items["case_id"] == case_id].iloc[0]
        times.append(float(row["OS_time"]))
        events.append(int(row["OS_event"]))
    return np.array(times), np.array(events)


def _run_patients(
    model, ds: WSISurvivalDataset, device, chunk_size: int, save_vis: bool, vis_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ds의 환자 전원에 대해 risk score를 계산해 (risks, times, events)를 반환한다."""
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate)
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

    return np.array(all_risks), np.array(all_times), np.array(all_events)


def _print_results(title: str, metrics: dict, td_auc: dict) -> None:
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("--- Time-dependent AUC (12/24/36개월) ---")
    for k, v in td_auc.items():
        print(f"  {k}: {v:.4f}")


def evaluate_survival(
    checkpoint: str,
    cfg: Config | None = None,
    dataset: str = "cptac",
    split: str = "test",
    external_dataset: str | None = None,
    save_vis: bool = False,
    vis_dir: str = "heatmaps",
    image_mode: bool = False,
    fusion: bool = False,
):
    """
    checkpoint를 dataset/split(internal)으로 평가한다. external_dataset을 주면, 학습에 전혀
    쓰이지 않은 그 코호트 전체(split="all")도 같은 checkpoint로 추가 평가한다(internal test와
    external test를 한 번의 호출로 동시에 수행).

    Returns: {"internal": {...metrics+td_auc}, "external": {...} | None}
    """
    if cfg is None:
        cfg = Config()
    cfg.data.precomputed = not image_mode
    if fusion and not cfg.data.precomputed:
        raise ValueError("--fusion은 precomputed(features.pt) 모드에서만 지원됩니다. --image와 함께 사용 불가.")
    if external_dataset is not None and (dataset == "both" or external_dataset == dataset):
        raise ValueError(
            f"--external-dataset={external_dataset}은 --dataset={dataset}과 겹칩니다 — "
            "external test는 checkpoint 학습에 전혀 쓰이지 않은 코호트여야 합니다."
        )
    device = torch.device(cfg.train.device)

    ds = WSISurvivalDataset(cfg.data, dataset=dataset, split=split)

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

    # time-dependent AUC의 censoring 분포는 이 checkpoint가 학습에 쓴 train split 기준으로 추정한다
    # (internal/external 평가 모두 동일 기준을 쓴다 — 모델 forward 없이 라벨만 필요하므로 가볍다).
    train_ds = ds if split == "train" else WSISurvivalDataset(cfg.data, dataset=dataset, split="train")
    train_times, train_events = _times_events(train_ds)

    print(f"\n--- Internal ({dataset}/{split}, n={len(ds)}) ---")
    risks, times, events = _run_patients(model, ds, device, chunk_size, save_vis, vis_dir)
    metrics = compute_survival_metrics(risks, times, events)
    td_auc  = compute_time_dependent_auc(train_times, train_events, times, events, risks)
    _print_results(f"Internal Test ({dataset}/{split})", metrics, td_auc)
    result = {"internal": {**metrics, **td_auc}, "external": None}

    if external_dataset is not None:
        ext_ds = WSISurvivalDataset(cfg.data, dataset=external_dataset, split="all")
        print(f"\n--- External ({external_dataset}, 전체 코호트, n={len(ext_ds)}) ---")
        ext_vis_dir = f"{vis_dir}_external" if save_vis else vis_dir
        ext_risks, ext_times, ext_events = _run_patients(
            model, ext_ds, device, chunk_size, save_vis, ext_vis_dir,
        )
        ext_metrics = compute_survival_metrics(ext_risks, ext_times, ext_events)
        ext_td_auc  = compute_time_dependent_auc(train_times, train_events, ext_times, ext_events, ext_risks)
        _print_results(f"External Test ({external_dataset}, 전체 코호트)", ext_metrics, ext_td_auc)
        result["external"] = {**ext_metrics, **ext_td_auc}

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--dataset", type=str, default="cptac",
                        choices=["tcga", "cptac", "both"],
                        help="평가에 사용할 데이터셋 (기본: cptac). checkpoint를 학습할 때와 "
                             "같은 --dataset을 줘야 동일한 6:2:2 split을 재현한다.")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="평가할 split (기본: test — held-out 최종 평가용)")
    parser.add_argument("--external-dataset", type=str, default=None,
                        choices=["tcga", "cptac"],
                        help="--dataset/--split(internal)과 별도로, checkpoint 학습에 전혀 쓰이지 "
                             "않은 코호트 전체를 external test로 함께 평가한다 (예: --dataset cptac "
                             "--split test --external-dataset tcga). 미지정 시 internal만 평가.")
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

    evaluate_survival(args.checkpoint, dataset=args.dataset, split=args.split,
                       external_dataset=args.external_dataset,
                       save_vis=args.vis, vis_dir=args.vis_dir, image_mode=args.image,
                       fusion=args.fusion)
