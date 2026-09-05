"""
scripts/viz_tumor_content_grid_tsne.py(종양함량 하나만)를 scripts/train_hdp_multi_head.py의
6채널(neoplastic/inflammatory/connective/epithelial/cellularity/nucleus_density) 전부로
확장한다. 사용자 요청(2026-09-04): "각각을 t-SNE로 한번 전사해보자" — 채널별 mean/std
스칼라 판별력(diagnose_hdp_multi_head_signal.py)은 다 약했지만, 그 값의 "공간적 배치(모양)"
자체는 아직 하나(종양함량)만 봤다. 나머지 5채널도 같은 32x32 grid(cell당 [channel 평균,
coverage]) 표현으로 만들어 t-SNE한다.

채널마다 별도로 grid를 만들고(같은 좌표계, coverage 채널은 채널 무관하게 동일하지만 형식
일관성을 위해 채널별로 다시 포함), 6채널 x (OS_event 색칠 / 모델 risk 색칠) = 12개 서브플롯을
한 그림에 담는다. 각 채널의 저위험-고위험/event-censored 중심거리 비율도 한 표로 모아 어느
채널이 그나마 제일 나은지 바로 비교할 수 있게 한다.

사용법:
    python -m scripts.viz_hdp_multi_channel_grid_tsne --dataset cptac --grid-size 32
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, pdac_consistency_gene_ids
from train import _identity_collate
from scripts.train_hdp_multi_head import MultiHead


def _build_grid(coords: np.ndarray, scores: np.ndarray, grid_size: int) -> np.ndarray:
    xy = coords.astype(np.float64)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = np.clip(hi - lo, 1e-6, None)
    norm = (xy - lo) / span
    bin_idx = np.clip((norm * grid_size).astype(np.int64), 0, grid_size - 1)

    content_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
    count = np.zeros((grid_size, grid_size), dtype=np.int64)
    np.add.at(content_sum, (bin_idx[:, 0], bin_idx[:, 1]), scores)
    np.add.at(count, (bin_idx[:, 0], bin_idx[:, 1]), 1)

    content_mean = np.divide(content_sum, count, out=np.zeros_like(content_sum), where=count > 0)
    coverage = (count > 0).astype(np.float64)
    return np.stack([content_mean, coverage], axis=-1)


def _ratio(X, mask_low, mask_high):
    low, high = X[mask_low], X[mask_high]
    if mask_low.sum() < 2 or mask_high.sum() < 2:
        return float("nan")
    d = np.linalg.norm(low.mean(0) - high.mean(0))
    s = np.linalg.norm(X - X.mean(0), axis=1).mean()
    return d / s


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default="cptac", choices=["tcga", "cptac"])
    parser.add_argument("--backbone", type=str, default="uni2native")
    parser.add_argument("--head-path", type=str, default="data/hdp_multi_head.pt")
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--risk-csv", type=str, default=".scratch/wsi_embedding_tsne_cptac_uni2native.csv")
    parser.add_argument("--out", type=str, default=".scratch/hdp_multi_channel_grid_tsne_cptac.png")
    parser.add_argument("--perplexity", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.head_path, map_location=device, weights_only=False)
    head = MultiHead(in_dim=ckpt["in_dim"], n_targets=len(ckpt["target_names"]), hidden_dim=ckpt["hidden_dim"]).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval(); head.requires_grad_(False)
    target_names = ckpt["target_names"]
    print(f"head 로드: targets={target_names}")

    cfg = Config()
    rna_gene_ids = pdac_consistency_gene_ids(1500)
    ds = WSISurvivalDataset(
        cfg.data, dataset=args.dataset, split="all",
        with_clinical=True, with_staging=True, with_margin=True, with_mutation=True,
        with_rna=True, with_cnv=True, rna_gene_ids=rna_gene_ids, feature_backbone=args.backbone,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)
    print(f"{args.dataset} N={len(ds)}")

    risk_by_case = {}
    if Path(args.risk_csv).exists():
        risk_df = pd.read_csv(args.risk_csv)
        risk_by_case = dict(zip(risk_df["case_id"], risk_df["risk"]))

    rows = []
    grids_by_channel = {name: [] for name in target_names}
    with torch.no_grad():
        for patient_slides in loader:
            if not patient_slides:
                continue
            case_id = patient_slides[0]["case_id"]
            os_time = float(patient_slides[0]["OS_time"].item())
            os_event = int(patient_slides[0]["OS_event"].item())
            feats = torch.cat([s["features"] for s in patient_slides], dim=0).float()
            coords = torch.cat([s["coords"] for s in patient_slides], dim=0).numpy()

            scores = np.empty((feats.shape[0], len(target_names)), dtype=np.float64)
            for i in range(0, feats.shape[0], args.batch_size):
                chunk = feats[i:i + args.batch_size].to(device)
                scores[i:i + args.batch_size] = head(chunk).cpu().numpy()

            for j, name in enumerate(target_names):
                grid = _build_grid(coords, scores[:, j], args.grid_size)
                grids_by_channel[name].append(grid.reshape(-1))

            rows.append({
                "case_id": case_id, "OS_time": os_time, "OS_event": os_event,
                "model_risk": risk_by_case.get(case_id, np.nan),
            })

    df = pd.DataFrame(rows)
    has_risk = df["model_risk"].notna().all()
    if has_risk:
        edges = np.quantile(df["model_risk"], [1 / 3, 2 / 3])
        df["risk_tertile"] = np.digitize(df["model_risk"], edges)
    print(f"N={len(df)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_ch = len(target_names)
    n_cols = 2 if has_risk else 1
    fig, axes = plt.subplots(n_ch, n_cols, figsize=(6.5 * n_cols, 4.2 * n_ch), squeeze=False)

    ev_colors = {0: "#a0aec0", 1: "#c53030"}
    tcolors = {0: "#2b6cb0", 1: "#a0aec0", 2: "#c53030"}
    summary_rows = []

    for ci, name in enumerate(target_names):
        X = np.stack(grids_by_channel[name])
        tsne = TSNE(n_components=2, perplexity=args.perplexity, init="pca", random_state=args.seed)
        coords2d = tsne.fit_transform(X)
        df["tsne_x"], df["tsne_y"] = coords2d[:, 0], coords2d[:, 1]

        ax = axes[ci][0]
        for e in (0, 1):
            sub = df[df["OS_event"] == e]
            ax.scatter(sub["tsne_x"], sub["tsne_y"], c=ev_colors[e], s=25, edgecolors="k", linewidths=0.2,
                       label={0: "censored", 1: "event"}[e])
        ax.set_title(f"[{name}] by OS_event")
        if ci == 0:
            ax.legend(fontsize=7)

        ratio_event = _ratio(X, (df["OS_event"] == 0).to_numpy(), (df["OS_event"] == 1).to_numpy())
        ratio_risk = float("nan")
        if has_risk:
            ax2 = axes[ci][1]
            for t in (0, 1, 2):
                sub = df[df["risk_tertile"] == t]
                ax2.scatter(sub["tsne_x"], sub["tsne_y"], c=tcolors[t], s=25, edgecolors="k", linewidths=0.2,
                            label={0: "low", 1: "mid", 2: "high"}[t] + " risk")
            ax2.set_title(f"[{name}] by model risk tertile")
            if ci == 0:
                ax2.legend(fontsize=7)
            ratio_risk = _ratio(X, (df["risk_tertile"] == 0).to_numpy(), (df["risk_tertile"] == 2).to_numpy())

        summary_rows.append({"channel": name, "event_ratio": ratio_event, "risk_ratio": ratio_risk})
        print(f"  [{name}] event 중심거리 비율={ratio_event:.3f}" + (f", risk 중심거리 비율={ratio_risk:.3f}" if has_risk else ""))

    fig.suptitle(f"{args.dataset.upper()} — HDP 6채널 공간배치(macro shape) t-SNE, N={len(df)}")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"\n플롯 저장: {args.out}")

    summary = pd.DataFrame(summary_rows).sort_values("event_ratio", ascending=False)
    print("\n=== 채널별 요약(내림차순, event_ratio 기준) ===")
    print(summary.to_string(index=False))
    summary.to_csv(Path(args.out).with_suffix(".summary.csv"), index=False)


if __name__ == "__main__":
    main()
