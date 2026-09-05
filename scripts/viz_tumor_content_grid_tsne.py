"""
사용자 질문(2026-09-04): "HDP에서 패치 종양비율을 스칼라로 계산해서 이미지와 같은 사이즈로
맵핑한 거 기억나? 그걸로 t-SNE를 돌려보면 어떨까?" — 지금까지 미시(패치 단위, 텍스처/국소
디테일)와 극단적으로 뭉뚱그린 거시(환자당 스칼라 1개, mean_tumor_content 등, diagnose_hdp_
feature_signal.py) 둘 다 생존과 무관했다. 둘 사이의 중간 스케일 — "종양이 슬라이드 안에서
어떤 거시적 공간 배치(모양/구조)를 이루는가" — 은 아직 안 봤다는 게 사용자의 통찰이다.

TumorContentHead(models/tumor_content_head.py, PanNuke로 학습된 frozen 종양함량 회귀기)를
패치 하나하나에 적용해 스칼라(0~1)를 얻은 뒤, 그 패치의 (x,y) 좌표에 맞춰 고정 크기 grid(기본
32x32)에 뿌려 넣는다 — 환자마다 실제 WSI 크기/종횡비가 다르므로 좌표를 그 환자 자신의
bounding box 기준으로 [0,1] 정규화한 뒤 비닝한다(따라서 이 표현은 절대 크기가 아니라 상대적
"모양"만 담음, 종횡비도 정사각형으로 눌러 담기 때문에 왜곡될 수 있다는 한계가 있음 — 결과
해석 시 감안).

grid 셀 하나당 2채널을 저장한다:
  content:  그 셀에 속한 패치들의 평균 종양함량(빈 셀은 0)
  coverage: 그 셀에 패치가 있었는지 여부(0/1) — 이게 사실상 "조직이 이 위치에 있었는가"라
            거시적 실루엣/모양 자체를 담는다(패치는 애초에 조직이 있는 곳에서만 추출됐으므로).
두 채널을 펼쳐 이어붙인 고정 길이 벡터(grid*grid*2)를 환자 1명의 대표값으로 삼아 t-SNE한다.

사용법:
    python -m scripts.viz_tumor_content_grid_tsne --dataset cptac --grid-size 32
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
from models.tumor_content_head import TumorContentHead
from train import _identity_collate


def _load_head(device, head_path: Path) -> TumorContentHead:
    ckpt = torch.load(head_path, map_location=device, weights_only=False)
    head = TumorContentHead(in_dim=ckpt["in_dim"], hidden_dim=ckpt["hidden_dim"]).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    head.requires_grad_(False)
    return head


def _build_grid(coords: np.ndarray, scores: np.ndarray, grid_size: int) -> np.ndarray:
    """coords(N,2), scores(N,) -> (grid_size, grid_size, 2) [content_mean, coverage]. 환자 자신의
    bounding box로 [0, grid_size) 정규화 후 비닝 — 절대 크기/종횡비 정보는 버리고 상대적 배치만 남긴다."""
    xy = coords.astype(np.float64)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = np.clip(hi - lo, 1e-6, None)
    norm = (xy - lo) / span  # [0,1]
    bin_idx = np.clip((norm * grid_size).astype(np.int64), 0, grid_size - 1)

    content_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
    count = np.zeros((grid_size, grid_size), dtype=np.int64)
    np.add.at(content_sum, (bin_idx[:, 0], bin_idx[:, 1]), scores)
    np.add.at(count, (bin_idx[:, 0], bin_idx[:, 1]), 1)

    content_mean = np.divide(content_sum, count, out=np.zeros_like(content_sum), where=count > 0)
    coverage = (count > 0).astype(np.float64)
    return np.stack([content_mean, coverage], axis=-1)  # (G, G, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default="cptac", choices=["tcga", "cptac"])
    parser.add_argument("--backbone", type=str, default="uni2native")
    parser.add_argument("--head-path", type=str, default="data/hdp_pretrain_tumor_content_head.pt")
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--risk-csv", type=str, default=".scratch/wsi_embedding_tsne_cptac_uni2native.csv")
    parser.add_argument("--out-prefix", type=str, default=".scratch/tumor_content_grid_tsne_cptac")
    parser.add_argument("--perplexity", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = _load_head(device, Path(args.head_path))
    print(f"TumorContentHead 로드: {args.head_path}")

    cfg = Config()
    rna_gene_ids = pdac_consistency_gene_ids(1500)
    ds = WSISurvivalDataset(
        cfg.data, dataset=args.dataset, split="all",
        with_clinical=True, with_staging=True, with_margin=True, with_mutation=True,
        with_rna=True, with_cnv=True, rna_gene_ids=rna_gene_ids, feature_backbone=args.backbone,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)
    print(f"{args.dataset} N={len(ds)}, backbone={args.backbone}")

    risk_by_case = {}
    if Path(args.risk_csv).exists():
        risk_df = pd.read_csv(args.risk_csv)
        risk_by_case = dict(zip(risk_df["case_id"], risk_df["risk"]))

    rows, grids = [], []
    with torch.no_grad():
        for patient_slides in loader:
            if not patient_slides:
                continue
            case_id = patient_slides[0]["case_id"]
            os_time = float(patient_slides[0]["OS_time"].item())
            os_event = int(patient_slides[0]["OS_event"].item())
            feats = torch.cat([s["features"] for s in patient_slides], dim=0).float()
            coords = torch.cat([s["coords"] for s in patient_slides], dim=0).numpy()

            scores = np.empty(feats.shape[0], dtype=np.float64)
            for i in range(0, feats.shape[0], args.batch_size):
                chunk = feats[i:i + args.batch_size].to(device)
                scores[i:i + args.batch_size] = head(chunk).cpu().numpy()

            grid = _build_grid(coords, scores, args.grid_size)
            grids.append(grid.reshape(-1))
            rows.append({
                "case_id": case_id, "OS_time": os_time, "OS_event": os_event,
                "model_risk": risk_by_case.get(case_id, np.nan),
                "mean_tumor_content": scores.mean(), "n_patches": len(scores),
            })

    df = pd.DataFrame(rows)
    X = np.stack(grids)  # (N, grid*grid*2)
    print(f"N={len(df)}, grid vector dim={X.shape[1]} ({args.grid_size}x{args.grid_size}x2)")

    has_risk = df["model_risk"].notna().all()
    if has_risk:
        edges = np.quantile(df["model_risk"], [1 / 3, 2 / 3])
        df["risk_tertile"] = np.digitize(df["model_risk"], edges)

    tsne = TSNE(n_components=2, perplexity=args.perplexity, init="pca", random_state=args.seed)
    coords2d = tsne.fit_transform(X)
    df["tsne_x"], df["tsne_y"] = coords2d[:, 0], coords2d[:, 1]
    df.to_csv(f"{args.out_prefix}.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 2 if has_risk else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5.5), squeeze=False)
    axes = axes[0]
    ev_colors = {0: "#a0aec0", 1: "#c53030"}
    for e in (0, 1):
        sub = df[df["OS_event"] == e]
        axes[0].scatter(sub["tsne_x"], sub["tsne_y"], c=ev_colors[e], s=40, edgecolors="k", linewidths=0.3,
                         label={0: "censored", 1: "event"}[e])
    axes[0].set_title(f"tumor-content spatial grid ({args.grid_size}x{args.grid_size}x2) t-SNE, colored by OS_event")
    axes[0].legend(fontsize=8)
    if has_risk:
        tcolors = {0: "#2b6cb0", 1: "#a0aec0", 2: "#c53030"}
        for t in (0, 1, 2):
            sub = df[df["risk_tertile"] == t]
            axes[1].scatter(sub["tsne_x"], sub["tsne_y"], c=tcolors[t], s=40, edgecolors="k", linewidths=0.3,
                             label={0: "low", 1: "mid", 2: "high"}[t] + " model risk")
        axes[1].set_title("same coords, colored by trained-model risk tertile")
        axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    fig.suptitle(f"{args.dataset.upper()} — patient 종양함량 공간 배치(macro shape) t-SNE, N={len(df)}")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}.png", dpi=150)
    print(f"플롯 저장: {args.out_prefix}.png")

    def _ratio(mask_low, mask_high):
        low, high = X[mask_low], X[mask_high]
        if len(low) < 2 or len(high) < 2:
            return float("nan")
        d = np.linalg.norm(low.mean(0) - high.mean(0))
        s = np.linalg.norm(X - X.mean(0), axis=1).mean()
        return d / s

    ratio_event = _ratio((df["OS_event"] == 0).to_numpy(), (df["OS_event"] == 1).to_numpy())
    print(f"[grid, 원본 D={X.shape[1]}차원] event vs censored 중심거리 비율 = {ratio_event:.3f}")
    if has_risk:
        ratio_risk = _ratio((df["risk_tertile"] == 0).to_numpy(), (df["risk_tertile"] == 2).to_numpy())
        print(f"[grid, 원본 D={X.shape[1]}차원] low vs high model-risk 중심거리 비율 = {ratio_risk:.3f}")

    # 참고: mean_tumor_content(스칼라 1개, diagnose_hdp_feature_signal.py와 동일 축) 단독의
    # event 판별력도 같은 자리에서 다시 확인 — grid 표현이 이 스칼라보다 더 나은 신호를 담고
    # 있는지 직접 비교하기 위함.
    from utils.metrics import compute_survival_metrics
    m_pos = compute_survival_metrics(df["mean_tumor_content"].to_numpy(), df["OS_time"].to_numpy(), df["OS_event"].to_numpy())
    m_neg = compute_survival_metrics(-df["mean_tumor_content"].to_numpy(), df["OS_time"].to_numpy(), df["OS_event"].to_numpy())
    print(f"(참고) mean_tumor_content 스칼라 단독 c_index = {max(m_pos['c_index'], m_neg['c_index']):.4f}")


if __name__ == "__main__":
    main()
