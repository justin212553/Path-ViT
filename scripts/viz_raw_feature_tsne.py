"""
사용자 질문(2026-09-04): "원본으로 t-SNE를 돌리든 모델을 돌려서 t-SNE를 돌리든 거기서
거기란 뜻?" — scripts/viz_wsi_embedding_tsne.py, viz_patch_embedding_tsne.py는 전부
model.cnn(투영) + model.vit(self-attention mixing) + (WSI 단위는 attn_pool까지) 통과한
*학습된* 표현이었다. 즉 risk 예측하도록 최적화된 레이어를 이미 한번 거친 벡터라, "신호가
원래 없다"인지 "우리 학습 과정에서 신호가 씻겨나갔다"인지 구분이 안 됐다.

여기서는 모델을 아예 안 거친다 — WSISurvivalDataset이 로드하는 UNI2-h/uni2native(또는
resnet50) *원본* precomputed patch feature를 그대로 t-SNE한다. 모델 forward도, 체크포인트도
필요 없다(feature encoder는 이미 고정된 사전학습 가중치라 우리 학습과 무관).

patch 단위(원본 그대로)와 WSI 단위(환자별 단순 평균 풀링 — attn_pool 없이, 학습된 가중치가
전혀 안 들어간 가장 순수한 patient-level 대표값) 둘 다 낸다. 색은 실제 OS_event/OS_time
라벨(가장 객관적) + (있으면) 기존에 저장해 둔 모델 예측 risk(.csv, 참고용 비교)를 같이 쓴다.

사용법:
    python -m scripts.viz_raw_feature_tsne --backbone uni2native
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default="cptac", choices=["tcga", "cptac"])
    parser.add_argument("--backbone", type=str, default="uni2native")
    parser.add_argument("--out-prefix", type=str, default=".scratch/raw_feature_tsne_cptac")
    parser.add_argument("--risk-csv", type=str, default=".scratch/wsi_embedding_tsne_cptac_uni2native.csv",
                         help="이미 저장된 모델 예측 risk(case_id별)를 참고 색상으로 같이 쓰기 위함 — 없으면 생략.")
    parser.add_argument("--max-patches", type=int, default=25000)
    parser.add_argument("--perplexity-wsi", type=float, default=20.0)
    parser.add_argument("--perplexity-patch", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=84)
    args = parser.parse_args()

    np.random.seed(args.seed)
    cfg = Config()
    rna_gene_ids = pdac_consistency_gene_ids(1500)
    ds = WSISurvivalDataset(
        cfg.data, dataset=args.dataset, split="all",
        with_clinical=True, with_staging=True, with_margin=True, with_mutation=True,
        with_rna=True, with_cnv=True, rna_gene_ids=rna_gene_ids, feature_backbone=args.backbone,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)
    print(f"{args.dataset} N={len(ds)}, backbone={args.backbone} (모델 미사용, 원본 precomputed feature)")

    risk_by_case = {}
    if Path(args.risk_csv).exists():
        risk_df = pd.read_csv(args.risk_csv)
        risk_by_case = dict(zip(risk_df["case_id"], risk_df["risk"]))
        print(f"참고용 모델 risk 로드: {args.risk_csv} ({len(risk_by_case)}명)")

    wsi_rows, wsi_vecs = [], []
    patch_rows, patch_vecs = [], []
    for patient_slides in loader:
        if not patient_slides:
            continue
        case_id = patient_slides[0]["case_id"]
        os_time = float(patient_slides[0]["OS_time"].item())
        os_event = int(patient_slides[0]["OS_event"].item())
        all_feats = torch.cat([s["features"] for s in patient_slides], dim=0).float().numpy()  # (N_total, D)
        # WSI 단위: 학습된 attn_pool 없이 가장 순수한 단순 평균(mean pooling) — 모델 가중치 0개 개입.
        wsi_vecs.append(all_feats.mean(axis=0))
        wsi_rows.append({
            "case_id": case_id, "OS_time": os_time, "OS_event": os_event,
            "model_risk": risk_by_case.get(case_id, np.nan),
        })
        for j in range(all_feats.shape[0]):
            patch_rows.append({
                "case_id": case_id, "OS_time": os_time, "OS_event": os_event,
                "model_risk": risk_by_case.get(case_id, np.nan),
            })
        patch_vecs.append(all_feats)

    wsi_df = pd.DataFrame(wsi_rows)
    Xw = np.stack(wsi_vecs)
    patch_df = pd.DataFrame(patch_rows)
    Xp = np.concatenate(patch_vecs, axis=0)
    print(f"WSI-pooled(단순평균) N={len(wsi_df)}, D={Xw.shape[1]} | 총 패치 수={len(patch_df)}")

    if len(patch_df) > args.max_patches:
        rng = np.random.default_rng(args.seed)
        frac = args.max_patches / len(patch_df)
        keep_idx = []
        offset = 0
        for feats in patch_vecs:
            n = feats.shape[0]
            k = max(1, round(n * frac))
            local_idx = rng.choice(n, size=min(k, n), replace=False)
            keep_idx.extend((local_idx + offset).tolist())
            offset += n
        keep_idx = np.array(sorted(keep_idx))
        patch_df = patch_df.loc[keep_idx].reset_index(drop=True)
        Xp = Xp[keep_idx]
        print(f"패치 서브샘플 후={len(patch_df)}")

    def _add_tertile(df, col):
        edges = np.quantile(df[col].dropna(), [1 / 3, 2 / 3])
        df[f"{col}_tertile"] = np.digitize(df[col].fillna(edges[0]), edges)

    has_model_risk = wsi_df["model_risk"].notna().all()
    if has_model_risk:
        _add_tertile(wsi_df, "model_risk")
        _add_tertile(patch_df, "model_risk")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _center_dist_ratio(X, mask_low, mask_high):
        low, high = X[mask_low], X[mask_high]
        if len(low) < 2 or len(high) < 2:
            return float("nan")
        center_dist = np.linalg.norm(low.mean(0) - high.mean(0))
        overall = np.linalg.norm(X - X.mean(0), axis=1).mean()
        return center_dist / overall

    # === WSI 단위(원본 단순평균) ===
    tsne_w = TSNE(n_components=2, perplexity=args.perplexity_wsi, init="pca", random_state=args.seed)
    coords_w = tsne_w.fit_transform(Xw)
    wsi_df["tsne_x"], wsi_df["tsne_y"] = coords_w[:, 0], coords_w[:, 1]
    wsi_df.to_csv(f"{args.out_prefix}_wsi.csv", index=False)

    n_panels = 2 if has_model_risk else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5.5), squeeze=False)
    axes = axes[0]
    ev_colors = {0: "#a0aec0", 1: "#c53030"}
    for e in (0, 1):
        sub = wsi_df[wsi_df["OS_event"] == e]
        axes[0].scatter(sub["tsne_x"], sub["tsne_y"], c=ev_colors[e], s=40, edgecolors="k", linewidths=0.3,
                         label={0: "censored", 1: "event"}[e])
    axes[0].set_title("raw UNI2-h WSI (mean-pool) t-SNE, colored by OS_event")
    axes[0].legend(fontsize=8)
    if has_model_risk:
        tcolors = {0: "#2b6cb0", 1: "#a0aec0", 2: "#c53030"}
        for t in (0, 1, 2):
            sub = wsi_df[wsi_df["model_risk_tertile"] == t]
            axes[1].scatter(sub["tsne_x"], sub["tsne_y"], c=tcolors[t], s=40, edgecolors="k", linewidths=0.3,
                             label={0: "low", 1: "mid", 2: "high"}[t] + " model risk")
        axes[1].set_title("same coords, colored by trained-model risk tertile")
        axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    fig.suptitle(f"{args.dataset.upper()} — 원본 {args.backbone} patch feature 단순평균(모델 미사용), N={len(wsi_df)}")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_wsi.png", dpi=150)
    print(f"WSI-단위 플롯 저장: {args.out_prefix}_wsi.png")

    ratio_event = _center_dist_ratio(Xw, (wsi_df["OS_event"] == 0).to_numpy(), (wsi_df["OS_event"] == 1).to_numpy())
    print(f"[원본, WSI 단위] event vs censored 중심거리 비율 = {ratio_event:.3f}")
    if has_model_risk:
        ratio_risk = _center_dist_ratio(Xw, (wsi_df["model_risk_tertile"] == 0).to_numpy(),
                                         (wsi_df["model_risk_tertile"] == 2).to_numpy())
        print(f"[원본, WSI 단위] low vs high model-risk 중심거리 비율 = {ratio_risk:.3f}")

    # === 패치 단위(원본) ===
    tsne_p = TSNE(n_components=2, perplexity=args.perplexity_patch, init="pca", random_state=args.seed)
    coords_p = tsne_p.fit_transform(Xp)
    patch_df["tsne_x"], patch_df["tsne_y"] = coords_p[:, 0], coords_p[:, 1]
    patch_df.to_csv(f"{args.out_prefix}_patch.csv", index=False)

    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5.5), squeeze=False)
    axes = axes[0]
    for e in (0, 1):
        sub = patch_df[patch_df["OS_event"] == e]
        axes[0].scatter(sub["tsne_x"], sub["tsne_y"], c=ev_colors[e], s=3, alpha=0.5,
                         label={0: "censored", 1: "event"}[e])
    axes[0].set_title("raw UNI2-h patch t-SNE, colored by OS_event")
    axes[0].legend(fontsize=8, markerscale=4)
    if has_model_risk:
        for t in (0, 1, 2):
            sub = patch_df[patch_df["model_risk_tertile"] == t]
            axes[1].scatter(sub["tsne_x"], sub["tsne_y"], c=tcolors[t], s=3, alpha=0.5,
                             label={0: "low", 1: "mid", 2: "high"}[t] + " model risk")
        axes[1].set_title("same coords, colored by trained-model risk tertile")
        axes[1].legend(fontsize=8, markerscale=4)
    for ax in axes:
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    fig.suptitle(f"{args.dataset.upper()} — 원본 {args.backbone} patch feature(모델 미사용), N_patch={len(patch_df)}")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_patch.png", dpi=150)
    print(f"패치-단위 플롯 저장: {args.out_prefix}_patch.png")

    ratio_event_p = _center_dist_ratio(Xp, (patch_df["OS_event"] == 0).to_numpy(), (patch_df["OS_event"] == 1).to_numpy())
    print(f"[원본, 패치 단위] event vs censored 중심거리 비율 = {ratio_event_p:.3f}")
    if has_model_risk:
        ratio_risk_p = _center_dist_ratio(Xp, (patch_df["model_risk_tertile"] == 0).to_numpy(),
                                           (patch_df["model_risk_tertile"] == 2).to_numpy())
        print(f"[원본, 패치 단위] low vs high model-risk 중심거리 비율 = {ratio_risk_p:.3f}")


if __name__ == "__main__":
    main()
