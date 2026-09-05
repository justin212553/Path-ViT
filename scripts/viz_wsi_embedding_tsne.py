"""
사용자 질문(2026-09-04): "AI가 위험도가 높다고 평가한 환자와 위험도가 낮다고 평가한 환자랑
대체 WSI 수준에서 뭐가 다른거지? 그걸 tSNE 같은 걸로 그룹화 할 수 있나?"

채택된 M4 레시피(pdac_consistency_1500 + CNV + mutation + clinical-staging/margin,
--combine-mode cox_add, --clinical-lr-mult 100)로 fold0/seed84 체크포인트를 새로 학습해
(scripts/experiment_m4_wsi_cox_add.py에는 있던 checkpoint 저장이 train.py 본 실행에는
있으므로 그쪽을 재사용), model.risk_head 직전의 fused=[z_wsi‖z_rna] 중 z_wsi 절반만 뽑아
CPTAC(external, N=136, 학습에 전혀 안 쓰인 고정 코호트)에서 patient별 pooled WSI 임베딩을
모은다. t-SNE로 2D 투영 후 예측 risk score(연속/tertile)로 색칠해, WSI 임베딩 공간에서
고위험/저위험군이 분리되는지 시각적으로 확인한다.

사용법:
    python -m scripts.viz_wsi_embedding_tsne --ckpt <path/to/best_pma.pt>
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
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, pdac_consistency_gene_ids
from models import ViT_M4
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv, stage_stats_from_df, mutation_stats_from_df
from train import _patient_risk, _identity_collate, _make_amp_ctx


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="resnet50",
                         help="2026-09-04: 체크포인트를 학습할 때 쓴 --backbone과 반드시 일치해야 "
                              "함(모델 cnn 인코더 입력 차원 + 로드할 feature 캐시 파일이 둘 다 "
                              "여기 달림). 기본값(resnet50)은 논문이 실제 채택한 백본이 아니라 "
                              "train.py --backbone 기본값일 뿐 — 논문 헤드라인 모델과 맞추려면 "
                              "uni2native를 명시해야 한다.")
    parser.add_argument("--train-dataset", type=str, default="tcga", choices=["tcga", "cptac"],
                         help="체크포인트가 학습된 코호트 — external은 그 반대로 자동 결정.")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--out", type=str, default=".scratch/wsi_embedding_tsne_cptac.png")
    parser.add_argument("--perplexity", type=float, default=15.0,
                         help="N=136 기준(CPTAC) 권장 10~30. 너무 크면(>N/3) sklearn이 에러.")
    parser.add_argument("--seed", type=int, default=84)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx = _make_amp_ctx()
    external_dataset = {"tcga": "cptac", "cptac": "tcga"}[args.train_dataset]

    cfg = Config()
    rna_gene_ids = pdac_consistency_gene_ids(1500)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.train_dataset])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS[args.train_dataset])
    stage_stats = stage_stats_from_df(pd.read_csv(CLINICAL_PATHS[args.train_dataset]))
    mutation_stats = mutation_stats_from_df(pd.read_csv(CLINICAL_PATHS[args.train_dataset]))

    ds_kwargs = dict(
        with_clinical=True, with_staging=True, with_margin=True, with_mutation=True,
        with_rna=True, with_cnv=True, rna_gene_ids=rna_gene_ids, feature_backbone=args.backbone,
    )
    external_ds = WSISurvivalDataset(cfg.data, dataset=external_dataset, split="all", **ds_kwargs)
    external_loader = DataLoader(external_ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)
    print(f"external({external_dataset}) N={len(external_ds)}")

    rna_input_dim = len(rna_gene_ids) + 8
    model = ViT_M4(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
        precomputed=cfg.data.precomputed, combine_mode="cox_add", backbone=args.backbone,
        use_staging=True, stage_stats=stage_stats, use_margin=True, margin_stats=margin_stats,
        use_mutation=True, mutation_stats=mutation_stats,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"체크포인트 로드: {args.ckpt} (epoch={ckpt.get('epoch')}, val_c_index={ckpt.get('val_c_index')})")

    # model.risk_head 직전 입력(fused=[z_wsi‖z_rna], cox_add)을 forward_pre_hook으로 가로챈다 —
    # models/vit_m4.py를 안 건드리고도 risk_head가 실제로 보는 텐서를 그대로 얻는다.
    captured = {}

    def _hook(module, inputs):
        captured["fused"] = inputs[0].detach()

    handle = model.risk_head.register_forward_pre_hook(_hook)

    embed_dim = cfg.model.embed_dim
    z_wsi_list, risks, times, events, case_ids = [], [], [], [], []
    chunk_size = cfg.train.cnn_chunk_size
    with torch.no_grad():
        for patient_slides in external_loader:
            if not patient_slides:
                continue
            risk, _, _ = _patient_risk(model, patient_slides, device, amp_ctx, None, chunk_size)
            fused = captured["fused"]  # (1, 2*embed_dim)
            z_wsi = fused[:, :embed_dim].float().cpu().numpy().reshape(-1)
            z_wsi_list.append(z_wsi)
            risks.append(risk.float().item())
            times.append(float(patient_slides[0]["OS_time"].item()))
            events.append(int(patient_slides[0]["OS_event"].item()))
            case_ids.append(patient_slides[0]["case_id"])
    handle.remove()

    Z = np.stack(z_wsi_list)  # (N, embed_dim)
    risks = np.array(risks)
    times = np.array(times)
    events = np.array(events)
    print(f"z_wsi shape={Z.shape}, risk range=[{risks.min():.3f}, {risks.max():.3f}]")

    tsne = TSNE(n_components=2, perplexity=args.perplexity, init="pca", random_state=args.seed)
    coords = tsne.fit_transform(Z)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "case_id": case_ids, "tsne_x": coords[:, 0], "tsne_y": coords[:, 1],
        "risk": risks, "OS_time": times, "OS_event": events,
    })
    tertile_edges = np.quantile(risks, [1 / 3, 2 / 3])
    df["risk_tertile"] = np.digitize(risks, tertile_edges)  # 0=low, 1=mid, 2=high
    csv_path = out_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"좌표+risk CSV 저장: {csv_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    sc = axes[0].scatter(df["tsne_x"], df["tsne_y"], c=df["risk"], cmap="viridis", s=40, edgecolors="k", linewidths=0.3)
    axes[0].set_title("WSI embedding t-SNE, colored by predicted risk (continuous)")
    axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")
    fig.colorbar(sc, ax=axes[0], label="risk score")

    tertile_colors = {0: "#2b6cb0", 1: "#a0aec0", 2: "#c53030"}
    tertile_labels = {0: "low risk (bottom 1/3)", 1: "mid risk", 2: "high risk (top 1/3)"}
    for t in (0, 1, 2):
        sub = df[df["risk_tertile"] == t]
        axes[1].scatter(sub["tsne_x"], sub["tsne_y"], c=tertile_colors[t], s=40,
                         edgecolors="k", linewidths=0.3, label=tertile_labels[t])
    axes[1].set_title("WSI embedding t-SNE, colored by risk tertile")
    axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"CPTAC-PDA external cohort (N={len(df)}) — pooled WSI embedding (z_wsi, D={embed_dim}) before risk_head")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"플롯 저장: {out_path}")

    # 참고: risk tertile 간 WSI 임베딩 분리도를 대략적인 수치로도 확인 — tertile 중심 간
    # 거리를 임베딩 전체 분산(평균 pairwise 거리)으로 정규화한 값. 1 근처면 군집 간 거리가
    # 무작위 두 점 사이 거리와 비슷하다는 뜻(=t-SNE에서 겹쳐 보여도 이상하지 않음).
    low = Z[df["risk_tertile"] == 0]
    high = Z[df["risk_tertile"] == 2]
    center_dist = np.linalg.norm(low.mean(0) - high.mean(0))
    overall_scale = np.linalg.norm(Z - Z.mean(0), axis=1).mean()
    print(f"\nlow-risk vs high-risk WSI 임베딩 중심 거리 = {center_dist:.4f} "
          f"(임베딩 전체 평균 반경 {overall_scale:.4f} 대비 비율 {center_dist / overall_scale:.3f})")


if __name__ == "__main__":
    main()
