"""
사용자 질문(2026-09-04): "WSI(환자 단위 pooled) 말고 패치 기준으로도 t-SNE 할 수 있나?"
— scripts/viz_wsi_embedding_tsne.py가 본 z_wsi(환자당 1개, attn_pool 이후)는 attention
가중합으로 이미 뭉개진 뒤라 신호가 평균에 묻혔을 수 있다. 여기서는 attn_pool *직전*
patch token(models/vit_m1.py::ViT_M1.forward의 ctx_tokens, self.vit를 지나 패치끼리
context는 섞였지만 아직 한 벡터로 풀링되진 않은 상태) 전부를 모아 t-SNE한다 — 패치 개수가
많아(CPTAC 136명 합계 21,606개, 환자당 median 109개) 환자 단위보다 훨씬 촘촘한 해상도로
"어떤 조직 패치가 고위험/저위험 환자에게서 나오는가"를 볼 수 있다.

model.attn_pool에 forward_pre_hook을 걸어 매 슬라이드 forward마다 (tokens, context) 입력을
가로챈다 — models/vit_m1.py를 전혀 안 건드림. 패치 하나하나에 그 패치가 속한 환자의
risk/OS 라벨을 그대로 상속시켜 색칠한다. attn_weight(그 패치가 pooling에서 받은 중요도)도
같이 저장해, "attention이 높은 패치일수록 더 갈리는지"도 별도로 볼 수 있게 한다.

사용법:
    python -m scripts.viz_patch_embedding_tsne --ckpt <path/to/best_clinical_rna.pt>
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
                         help="2026-09-04: 체크포인트 학습 때 쓴 --backbone과 일치해야 함 — "
                              "논문 헤드라인 모델과 맞추려면 uni2native를 명시.")
    parser.add_argument("--train-dataset", type=str, default="tcga", choices=["tcga", "cptac"])
    parser.add_argument("--out", type=str, default=".scratch/patch_embedding_tsne_cptac.png")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-patches", type=int, default=25000,
                         help="t-SNE 비용 상한 — 전체 패치 수가 이보다 크면 환자별 균등 랜덤 서브샘플.")
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

    # model.attn_pool 직전 patch token(ctx_tokens, self.vit 통과 후)을 가로챈다 — 슬라이드 1장당
    # 1번 호출되므로, 환자당 슬라이드가 여러 장이면 여러 번 쌓인다(patient 루프 시작 시 비움).
    captured_slides: list[torch.Tensor] = []
    captured_attn: list[torch.Tensor] = []

    def _hook(module, inputs):
        captured_slides.append(inputs[0].detach())
        # context(inputs[1])는 RNA 컨텍스트라 패치마다 동일 — attn_weight는 forward *출력*이라
        # pre_hook으로는 못 얻는다. 대신 forward_hook을 별도로 안 걸고, out["attn_weights"]를
        # _patient_risk가 리턴하지 않으므로, forward_hook으로 출력까지 같이 잡는다.

    def _hook_out(module, inputs, output):
        wsi_embed, attn_weights = output
        captured_attn.append(attn_weights.detach())

    h1 = model.attn_pool.register_forward_pre_hook(_hook)
    h2 = model.attn_pool.register_forward_hook(_hook_out)

    embed_dim = cfg.model.embed_dim
    chunk_size = cfg.train.cnn_chunk_size
    rows = []  # dict per patch
    patch_vecs = []
    with torch.no_grad():
        for patient_slides in external_loader:
            if not patient_slides:
                continue
            captured_slides.clear()
            captured_attn.clear()
            risk, _, _ = _patient_risk(model, patient_slides, device, amp_ctx, None, chunk_size)
            case_id = patient_slides[0]["case_id"]
            os_time = float(patient_slides[0]["OS_time"].item())
            os_event = int(patient_slides[0]["OS_event"].item())
            risk_val = risk.float().item()
            for tokens, attn in zip(captured_slides, captured_attn):
                tokens = tokens.float().cpu().numpy()  # (N, D)
                attn = attn.float().cpu().numpy()       # (N,)
                for j in range(tokens.shape[0]):
                    rows.append({
                        "case_id": case_id, "risk": risk_val, "OS_time": os_time, "OS_event": os_event,
                        "attn_weight": float(attn[j]),
                    })
                    patch_vecs.append(tokens[j])
    h1.remove(); h2.remove()

    df = pd.DataFrame(rows)
    X = np.stack(patch_vecs)
    print(f"총 패치 수={len(df)} (환자 {df['case_id'].nunique()}명), z_patch dim={X.shape[1]}")

    # [--max-patches] 비용 상한 — 환자별 patch 개수 비율은 유지한 채 균등 랜덤 서브샘플.
    if len(df) > args.max_patches:
        rng = np.random.default_rng(args.seed)
        frac = args.max_patches / len(df)
        keep_idx = []
        for cid, sub in df.groupby("case_id"):
            k = max(1, round(len(sub) * frac))
            keep_idx.extend(rng.choice(sub.index.to_numpy(), size=min(k, len(sub)), replace=False))
        keep_idx = np.array(sorted(keep_idx))
        df = df.loc[keep_idx].reset_index(drop=True)
        X = X[keep_idx]
        print(f"서브샘플 후 패치 수={len(df)}")

    risk_edges = np.quantile(df["risk"], [1 / 3, 2 / 3])
    df["risk_tertile"] = np.digitize(df["risk"], risk_edges)

    tsne = TSNE(n_components=2, perplexity=args.perplexity, init="pca", random_state=args.seed)
    coords2d = tsne.fit_transform(X)
    df["tsne_x"] = coords2d[:, 0]
    df["tsne_y"] = coords2d[:, 1]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"패치 좌표 CSV 저장: {out_path.with_suffix('.csv')}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    tertile_colors = {0: "#2b6cb0", 1: "#a0aec0", 2: "#c53030"}
    tertile_labels = {0: "low risk", 1: "mid risk", 2: "high risk"}
    for t in (0, 1, 2):
        sub = df[df["risk_tertile"] == t]
        axes[0].scatter(sub["tsne_x"], sub["tsne_y"], c=tertile_colors[t], s=3, alpha=0.5, label=tertile_labels[t])
    axes[0].set_title("patch t-SNE, colored by patient risk tertile")
    axes[0].legend(fontsize=8, markerscale=4)

    event_colors = {0: "#a0aec0", 1: "#c53030"}
    event_labels = {0: "censored", 1: "event (death)"}
    for e in (0, 1):
        sub = df[df["OS_event"] == e]
        axes[1].scatter(sub["tsne_x"], sub["tsne_y"], c=event_colors[e], s=3, alpha=0.5, label=event_labels[e])
    axes[1].set_title("patch t-SNE, colored by OS_event (ground truth)")
    axes[1].legend(fontsize=8, markerscale=4)

    sc = axes[2].scatter(df["tsne_x"], df["tsne_y"], c=df["attn_weight"], cmap="magma", s=3, alpha=0.6)
    axes[2].set_title("patch t-SNE, colored by attn_pool weight (importance)")
    fig.colorbar(sc, ax=axes[2], label="attn weight")

    for ax in axes:
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    fig.suptitle(f"CPTAC-PDA external — patch-level token t-SNE (N_patch={len(df)}, {df['case_id'].nunique()} patients)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"플롯 저장: {out_path}")

    # tertile 간 분리도 정량 체크(원본 D차원 공간 기준, WSI 스크립트와 동일 관례).
    low = X[df["risk_tertile"] == 0]
    high = X[df["risk_tertile"] == 2]
    center_dist = np.linalg.norm(low.mean(0) - high.mean(0))
    overall_scale = np.linalg.norm(X - X.mean(0), axis=1).mean()
    print(f"\nlow-risk vs high-risk 패치 임베딩 중심 거리 = {center_dist:.4f} "
          f"(전체 평균 반경 {overall_scale:.4f} 대비 비율 {center_dist / overall_scale:.3f})")

    # attn_weight가 높은 패치(모델이 실제로 중요하다고 본 patch)만 따로 같은 체크 — 낮은
    # attn 패치(모델이 무시한, 배경/불필요 조직일 가능성)까지 섞으면 신호가 더 희석될 수 있어
    # 상위 20%만으로도 반복.
    top_attn = df["attn_weight"] >= df["attn_weight"].quantile(0.8)
    low_top = X[top_attn.to_numpy() & (df["risk_tertile"] == 0).to_numpy()]
    high_top = X[top_attn.to_numpy() & (df["risk_tertile"] == 2).to_numpy()]
    if len(low_top) > 1 and len(high_top) > 1:
        center_dist_top = np.linalg.norm(low_top.mean(0) - high_top.mean(0))
        print(f"(상위 attn 20% 패치만) low vs high 중심 거리 = {center_dist_top:.4f} "
              f"(전체 평균 반경 대비 비율 {center_dist_top / overall_scale:.3f})")


if __name__ == "__main__":
    main()
