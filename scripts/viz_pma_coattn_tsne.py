"""
사용자 요청(2026-09-04): "PMA 모델의 RNA co-attention 이후 나온 성분벡터로 t-SNE를 돌려보자.
우리 모델이 그걸 어떻게 망가뜨렸는지/증가시켰는지 비교해야 한다." — 지금까지는 UNI2-h 원본
patch feature(모델 미개입) 또는 ViT_M4의 attn_pool 이후 임베딩만 t-SNE했다. ViT_PMA는 구조가
다르다 — patch들을 4가지 통계적 관점(mean/std/attention-weighted/top-k-mean, models/
multi_component_pooling.py)으로 미리 요약한 뒤(patient_embed, (4,D)), RNA를 query로 한
co-attention(component_coattn, models/vit_m4a.py::CoAttentionPooling)이 "이 환자의 RNA
아형에는 4개 관점 중 뭐가 중요한가"를 판단해 가중합해 z_wsi(D,)를 만든다.

이 스크립트는 model.component_coattn에 forward hook을 걸어 그 직전(입력, patient_embed 4개
관점 평균 = "co-attention 없이 균등 평균"과 동일한 baseline)과 직후(출력 z_wsi = 실제 RNA
co-attention 가중합) 둘 다를 뽑는다. 같은 환자, 같은 좌표계로 두 표현을 각각 t-SNE해서
직접 비교하면 "RNA co-attention이 클래스(risk/OS) 분리도를 올렸는지/내렸는지"를 볼 수 있다.

사용법:
    python -m scripts.viz_pma_coattn_tsne --ckpt <path/to/best_pma.pt>
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
from models import ViT_PMA
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv, stage_stats_from_df
from train import _patient_risk, _identity_collate, _make_amp_ctx


def _ratio(X, mask_low, mask_high):
    low, high = X[mask_low], X[mask_high]
    if mask_low.sum() < 2 or mask_high.sum() < 2:
        return float("nan")
    d = np.linalg.norm(low.mean(0) - high.mean(0))
    s = np.linalg.norm(X - X.mean(0), axis=1).mean()
    return d / s


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="uni2native")
    parser.add_argument("--train-dataset", type=str, default="tcga", choices=["tcga", "cptac"])
    parser.add_argument("--out", type=str, default=".scratch/pma_coattn_tsne_cptac.png")
    parser.add_argument("--perplexity", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=84)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx = _make_amp_ctx()
    external_dataset = {"tcga": "cptac", "cptac": "tcga"}[args.train_dataset]

    cfg = Config()
    rna_gene_ids = pdac_consistency_gene_ids(1500)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.train_dataset])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS[args.train_dataset])
    stage_stats = stage_stats_from_df(pd.read_csv(CLINICAL_PATHS[args.train_dataset]))

    ds_kwargs = dict(
        with_clinical=True, with_staging=True, with_margin=True,
        with_rna=True, with_cnv=True, rna_gene_ids=rna_gene_ids, feature_backbone=args.backbone,
    )
    external_ds = WSISurvivalDataset(cfg.data, dataset=external_dataset, split="all", **ds_kwargs)
    loader = DataLoader(external_ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)
    print(f"external({external_dataset}) N={len(external_ds)}")

    rna_input_dim = len(rna_gene_ids) + 8
    model = ViT_PMA(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
        precomputed=cfg.data.precomputed, backbone=args.backbone, combine_mode="cox_add",
        use_staging=True, stage_stats=stage_stats, use_margin=True, margin_stats=margin_stats,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"체크포인트 로드: {args.ckpt} (epoch={ckpt.get('epoch')}, val_c_index={ckpt.get('val_c_index')})")

    captured = {}

    def _hook(module, inputs, output):
        patient_embed = inputs[0]        # (4, D) — co-attention 이전, 4개 관점
        z_wsi, _ = output                 # (D,) — co-attention 이후
        captured["before"] = patient_embed.mean(dim=0).detach()  # 균등 평균(co-attn 없을 때의 baseline)
        captured["after"] = z_wsi.detach()

    handle = model.component_coattn.register_forward_hook(_hook)

    chunk_size = cfg.train.cnn_chunk_size
    rows, before_vecs, after_vecs = [], [], []
    with torch.no_grad():
        for patient_slides in loader:
            if not patient_slides:
                continue
            risk, _, _ = _patient_risk(model, patient_slides, device, amp_ctx, None, chunk_size)
            case_id = patient_slides[0]["case_id"]
            rows.append({
                "case_id": case_id,
                "risk": risk.float().item(),
                "OS_time": float(patient_slides[0]["OS_time"].item()),
                "OS_event": int(patient_slides[0]["OS_event"].item()),
            })
            before_vecs.append(captured["before"].float().cpu().numpy())
            after_vecs.append(captured["after"].float().cpu().numpy())
    handle.remove()

    df = pd.DataFrame(rows)
    Xb, Xa = np.stack(before_vecs), np.stack(after_vecs)
    edges = np.quantile(df["risk"], [1 / 3, 2 / 3])
    df["risk_tertile"] = np.digitize(df["risk"], edges)
    print(f"N={len(df)}, D={Xb.shape[1]}")

    ev_mask_low, ev_mask_high = (df["OS_event"] == 0).to_numpy(), (df["OS_event"] == 1).to_numpy()
    risk_mask_low, risk_mask_high = (df["risk_tertile"] == 0).to_numpy(), (df["risk_tertile"] == 2).to_numpy()
    print("\n=== co-attention 전/후, 원본 D차원 공간에서의 중심거리 비율 ===")
    for name, X in (("BEFORE(4관점 단순평균, co-attn 없음)", Xb), ("AFTER(RNA co-attention 가중합)", Xa)):
        r_ev = _ratio(X, ev_mask_low, ev_mask_high)
        r_risk = _ratio(X, risk_mask_low, risk_mask_high)
        print(f"  {name:38s} event비율={r_ev:.3f}  risk비율={r_risk:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    ev_colors = {0: "#a0aec0", 1: "#c53030"}
    tcolors = {0: "#2b6cb0", 1: "#a0aec0", 2: "#c53030"}

    for col, (name, X) in enumerate((("BEFORE (4관점 단순평균)", Xb), ("AFTER (RNA co-attention)", Xa))):
        tsne = TSNE(n_components=2, perplexity=args.perplexity, init="pca", random_state=args.seed)
        coords2d = tsne.fit_transform(X)
        df["tx"], df["ty"] = coords2d[:, 0], coords2d[:, 1]

        ax = axes[0][col]
        for e in (0, 1):
            sub = df[df["OS_event"] == e]
            ax.scatter(sub["tx"], sub["ty"], c=ev_colors[e], s=40, edgecolors="k", linewidths=0.3,
                       label={0: "censored", 1: "event"}[e])
        ax.set_title(f"{name}\ncolored by OS_event")
        ax.legend(fontsize=8)

        ax2 = axes[1][col]
        for t in (0, 1, 2):
            sub = df[df["risk_tertile"] == t]
            ax2.scatter(sub["tx"], sub["ty"], c=tcolors[t], s=40, edgecolors="k", linewidths=0.3,
                        label={0: "low", 1: "mid", 2: "high"}[t] + " risk")
        ax2.set_title(f"{name}\ncolored by model risk tertile")
        ax2.legend(fontsize=8)

    fig.suptitle(f"{external_dataset.upper()} — PMA component_coattn 전/후 비교, N={len(df)}")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"\n플롯 저장: {args.out}")


if __name__ == "__main__":
    main()
