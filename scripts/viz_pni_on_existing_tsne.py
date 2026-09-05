"""
사용자 아이디어(2026-09-04): "CPTAC엔 병리의가 판독한 PNI 라벨이 있으니, 새로 임베딩을 뽑지
말고 오늘 이미 만든 t-SNE 좌표들에 그 라벨만 색칠해서 다시 보자."

오늘 만든 5개 t-SNE 결과(WSI-pooled 모델토큰/원본평균, 패치 단위 모델토큰/원본, tumor-content
macro grid)는 전부 case_id+tsne_x+tsne_y를 CSV로 갖고 있다 — 임베딩 재추출/재학습 없이 PNI
라벨(scripts에서 받은 .scratch/cptac_clinical_sample_wide.csv::PERINEURAL_INVASION)만 merge해
다시 색칠하고, 지금까지와 동일한 중심거리 비율로 정량 비교한다.

사용법:
    python -m scripts.viz_pni_on_existing_tsne
"""
from pathlib import Path

import numpy as np
import pandas as pd

INPUTS = {
    "wsi_model_uni2native": ".scratch/wsi_embedding_tsne_cptac_uni2native.csv",
    "wsi_raw": ".scratch/raw_feature_tsne_cptac_wsi.csv",
    "patch_model_uni2native": ".scratch/patch_embedding_tsne_cptac_uni2native.csv",
    "patch_raw": ".scratch/raw_feature_tsne_cptac_patch.csv",
    "tumor_content_grid": ".scratch/tumor_content_grid_tsne_cptac.csv",
}
PNI_PATH = ".scratch/cptac_clinical_sample_wide.csv"
OUT_PNG = ".scratch/pni_on_existing_tsne.png"


def _ratio(df, x, y):
    xy = df[[x, y]].to_numpy()
    pos = df["pni"].to_numpy() == 1
    neg = df["pni"].to_numpy() == 0
    if pos.sum() < 2 or neg.sum() < 2:
        return float("nan")
    d = np.linalg.norm(xy[pos].mean(0) - xy[neg].mean(0))
    s = np.linalg.norm(xy - xy.mean(0), axis=1).mean()
    return d / s


def main():
    pni_df = pd.read_csv(PNI_PATH)[["patientId", "PERINEURAL_INVASION"]].rename(columns={"patientId": "case_id"})
    pni_df = pni_df[pni_df["PERINEURAL_INVASION"].isin(["Present", "Not identified"])].copy()
    pni_df["pni"] = (pni_df["PERINEURAL_INVASION"] == "Present").astype(int)
    print(f"PNI 라벨 보유(Present/Not identified만): N={len(pni_df)} "
          f"(Present={pni_df['pni'].sum()}, Not identified={(1 - pni_df['pni']).sum()})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = {k: v for k, v in INPUTS.items() if Path(v).exists()}
    fig, axes = plt.subplots(1, len(valid), figsize=(6 * len(valid), 5.5), squeeze=False)
    axes = axes[0]

    colors = {0: "#a0aec0", 1: "#c53030"}
    labels = {0: "PNI not identified", 1: "PNI present"}
    print("\n=== 표현별 PNI present vs not-identified 중심거리 비율 ===")
    for ax, (name, path) in zip(axes, valid.items()):
        df = pd.read_csv(path)
        merged = df.merge(pni_df[["case_id", "pni"]], on="case_id", how="inner")
        # patch-level CSV는 환자당 여러 행 — merge 자체는 그대로 되지만 N 표기만 환자 수 기준으로.
        n_patients = merged["case_id"].nunique()
        for p in (0, 1):
            sub = merged[merged["pni"] == p]
            ax.scatter(sub["tsne_x"], sub["tsne_y"], c=colors[p], s=(6 if len(merged) > 1000 else 40),
                       alpha=(0.5 if len(merged) > 1000 else 1.0),
                       edgecolors=None if len(merged) > 1000 else "k", linewidths=0.3, label=labels[p])
        ax.set_title(f"{name}\n(N_patient={n_patients}, N_row={len(merged)})", fontsize=9)
        ax.legend(fontsize=7)
        r = _ratio(merged, "tsne_x", "tsne_y")
        print(f"  {name:26s} 중심거리 비율={r:.3f}")

    fig.suptitle("CPTAC — 오늘 만든 기존 t-SNE 좌표에 실제 PNI(병리의 판독) 라벨만 재색칠")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"\n플롯 저장: {OUT_PNG}")


if __name__ == "__main__":
    main()
