"""
사용자 질문(2026-09-04): "예측 risk 말고, 그냥 객관적인 기준(OS 라벨)을 따라 그룹화해볼 순
없나?" — 모델이 예측한 risk score 대신, 실제 관찰된 OS_event(사망/생존)와 OS_time(생존기간)
자체로 WSI 임베딩 t-SNE를 색칠한다. 모델 예측이 약해서 분리가 안 보인 건지, 애초에 실제
라벨 기준으로도 WSI 임베딩이 분리가 안 되는지를 가른다 — 후자면 "이 WSI 인코더/코호트
규모에서는 조직 형태만으로 예후를 가르는 신호 자체가 약하다"는 더 강한 근거가 된다.

scripts/viz_wsi_embedding_tsne.py가 이미 저장해 둔 .scratch/wsi_embedding_tsne_cptac.csv
(case_id, tsne_x, tsne_y, risk, OS_time, OS_event, risk_tertile)를 그대로 재사용 — WSI
임베딩 재추출/재학습 불필요, t-SNE 좌표도 그대로(모델 output이 아니라 z_wsi 자체의 좌표라
라벨과 무관하게 고정돼 있음).

사용법:
    python -m scripts.viz_wsi_embedding_tsne_oslabel
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=str, default=".scratch/wsi_embedding_tsne_cptac.csv")
    parser.add_argument("--out", type=str, default=".scratch/wsi_embedding_tsne_cptac_oslabel.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"N={len(df)}, event={df['OS_event'].sum()}, censored={(df['OS_event'] == 0).sum()}")

    # 생존기간 중앙값 기준 장기/단기 생존 이분 — event=0(censored)인데 OS_time이 짧은 환자는
    # "단기 추적 후 소실"일 뿐 실제 나쁜 예후인지 알 수 없어 별도 표시(회색)한다. event=1(사망)
    # 환자만 short/long으로 명확히 나뉘고, censored는 중립색으로 깔아 왜곡을 피한다.
    median_time = df.loc[df["OS_event"] == 1, "OS_time"].median()
    print(f"사망 환자 기준 생존기간 중앙값 = {median_time:.0f}일")

    def _label(row):
        if row["OS_event"] == 0:
            return "censored (생존/소실)"
        return "event, short OS (<median)" if row["OS_time"] < median_time else "event, long OS (>=median)"

    df["os_group"] = df.apply(_label, axis=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # (a) OS_event만으로 이분(가장 객관적 — censoring 여부는 추적 종료 시점의 사실이라
    #     time cutoff 선택에 좌우되지 않는다).
    colors_event = {0: "#a0aec0", 1: "#c53030"}
    labels_event = {0: "censored (생존/추적 종료)", 1: "event (사망)"}
    for e in (0, 1):
        sub = df[df["OS_event"] == e]
        axes[0].scatter(sub["tsne_x"], sub["tsne_y"], c=colors_event[e], s=40,
                         edgecolors="k", linewidths=0.3, label=labels_event[e])
    axes[0].set_title("colored by OS_event (사망 여부)")
    axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")
    axes[0].legend(fontsize=8)

    # (b) 사망 환자만 단기/장기로 세분, censored는 중립색.
    colors_group = {
        "censored (생존/소실)": "#cbd5e0",
        "event, short OS (<median)": "#c53030",
        "event, long OS (>=median)": "#2b6cb0",
    }
    for g, c in colors_group.items():
        sub = df[df["os_group"] == g]
        axes[1].scatter(sub["tsne_x"], sub["tsne_y"], c=c, s=40,
                         edgecolors="k", linewidths=0.3, label=g)
    axes[1].set_title("colored by OS_event + 생존기간(사망자만 short/long)")
    axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")
    axes[1].legend(fontsize=7)

    fig.suptitle(f"CPTAC-PDA external cohort (N={len(df)}) — 동일 WSI 임베딩 t-SNE 좌표, 실제 OS 라벨로 색칠")
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"플롯 저장: {out_path}")

    # risk tertile 때와 동일한 정량 체크 — event vs censored 중심 거리 비율.
    from_cols = ["tsne_x", "tsne_y"]
    # t-SNE 2D 좌표가 아니라 원본 임베딩 공간에서 봐야 의미가 있지만, z_wsi 원본은 이 CSV에
    # 없으므로(용량 문제로 저장 안 함) 2D t-SNE 좌표 기준 중심 거리만 참고용으로 낸다 —
    # risk tertile 스크립트의 원본-공간 수치와 직접 비교는 불가, 같은 t-SNE 좌표 내
    # "패턴이 안 보이는 게 우연이 아니다"를 보여주는 보조 지표로만 사용.
    event_xy = df.loc[df["OS_event"] == 1, from_cols].to_numpy()
    cens_xy = df.loc[df["OS_event"] == 0, from_cols].to_numpy()
    center_dist = np.linalg.norm(event_xy.mean(0) - cens_xy.mean(0))
    overall_scale = np.linalg.norm(df[from_cols].to_numpy() - df[from_cols].to_numpy().mean(0), axis=1).mean()
    print(f"\n(t-SNE 2D 좌표 기준, 참고용) event vs censored 중심 거리 = {center_dist:.4f} "
          f"(전체 평균 반경 {overall_scale:.4f} 대비 비율 {center_dist / overall_scale:.3f})")


if __name__ == "__main__":
    main()
