"""
사용자 질문(2026-09-05): "클러스터 별 COX 회귀를 한번 돌려볼 수 있어?" — 딥러닝 파이프라인
전체(Nystrom/ABMIL/cluster_pool/co-attention)를 거치지 않고, "환자별로 K=11개 unsupervised
군집(data/cluster_centroids_uni2native.pt, TCGA-only 재적합)에 패치가 얼마나 분포하는가"라는
가장 단순한 통계량만으로 생존과 직접 연관이 있는지를 본다.

환자마다 (N_patches, raw_dim) -> 가장 가까운 군집 배정 -> 군집별 패치 비율(K-dim, 합=1)을
계산한 뒤, 각 군집 비율 하나씩을 univariate Cox 회귀(lifelines)에 넣어 HR/p-value를 본다.
클러스터풀 이후 성능이 안 나오는 게 "군집 구성 자체엔 신호가 없어서"인지 "신호는 있는데
그 뒤 aggregation/co-attention 단계에서 못 뽑아내는 건지"를 가르는 진단이다.

TCGA(internal)와 CPTAC(external) 둘 다 따로 돌려서, 방향이 일관되게 나오는 군집이 있는지도 본다
(한쪽에서만 유의하면 노이즈일 가능성, 양쪽에서 같은 방향으로 나오면 진짜 신호일 가능성).

사용법:
    python -m scripts.diagnose_cluster_cox_regression --dataset tcga
    python -m scripts.diagnose_cluster_cox_regression --dataset cptac
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, pdac_consistency_gene_ids
from train import _identity_collate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default="tcga", choices=["tcga", "cptac"])
    parser.add_argument("--backbone", type=str, default="uni2native")
    parser.add_argument("--centroids-path", type=str, default="data/cluster_centroids_uni2native.pt")
    args = parser.parse_args()

    cfg = Config()
    rna_gene_ids = pdac_consistency_gene_ids(1500)
    ds = WSISurvivalDataset(
        cfg.data, dataset=args.dataset, split="all",
        with_clinical=True, with_staging=True, with_margin=True, with_mutation=True,
        with_rna=True, with_cnv=True, rna_gene_ids=rna_gene_ids, feature_backbone=args.backbone,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)
    centroids = torch.load(args.centroids_path, weights_only=True).float()  # (K, raw_dim)
    k = centroids.shape[0]
    print(f"{args.dataset} N={len(ds)}, backbone={args.backbone}, K={k} ({args.centroids_path})")

    rows = []
    for patient_slides in loader:
        if not patient_slides:
            continue
        raw = torch.cat([s["features"] for s in patient_slides], dim=0).float()  # (N, raw_dim)
        assign = torch.cdist(raw, centroids).argmin(dim=1)  # (N,)
        counts = torch.bincount(assign, minlength=k).float()
        frac = (counts / counts.sum()).numpy()
        row = {f"cluster_{i}_frac": frac[i] for i in range(k)}
        row["OS_time"] = float(patient_slides[0]["OS_time"].item())
        row["OS_event"] = int(patient_slides[0]["OS_event"].item())
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"환자 {len(df)}명, events={int(df['OS_event'].sum())}")

    print(f"\n=== 군집별 patch 비율, univariate Cox 회귀 (dataset={args.dataset}) ===")
    print(f"{'cluster':>10} {'mean_frac':>10} {'HR':>8} {'95%CI':>18} {'p-value':>10}")
    results = []
    for i in range(k):
        col = f"cluster_{i}_frac"
        sub = df[[col, "OS_time", "OS_event"]].copy()
        if sub[col].std() < 1e-8:
            print(f"{'cluster_'+str(i):>10} {sub[col].mean():>10.4f}   (분산 0, 스킵)")
            continue
        cph = CoxPHFitter()
        try:
            cph.fit(sub, duration_col="OS_time", event_col="OS_event")
        except Exception as e:
            print(f"{'cluster_'+str(i):>10}   fit 실패: {e}")
            continue
        hr = np.exp(cph.params_[col])
        ci_low, ci_high = np.exp(cph.confidence_intervals_.loc[col])
        p = cph.summary.loc[col, "p"]
        results.append((i, sub[col].mean(), hr, ci_low, ci_high, p))
        flag = " *" if p < 0.05 else ""
        print(f"{'cluster_'+str(i):>10} {sub[col].mean():>10.4f} {hr:>8.3f} [{ci_low:>6.3f},{ci_high:>6.3f}] {p:>10.4f}{flag}")

    n_sig = sum(1 for r in results if r[5] < 0.05)
    print(f"\n총 {len(results)}개 군집 검정, p<0.05 인 것: {n_sig}개")
    print("(다중비교 보정 없음 — 참고용. Bonferroni 기준이면 p < {:.4f} 필요)".format(0.05 / max(len(results), 1)))


if __name__ == "__main__":
    main()
