"""
사용자 질문(2026-09-05): "그렇다면 WSI 폴드의 문제인지 아닌지 확인하려면, 역시 모든 폴드에서
전부 확인해봐야겠네. ResNet50의 raw feature에도 똑같이 permutation test를 해봐." —
perm_test_coattn_ratio.py로 본 "가장 높은 점수"가 특정 fold의 학습 결과(co-attention의 기여)가
아니라 그 fold가 우연히 잘 나온 것일 수 있다는 문제 제기에 이어, 아예 모델 학습 자체가 없는
raw feature(원본 precomputed patch feature 단순평균, viz_raw_feature_tsne.py와 동일 추출)
단계에서부터 이미 event 분리가 유의한지를 확인한다. raw feature는 어떤 fold로 학습했는지와
무관하게 CPTAC 전체 코호트에서 항상 동일하므로, 여기서 유의하다면 "fold 운"이 아니라 backbone
자체(또는 실제 생물학적 신호)에 내재된 것이라는 뜻이다.

사용법:
    python -m scripts.perm_test_raw_feature_ratio --backbone resnet50 --n-perm 5000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, pdac_consistency_gene_ids
from train import _identity_collate


def _ratio(X, mask_low, mask_high):
    low, high = X[mask_low], X[mask_high]
    if mask_low.sum() < 2 or mask_high.sum() < 2:
        return float("nan")
    d = np.linalg.norm(low.mean(0) - high.mean(0))
    s = np.linalg.norm(X - X.mean(0), axis=1).mean()
    return d / s


def _perm_test(X, mask_low, mask_high, n_perm, rng):
    observed = _ratio(X, mask_low, mask_high)
    n_low, n_high = mask_low.sum(), mask_high.sum()
    idx_pool = np.where(mask_low | mask_high)[0]
    n_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(idx_pool)
        plow = np.zeros(len(X), dtype=bool)
        phigh = np.zeros(len(X), dtype=bool)
        plow[perm[:n_low]] = True
        phigh[perm[n_low:n_low + n_high]] = True
        r = _ratio(X, plow, phigh)
        if r >= observed:
            n_ge += 1
    return observed, (n_ge + 1) / (n_perm + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default="cptac", choices=["tcga", "cptac"])
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=84)
    args = parser.parse_args()

    cfg = Config()
    rna_gene_ids = pdac_consistency_gene_ids(1500)
    ds = WSISurvivalDataset(
        cfg.data, dataset=args.dataset, split="all",
        with_clinical=True, with_staging=True, with_margin=True, with_mutation=True,
        with_rna=True, with_cnv=True, rna_gene_ids=rna_gene_ids, feature_backbone=args.backbone,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)

    wsi_vecs, events = [], []
    for patient_slides in loader:
        if not patient_slides:
            continue
        all_feats = torch.cat([s["features"] for s in patient_slides], dim=0).float().numpy()
        wsi_vecs.append(all_feats.mean(axis=0))
        events.append(int(patient_slides[0]["OS_event"].item()))

    X = np.stack(wsi_vecs)
    events = np.array(events)
    ev_low, ev_high = events == 0, events == 1
    print(f"backbone={args.backbone}, N={len(X)} (event={ev_high.sum()}, censored={ev_low.sum()}), D={X.shape[1]}")

    rng = np.random.default_rng(args.seed)
    r, p = _perm_test(X, ev_low, ev_high, args.n_perm, rng)
    print(f"\n=== raw feature(모델 미개입, WSI 단순평균), permutation test (n_perm={args.n_perm}) ===")
    print(f"  event 분리: ratio={r:.3f}  p={p:.4f}")


if __name__ == "__main__":
    main()
