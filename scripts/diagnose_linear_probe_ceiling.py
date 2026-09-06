"""
사용자 결정(2026-09-05): 클러스터 구성 자체에 생존 신호가 없었던 것(scripts/diagnose_cluster_
cox_regression.py, 11개 군집 전부 p>0.05)에 이어 — "이 feature(frozen UNI2-h)에 애초에 뽑을
survival 신호가 있긴 한지" 상한선을 확인한다. Nystrom/ABMIL/cluster_pool/co-attention 등
우리 아키텍처를 전혀 안 거치고, 환자당 전체 패치를 단순 평균한 raw feature(1536차원)에
표준화 + PCA(차원 축소, N~110 대비 과적합 방지) + CoxPH만 적용하는 가장 단순하고 강건한
방법으로 "linear-probe 상한선"을 잰다.

이게 M7(RNA+clinical only) 수준에도 못 미치면 "이 규모(N~110~150)에서 frozen WSI feature
자체엔 뽑을 게 별로 없다"는 뜻이고, M7만큼이라도 나오면 "정보는 있는데 우리 아키텍처가
그걸 못 뽑아내는 것"이라는 뜻이라 완전히 다른 결론으로 이어진다.

TCGA 내부는 5-fold CV(pooled out-of-fold, 우리 다른 실험들과 동일 관례)로, CPTAC은 TCGA 전체로
학습한 모델의 진짜 external 평가로 본다.

사용법:
    python -m scripts.diagnose_linear_probe_ceiling --n-pcs 10
    python -m scripts.diagnose_linear_probe_ceiling --n-pcs 5 20    # 여러 값 비교
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, pdac_consistency_gene_ids
from train import _identity_collate


def _load_meanpool(dataset: str, backbone: str, fold: int | None = None, n_folds: int = 5, split: str = "all"):
    cfg = Config()
    rna_gene_ids = pdac_consistency_gene_ids(1500)
    kwargs = dict(fold=fold, n_folds=n_folds) if fold is not None else {}
    ds = WSISurvivalDataset(
        cfg.data, dataset=dataset, split=split, feature_backbone=backbone,
        with_clinical=True, with_staging=True, with_margin=True, with_mutation=True,
        with_rna=True, with_cnv=True, rna_gene_ids=rna_gene_ids, **kwargs,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)
    X, times, events, case_ids = [], [], [], []
    for patient_slides in loader:
        if not patient_slides:
            continue
        raw = torch.cat([s["features"] for s in patient_slides], dim=0).float()
        X.append(raw.mean(dim=0).numpy())
        times.append(float(patient_slides[0]["OS_time"].item()))
        events.append(int(patient_slides[0]["OS_event"].item()))
        case_ids.append(patient_slides[0]["case_id"])
    return np.stack(X), np.array(times), np.array(events), case_ids


def _fit_predict(X_train, t_train, e_train, X_test, n_pcs: int):
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)
    pca = PCA(n_components=n_pcs, random_state=42).fit(Xtr)
    Ztr = pca.transform(Xtr)
    Zte = pca.transform(Xte)
    cols = [f"pc{i}" for i in range(n_pcs)]
    df_train = pd.DataFrame(Ztr, columns=cols)
    df_train["OS_time"] = t_train
    df_train["OS_event"] = e_train
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_train, duration_col="OS_time", event_col="OS_event")
    risk_test = cph.predict_log_partial_hazard(pd.DataFrame(Zte, columns=cols)).values
    return risk_test


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", type=str, default="uni2native")
    parser.add_argument("--n-pcs", type=int, nargs="+", default=[10])
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    print("TCGA/CPTAC mean-pooled raw feature 로드 중...")
    Xt_full, tt_full, et_full, _ = _load_meanpool("tcga", args.backbone, fold=None, split="all")
    Xc_full, tc_full, ec_full, _ = _load_meanpool("cptac", args.backbone, fold=None, split="all")
    print(f"tcga N={len(Xt_full)}, cptac N={len(Xc_full)}, raw_dim={Xt_full.shape[1]}")

    for n_pcs in args.n_pcs:
        print(f"\n########## n_pcs={n_pcs} ##########")

        # --- TCGA 내부: 5-fold CV pooled out-of-fold c-index ---
        pooled_risk, pooled_t, pooled_e = [], [], []
        for fold in range(args.n_folds):
            Xtr, ttr, etr, _ = _load_meanpool("tcga", args.backbone, fold=fold, n_folds=args.n_folds, split="train")
            Xva, tva, eva, _ = _load_meanpool("tcga", args.backbone, fold=fold, n_folds=args.n_folds, split="val")
            Xte, tte, ete, _ = _load_meanpool("tcga", args.backbone, fold=fold, n_folds=args.n_folds, split="test")
            Xtrain = np.concatenate([Xtr, Xva], axis=0)
            ttrain = np.concatenate([ttr, tva])
            etrain = np.concatenate([etr, eva])
            risk_te = _fit_predict(Xtrain, ttrain, etrain, Xte, n_pcs)
            pooled_risk.extend(list(risk_te)); pooled_t.extend(list(tte)); pooled_e.extend(list(ete))
        c_internal = concordance_index(pooled_t, -np.array(pooled_risk), pooled_e)
        print(f"TCGA internal(5-fold pooled out-of-fold) c-index = {c_internal:.4f} (N={len(pooled_t)})")

        # --- CPTAC external: TCGA 전체로 학습 -> CPTAC 전체 평가 ---
        risk_ext = _fit_predict(Xt_full, tt_full, et_full, Xc_full, n_pcs)
        c_external = concordance_index(tc_full, -risk_ext, ec_full)
        print(f"CPTAC external(TCGA 전체 학습) c-index = {c_external:.4f} (N={len(Xc_full)})")


if __name__ == "__main__":
    main()
