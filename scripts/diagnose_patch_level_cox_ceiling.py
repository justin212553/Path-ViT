"""
사용자 지적(2026-09-05): "환자 단위 mean pool이면 있는 신호도 다 희석되지 않겠어? 패치끼리
잘 구분도 안 되는 마당에." — scripts/diagnose_linear_probe_ceiling.py(환자당 전체 패치를
단순평균한 뒤 PCA+Cox)는 사실 ABMIL이 실패한 것과 똑같은 "N개 패치 평균으로 희석" 문제를
그대로 반복하는 편향된 상한선 테스트였다. 여기서는 패치를 절대 평균 내지 않고, **패치 하나하나를
독립적인 관측치처럼 취급**해 PCA+CoxPH를 돌린다 — 대신 같은 환자의 패치들은 라벨(OS_time/
OS_event)이 중복되므로, patient-clustered robust standard error(lifelines CoxPHFitter의
cluster_col)로 유의성 추정을 보정한다. 이게 진짜 "이 feature에 patch 단위로 뽑을 수 있는
신호가 있는가"에 대한 공정한 상한선 테스트다.

평가(concordance)는 patch 단위로는 정의가 안 되니(같은 환자 패치들이 서로 순서를 매길 이유가
없음), fold별로 patch-level Cox 모델을 학습한 뒤 held-out 환자의 패치별 예측 risk를 환자
단위로 mean/max/top10%-mean 등 여러 방식으로 집계해 patient-level c-index를 낸다 — "patch
자체엔 신호가 있는데 지금 방식(mean)의 집계가 나쁜 건지" vs "애초에 patch 단위에도 신호가
없는지"를 집계 방식 여러 개로 교차검증해서 가른다.

사용법:
    python -m scripts.diagnose_patch_level_cox_ceiling --n-pcs 10 --max-patches-per-patient 300
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


def _load_patches(dataset: str, backbone: str, max_patches_per_patient: int, seed: int,
                   fold: int | None = None, n_folds: int = 5, split: str = "all"):
    """반환: patch_X(list of (n_i, raw_dim) 배열), 환자별 case_id/time/event(길이=환자 수)."""
    cfg = Config()
    rna_gene_ids = pdac_consistency_gene_ids(1500)
    kwargs = dict(fold=fold, n_folds=n_folds) if fold is not None else {}
    ds = WSISurvivalDataset(
        cfg.data, dataset=dataset, split=split, feature_backbone=backbone,
        with_clinical=True, with_staging=True, with_margin=True, with_mutation=True,
        with_rna=True, with_cnv=True, rna_gene_ids=rna_gene_ids, **kwargs,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)
    rng = np.random.default_rng(seed)
    patient_feats, case_ids, times, events = [], [], [], []
    for patient_slides in loader:
        if not patient_slides:
            continue
        raw = torch.cat([s["features"] for s in patient_slides], dim=0).float().numpy()
        n = raw.shape[0]
        if max_patches_per_patient > 0 and n > max_patches_per_patient:
            idx = rng.choice(n, max_patches_per_patient, replace=False)
            raw = raw[idx]
        patient_feats.append(raw)
        case_ids.append(patient_slides[0]["case_id"])
        times.append(float(patient_slides[0]["OS_time"].item()))
        events.append(int(patient_slides[0]["OS_event"].item()))
    return patient_feats, case_ids, np.array(times), np.array(events)


def _fit_patch_cox(patient_feats, case_ids, times, events, n_pcs: int, penalizer: float):
    scaler = StandardScaler().fit(np.concatenate(patient_feats, axis=0))
    all_scaled = np.concatenate([scaler.transform(f) for f in patient_feats], axis=0)
    pca = PCA(n_components=n_pcs, random_state=42).fit(all_scaled)

    rows = []
    for feats, cid, t, e in zip(patient_feats, case_ids, times, events):
        z = pca.transform(scaler.transform(feats))  # (n_i, n_pcs)
        for row in z:
            rows.append({**{f"pc{i}": row[i] for i in range(n_pcs)},
                         "OS_time": t, "OS_event": e, "case_id": cid})
    df = pd.DataFrame(rows)
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, duration_col="OS_time", event_col="OS_event", cluster_col="case_id")
    return scaler, pca, cph


def _predict_patient_risks(scaler, pca, cph, patient_feats, n_pcs: int):
    """각 환자 patch별 risk를 예측한 뒤 mean/max/top10pct_mean 세 가지로 집계."""
    agg = {"mean": [], "max": [], "top10pct_mean": []}
    for feats in patient_feats:
        z = pca.transform(scaler.transform(feats))
        cols = [f"pc{i}" for i in range(n_pcs)]
        risk = cph.predict_log_partial_hazard(pd.DataFrame(z, columns=cols)).values
        agg["mean"].append(risk.mean())
        agg["max"].append(risk.max())
        k = max(1, int(np.ceil(len(risk) * 0.1)))
        agg["top10pct_mean"].append(np.sort(risk)[-k:].mean())
    return agg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", type=str, default="uni2native")
    parser.add_argument("--n-pcs", type=int, default=10)
    parser.add_argument("--max-patches-per-patient", type=int, default=300)
    parser.add_argument("--penalizer", type=float, default=0.5)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"n_pcs={args.n_pcs}, max_patches_per_patient={args.max_patches_per_patient}, penalizer={args.penalizer}")

    print("\n=== TCGA internal 5-fold CV (pooled out-of-fold) ===")
    pooled = {"mean": [], "max": [], "top10pct_mean": []}
    pooled_t, pooled_e = [], []
    for fold in range(args.n_folds):
        tr_feats, tr_ids, tr_t, tr_e = _load_patches("tcga", args.backbone, args.max_patches_per_patient, args.seed, fold=fold, n_folds=args.n_folds, split="train")
        va_feats, va_ids, va_t, va_e = _load_patches("tcga", args.backbone, args.max_patches_per_patient, args.seed, fold=fold, n_folds=args.n_folds, split="val")
        te_feats, te_ids, te_t, te_e = _load_patches("tcga", args.backbone, args.max_patches_per_patient, args.seed, fold=fold, n_folds=args.n_folds, split="test")
        train_feats = tr_feats + va_feats
        train_ids = tr_ids + va_ids
        train_t = np.concatenate([tr_t, va_t])
        train_e = np.concatenate([tr_e, va_e])
        scaler, pca, cph = _fit_patch_cox(train_feats, train_ids, train_t, train_e, args.n_pcs, args.penalizer)
        agg = _predict_patient_risks(scaler, pca, cph, te_feats, args.n_pcs)
        for k in pooled:
            pooled[k].extend(agg[k])
        pooled_t.extend(list(te_t)); pooled_e.extend(list(te_e))
        print(f"  fold {fold} 완료 (train patients={len(train_feats)}, test patients={len(te_feats)})")

    print(f"\nTCGA internal pooled c-index (N={len(pooled_t)}):")
    for k in pooled:
        c = concordance_index(pooled_t, -np.array(pooled[k]), pooled_e)
        print(f"  집계={k:15s} c-index={c:.4f}")

    print("\n=== CPTAC external (TCGA 전체로 학습) ===")
    full_feats, full_ids, full_t, full_e = _load_patches("tcga", args.backbone, args.max_patches_per_patient, args.seed, fold=None, split="all")
    scaler, pca, cph = _fit_patch_cox(full_feats, full_ids, full_t, full_e, args.n_pcs, args.penalizer)
    ext_feats, ext_ids, ext_t, ext_e = _load_patches("cptac", args.backbone, args.max_patches_per_patient, args.seed, fold=None, split="all")
    agg = _predict_patient_risks(scaler, pca, cph, ext_feats, args.n_pcs)
    print(f"CPTAC external c-index (N={len(ext_t)}):")
    for k in agg:
        c = concordance_index(ext_t, -np.array(agg[k]), ext_e)
        print(f"  집계={k:15s} c-index={c:.4f}")


if __name__ == "__main__":
    main()
