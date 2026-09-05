"""
사용자 질문(2026-09-04): "PNI/LVI는 데이터가 막혔으니, 분화도(tumor_grade)로 sanity check를
해보자" — data/clinical_{tcga,cptac}.csv에 이미 있는 병리의 판독 tumor_grade(G1/G2/G3)를
그대로 라벨로 써서, WSI 임베딩(patient-level pooled, 학습 안 된 원본 UNI2-h feature 평균)만으로
분화도를 분류할 수 있는지 본다. 외부 데이터셋 전혀 불필요 — 오늘 하루 종일 "생존과 상관있나"만
봤는데, 그 이전 단계인 "애초에 병리학적으로 의미있는 걸 구별할 능력이 있나"를 먼저 확인하는
sanity check. 여기서 못 맞추면 오늘 나온 모든 null 결과의 더 근본적인 원인(임베딩 자체의 한계)이
드러나는 셈이고, 맞추면 "형태 정보는 있는데 그게 이 코호트 예후엔 안 먹힌다"는 쪽으로 좁혀진다.

사용법:
    python -m scripts.diagnose_grade_from_wsi --dataset cptac
    python -m scripts.diagnose_grade_from_wsi --dataset tcga
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, pdac_consistency_gene_ids
from train import _identity_collate

CLINICAL_PATHS = {"tcga": "data/clinical_tcga.csv", "cptac": "data/clinical_cptac.csv"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default="cptac", choices=["tcga", "cptac"])
    parser.add_argument("--backbone", type=str, default="uni2native")
    parser.add_argument("--n-pca", type=int, default=20)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=84)
    args = parser.parse_args()

    grade_df = pd.read_csv(CLINICAL_PATHS[args.dataset])[["case_id", "tumor_grade"]]
    grade_df = grade_df[grade_df["tumor_grade"].isin(["G1", "G2", "G3"])].copy()
    print(f"{args.dataset} tumor_grade 분포(G1/G2/G3만): {grade_df['tumor_grade'].value_counts().to_dict()}")

    cfg = Config()
    rna_gene_ids = pdac_consistency_gene_ids(1500)
    ds = WSISurvivalDataset(
        cfg.data, dataset=args.dataset, split="all",
        with_clinical=True, with_staging=True, with_margin=True, with_mutation=True,
        with_rna=True, with_cnv=True, rna_gene_ids=rna_gene_ids, feature_backbone=args.backbone,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)

    grade_map = dict(zip(grade_df["case_id"], grade_df["tumor_grade"]))
    rows, vecs = [], []
    for patient_slides in loader:
        if not patient_slides:
            continue
        case_id = patient_slides[0]["case_id"]
        if case_id not in grade_map:
            continue
        feats = torch.cat([s["features"] for s in patient_slides], dim=0).float().numpy()
        vecs.append(feats.mean(axis=0))
        rows.append(grade_map[case_id])

    X = np.stack(vecs)
    y = np.array(rows)
    print(f"매칭된 환자 수: {len(y)}, D={X.shape[1]}")
    print(f"클래스 분포: {pd.Series(y).value_counts().to_dict()}")
    majority_acc = pd.Series(y).value_counts(normalize=True).max()
    print(f"다수 클래스(majority) baseline accuracy = {majority_acc:.4f}\n")

    n_pca = min(args.n_pca, len(y) - args.n_folds - 1)
    cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    # 2026-09-04: "선형으로만 안 보인 거 아니냐"(사용자 지적) — 로지스틱회귀(선형) 외에 비선형
    # 분류기(RBF-SVM, RandomForest) 둘 다 같은 PCA 임베딩·같은 CV split으로 나란히 비교.
    classifiers = {
        "LogisticRegression(선형)": LogisticRegression(max_iter=2000, C=0.1),
        "SVC-RBF(비선형)": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "RandomForest(비선형)": RandomForestClassifier(n_estimators=300, max_depth=4, random_state=args.seed),
    }
    print(f"=== WSI 원본 임베딩(원본 UNI2-h, patient-level mean-pool, PCA={n_pca}) -> tumor_grade 분류, "
          f"{args.n_folds}-fold CV, 분류기 3종 비교 ===")
    for name, clf in classifiers.items():
        pipe = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=n_pca, random_state=args.seed)), ("clf", clf)])
        y_pred = cross_val_predict(pipe, X, y, cv=cv)
        acc = accuracy_score(y, y_pred)
        bal_acc = balanced_accuracy_score(y, y_pred)
        print(f"\n--- {name} ---")
        print(f"accuracy = {acc:.4f} (majority baseline {majority_acc:.4f} 대비 {'+' if acc > majority_acc else ''}"
              f"{acc - majority_acc:.4f})")
        print(f"balanced_accuracy(macro recall) = {bal_acc:.4f} (랜덤 기준 {1/len(set(y)):.4f})")
        print(classification_report(y, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
