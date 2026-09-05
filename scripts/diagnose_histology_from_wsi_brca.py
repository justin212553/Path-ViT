"""
사용자 질문(2026-09-04): "PAAD의 표본 크기(100~140명)가 문제라면, 훨씬 큰 BRCA(N~1058)에서도
확인할 수 있나?" — scripts/diagnose_grade_from_wsi.py와 같은 sanity check(patient-level mean-
pooled WSI 임베딩만으로 병리학적으로 뚜렷한 카테고리를 분류할 수 있는지)를 BRCA로 재현한다.

BRCA는 분화도(tumor_grade)가 없다(GDC/cBioPortal 둘 다 확인 — 유방암은 Nottingham 등급 체계라
GDC의 범용 tumor_grade 필드에 안 들어감, scripts/extract_brca_labels.py 주석 참고). 대신 훨씬
더 고전적이고 문헌상 WSI로 더 잘 구별되는 것으로 알려진 축을 쓴다 — **조직학적 아형(histologic
subtype): 침윤성 관암(IDC, ICD-O-3 8500/3) vs 침윤성 소엽암(ILC, 8520/3)**. IDC는 응집력 있는
관/둥지 구조로 자라고, ILC는 E-cadherin 소실로 세포가 한 줄로 흩어져 침윤하는 뚜렷이 다른 성장
패턴이라, 계산병리학 문헌에서 H&E만으로 상당히 잘 구별된다고 보고된 고전적인 벤치마크 태스크다
(cBioPortal brca_tcga_pan_can_atlas_2018::ICD_O_3_HISTOLOGY, 768 IDC vs 199 ILC).

BRCA WSI feature는 uni2native가 아니라 UNI(v1, ViT-L/16, 1024-dim, scripts/brca_common.py::
BRCASlideDataset이 읽는 features_uni.pt) — PAAD 쪽과 backbone이 다르다는 점은 캐비어트로 남는다.

사용법:
    python -m scripts.diagnose_histology_from_wsi_brca
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

TILES_ROOT = Path("data/patches_tcga_brca/tiles")
MANIFEST_PATH = Path("data/brca_slide_manifest.csv")
CBIO_PATIENT_JSON = Path(".scratch/brca_clinical_patient.json")
HISTOLOGY_MAP = {"8500/3": "IDC", "8520/3": "ILC"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-pca", type=int, default=20)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=84)
    args = parser.parse_args()

    cbio = json.load(open(CBIO_PATIENT_JSON))
    cbio_df = pd.DataFrame(cbio)
    hist = cbio_df[cbio_df["clinicalAttributeId"] == "ICD_O_3_HISTOLOGY"][["patientId", "value"]]
    hist = hist[hist["value"].isin(HISTOLOGY_MAP)].copy()
    hist["label"] = hist["value"].map(HISTOLOGY_MAP)
    hist = hist.rename(columns={"patientId": "case_id"})
    print(f"ICD-O-3 조직형(IDC/ILC만): {hist['label'].value_counts().to_dict()}")

    manifest = pd.read_csv(MANIFEST_PATH)
    label_map = dict(zip(hist["case_id"], hist["label"]))

    rows, vecs = [], []
    for case_id, slide_rows in manifest.groupby("case_id"):
        if case_id not in label_map:
            continue
        feats_list = []
        for _, srow in slide_rows.iterrows():
            slide_dir = TILES_ROOT / srow["slide_id"]
            fpath = slide_dir / "features_uni.pt"
            if not fpath.exists():
                continue
            feats_list.append(torch.load(fpath, weights_only=True).float().numpy())
        if not feats_list:
            continue
        feats = np.concatenate(feats_list, axis=0)
        vecs.append(feats.mean(axis=0))
        rows.append(label_map[case_id])

    X = np.stack(vecs)
    y = np.array(rows)
    print(f"매칭된 환자 수: {len(y)}, D={X.shape[1]}")
    print(f"클래스 분포: {pd.Series(y).value_counts().to_dict()}")
    majority_acc = pd.Series(y).value_counts(normalize=True).max()
    print(f"다수 클래스(majority) baseline accuracy = {majority_acc:.4f}\n")

    n_pca = min(args.n_pca, len(y) - args.n_folds - 1)
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=args.seed)),
        ("clf", LogisticRegression(max_iter=2000, C=0.1)),
    ])
    cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    y_pred = cross_val_predict(pipe, X, y, cv=cv)
    y_proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

    acc = accuracy_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)
    auc = roc_auc_score((y == "ILC").astype(int), y_proba)
    print(f"=== BRCA WSI 임베딩(UNI v1, patient-level mean-pool, PCA={n_pca}) -> IDC/ILC 분류, "
          f"{args.n_folds}-fold CV, N={len(y)} ===")
    print(f"accuracy = {acc:.4f} (majority baseline {majority_acc:.4f} 대비 "
          f"{'+' if acc > majority_acc else ''}{acc - majority_acc:.4f})")
    print(f"balanced_accuracy(macro recall) = {bal_acc:.4f} (랜덤 기준 0.5)")
    print(f"ROC-AUC = {auc:.4f}")
    print("\n" + classification_report(y, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
