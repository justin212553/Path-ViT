"""
사용자 아이디어(2026-09-04): "그 가중치를 그대로 써보자 — BRCA에서 pretrain시킨 뒤 TCGA에서
시도." scripts/diagnose_histology_from_wsi_brca.py가 BRCA(N=934, UNI v1)로 IDC(응집성 관/둥지
구조) vs ILC(E-cadherin 소실, 한 줄로 흩어진 침윤) 분류기를 성공적으로 학습시켰다(AUC 0.925).
IDC/ILC라는 축 자체는 PDAC엔 없지만, "응집력 있게 조직화된 성장 vs 흩어진/미분화 성장"이라는
개념은 분화도(gland 형성 잘 됨=IDC와 결이 비슷 vs 안 됨=ILC와 결이 비슷)와 느슨하게 통할 수
있다 — 이 학습된 분류기의 결정 방향(PCA+로지스틱회귀 계수)을 그대로 얼려서, PAAD(TCGA/CPTAC)
환자들에게 통과시켜 "ILC-유사도 점수"를 뽑고, 그게 PAAD의 실제 분화도/생존과 관련 있는지 본다.

새로 학습하는 게 하나도 없다 — BRCA에서 fit한 파이프라인(scaler+PCA+LR)을 freeze해서 PAAD
WSI 임베딩(UNI v1, BRCA와 동일 백본 — data/dataset.py::feature_backbone="uni")에 그대로
transform+predict만 적용.

사용법:
    python -m scripts.transfer_brca_idcilc_to_paad
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, pdac_consistency_gene_ids
from train import _identity_collate
from utils.metrics import compute_survival_metrics

TILES_ROOT = Path("data/patches_tcga_brca/tiles")
MANIFEST_PATH = Path("data/brca_slide_manifest.csv")
CBIO_PATIENT_JSON = Path(".scratch/brca_clinical_patient.json")
HISTOLOGY_MAP = {"8500/3": "IDC", "8520/3": "ILC"}
GRADE_ORD = {"G1": 0, "G2": 1, "G3": 2}
CLINICAL_PATHS = {"tcga": "data/clinical_tcga.csv", "cptac": "data/clinical_cptac.csv"}


def _fit_brca_pipeline(n_pca=20, seed=84):
    cbio_df = pd.DataFrame(json.load(open(CBIO_PATIENT_JSON)))
    hist = cbio_df[cbio_df["clinicalAttributeId"] == "ICD_O_3_HISTOLOGY"][["patientId", "value"]]
    hist = hist[hist["value"].isin(HISTOLOGY_MAP)].copy()
    hist["label"] = hist["value"].map(HISTOLOGY_MAP)
    hist = hist.rename(columns={"patientId": "case_id"})
    label_map = dict(zip(hist["case_id"], hist["label"]))

    manifest = pd.read_csv(MANIFEST_PATH)
    rows, vecs = [], []
    for case_id, slide_rows in manifest.groupby("case_id"):
        if case_id not in label_map:
            continue
        feats_list = []
        for _, srow in slide_rows.iterrows():
            fpath = TILES_ROOT / srow["slide_id"] / "features_uni.pt"
            if fpath.exists():
                feats_list.append(torch.load(fpath, weights_only=True).float().numpy())
        if not feats_list:
            continue
        vecs.append(np.concatenate(feats_list, axis=0).mean(axis=0))
        rows.append(label_map[case_id])

    X = np.stack(vecs)
    y = np.array([1 if r == "ILC" else 0 for r in rows])  # 1=ILC(흩어진 침윤), 0=IDC(응집성)
    print(f"[BRCA pretrain] N={len(y)} (IDC={int((y==0).sum())}, ILC={int((y==1).sum())})")

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=seed)),
        ("clf", LogisticRegression(max_iter=2000, C=0.1)),
    ])
    pipe.fit(X, y)
    train_acc = pipe.score(X, y)
    print(f"[BRCA pretrain] 전체 데이터 fit 완료 (in-sample acc={train_acc:.4f}, 참고용 — 앞서 CV로 AUC 0.925 확인됨)")
    return pipe


def _paad_patient_embeddings(dataset: str, backbone: str = "uni"):
    cfg = Config()
    rna_gene_ids = pdac_consistency_gene_ids(1500)
    ds = WSISurvivalDataset(
        cfg.data, dataset=dataset, split="all",
        with_clinical=True, with_staging=True, with_margin=True, with_mutation=True,
        with_rna=True, with_cnv=True, rna_gene_ids=rna_gene_ids, feature_backbone=backbone,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)
    rows = []
    for patient_slides in loader:
        if not patient_slides:
            continue
        case_id = patient_slides[0]["case_id"]
        os_time = float(patient_slides[0]["OS_time"].item())
        os_event = int(patient_slides[0]["OS_event"].item())
        feats = torch.cat([s["features"] for s in patient_slides], dim=0).float().numpy()
        rows.append((case_id, feats.mean(axis=0), os_time, os_event))
    return rows


def main():
    pipe = _fit_brca_pipeline()

    grade_lookup = {}
    for ds_name, path in CLINICAL_PATHS.items():
        g = pd.read_csv(path)[["case_id", "tumor_grade"]]
        g = g[g["tumor_grade"].isin(GRADE_ORD)]
        grade_lookup.update(dict(zip(g["case_id"], g["tumor_grade"])))

    for ds_name in ("tcga", "cptac"):
        print(f"\n{'='*20} {ds_name} {'='*20}")
        rows = _paad_patient_embeddings(ds_name)
        X = np.stack([r[1] for r in rows])
        ilc_score = pipe.predict_proba(X)[:, 1]  # "ILC-유사도"(=응집성 낮음/미분화 유사) 점수
        case_ids = [r[0] for r in rows]
        os_time = np.array([r[2] for r in rows])
        os_event = np.array([r[3] for r in rows])

        df = pd.DataFrame({"case_id": case_ids, "ilc_score": ilc_score, "OS_time": os_time, "OS_event": os_event})
        df["grade"] = df["case_id"].map(grade_lookup)

        # (1) 생존과의 관계 — 오늘 하루 쓴 것과 동일한 관례.
        m_pos = compute_survival_metrics(df["ilc_score"].to_numpy(), os_time, os_event)
        m_neg = compute_survival_metrics(-df["ilc_score"].to_numpy(), os_time, os_event)
        best_c = max(m_pos["c_index"], m_neg["c_index"])
        best = m_pos if m_pos["c_index"] >= m_neg["c_index"] else m_neg
        print(f"[생존] ILC-유사도 점수 단독 c_index={best_c:.4f} HR={best['hr']:.3f} log_rank_p={best['log_rank_p']:.4f}")

        # (2) 분화도(G1<G2<G3)와의 관계 — ILC-유사도가 높을수록 미분화(G3)에 가까운지.
        sub = df.dropna(subset=["grade"]).copy()
        sub["grade_ord"] = sub["grade"].map(GRADE_ORD)
        if len(sub) > 5:
            rho, p = spearmanr(sub["ilc_score"], sub["grade_ord"])
            print(f"[분화도] ILC-유사도 vs grade(G1=0,G2=1,G3=2) Spearman rho={rho:.4f} p={p:.4f} (N={len(sub)})")
            g1 = sub[sub["grade"] == "G1"]["ilc_score"]
            g3 = sub[sub["grade"] == "G3"]["ilc_score"]
            if len(g1) > 1 and len(g3) > 1:
                u, p2 = mannwhitneyu(g1, g3)
                print(f"  G1(N={len(g1)}, mean={g1.mean():.3f}) vs G3(N={len(g3)}, mean={g3.mean():.3f}) "
                      f"Mann-Whitney p={p2:.4f}")


if __name__ == "__main__":
    main()
