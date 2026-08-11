"""
[조사 2] WSI patch feature(UNI2-h, features_uni2.pt) 자체가 TCGA/CPTAC 두 기관 사이에서
도메인 시프트(스캐너·염색·전처리 차이로 인한 institution 클러스터링)를 보이는지 확인한다.

학습된 PMA checkpoint를 전혀 쓰지 않는다 — WSISurvivalDataset이 precomputed=True로 돌려주는
raw UNI2-h 패치 feature(슬라이드당 (N_patch, 1536))를 패치 축으로 mean-pool해 슬라이드 벡터를,
슬라이드 축으로 다시 mean-pool해 환자 벡터를 만든다. 어떤 학습 가중치에도 의존하지 않는(고정된
사전학습 backbone 출력만 쓰는) 가장 순수한 도메인 시프트 진단 — "모델이 도메인을 잘 못 배워서"
문제인지 "도메인 자체가 원래 다르게 생겨서"(그래서 어떤 모델을 써도 internal/external gap이
어느 정도는 필연적인지) 구분하는 게 목적이다.

산출:
  1. TCGA/CPTAC 전체 환자(152+144명) 환자 벡터에 StandardScaler+PCA(50)+t-SNE(2) 적용,
     institution 색으로 산점도 저장 (.logs/domain_shift_tsne.png), PCA(2) 산점도도 별도 저장
     (.logs/domain_shift_pca.png).
  2. institution을 라벨로 한 5-fold CV 로지스틱회귀 AUC — 정량적 분리도. 0.5에 가까우면 도메인
     구분이 어렵다(=분포가 비슷하다)는 뜻, 1.0에 가까우면 raw feature만 보고도 어느 기관
     슬라이드인지 거의 완벽하게 맞힐 수 있다는 뜻(강한 batch effect).
  3. 환자 단위 벡터를 .logs/domain_shift_patient_embeds.npz에 저장(재사용/추가 분석용).

사용법: python scripts/probe_wsi_domain_shift.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset

OUT_DIR = _ROOT / ".logs"


def _identity_collate(batch):
    return batch[0]


def _collect_patient_vectors(cfg, dataset_name: str):
    ds = WSISurvivalDataset(cfg.data, dataset=dataset_name, split="all", feature_backbone="uni2")
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate)
    case_ids, vecs, events, times = [], [], [], []
    for patient_slides in loader:
        p = patient_slides[0]
        slide_means = [slide["features"].mean(dim=0).numpy() for slide in patient_slides]
        vecs.append(np.mean(slide_means, axis=0))
        case_ids.append(p["case_id"])
        events.append(int(p["OS_event"].item()))
        times.append(float(p["OS_time"].item()))
    return case_ids, np.stack(vecs), np.array(events), np.array(times)


def main():
    cfg = Config()
    print("TCGA 환자 벡터 수집 중...")
    tcga_ids, tcga_X, tcga_events, tcga_times = _collect_patient_vectors(cfg, "tcga")
    print(f"  N={len(tcga_ids)}")
    print("CPTAC 환자 벡터 수집 중...")
    cptac_ids, cptac_X, cptac_events, cptac_times = _collect_patient_vectors(cfg, "cptac")
    print(f"  N={len(cptac_ids)}")

    X = np.concatenate([tcga_X, cptac_X], axis=0)
    institution = np.array([0] * len(tcga_ids) + [1] * len(cptac_ids))  # 0=tcga, 1=cptac
    events = np.concatenate([tcga_events, cptac_events])
    case_ids = tcga_ids + cptac_ids

    np.savez(OUT_DIR / "domain_shift_patient_embeds.npz",
             X=X, institution=institution, events=events, case_ids=np.array(case_ids))
    print(f"환자 벡터 저장: {OUT_DIR / 'domain_shift_patient_embeds.npz'} (X shape={X.shape})")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # 1) 정량적 분리도: institution을 라벨로 한 5-fold CV 로지스틱회귀 AUC
    clf = LogisticRegression(max_iter=2000, C=1.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    aucs = cross_val_score(clf, Xs, institution, cv=cv, scoring="roc_auc")
    print(f"\n=== institution 분류기 (raw 1536차원, 5-fold CV) ===")
    print(f"  AUC: mean={aucs.mean():.4f} std={aucs.std():.4f} (개별: {np.round(aucs, 4).tolist()})")
    print(f"  (0.5=완전히 구분 안 됨/도메인 시프트 없음, 1.0=raw feature만으로 기관을 거의 완벽히 맞힘)")

    # PCA(50)로 축소한 뒤에도 같은 체크 (과적합 방지, t-SNE 입력과 동일 전처리로 일관성 유지)
    pca50 = PCA(n_components=min(50, X.shape[0] - 1), random_state=0).fit(Xs)
    Xp = pca50.transform(Xs)
    aucs_pca = cross_val_score(clf, Xp, institution, cv=cv, scoring="roc_auc")
    print(f"  (PCA-50 축소 후 동일 체크: mean={aucs_pca.mean():.4f} std={aucs_pca.std():.4f})")

    # 2) PCA(2) 산점도
    pca2 = PCA(n_components=2, random_state=0).fit(Xs)
    Z_pca = pca2.transform(Xs)
    _scatter(Z_pca, institution, events,
              f"WSI patch feature PCA (institution 분류기 AUC={aucs.mean():.3f})",
              OUT_DIR / "domain_shift_pca.png",
              xlabel=f"PC1 ({pca2.explained_variance_ratio_[0]:.1%})",
              ylabel=f"PC2 ({pca2.explained_variance_ratio_[1]:.1%})")
    print(f"저장: {OUT_DIR / 'domain_shift_pca.png'}")

    # 3) t-SNE(2), PCA-50 위에서 계산(권장 관례)
    tsne = TSNE(n_components=2, random_state=0, perplexity=30, init="pca")
    Z_tsne = tsne.fit_transform(Xp)
    _scatter(Z_tsne, institution, events,
              f"WSI patch feature t-SNE (institution 분류기 AUC={aucs.mean():.3f})",
              OUT_DIR / "domain_shift_tsne.png",
              xlabel="t-SNE 1", ylabel="t-SNE 2")
    print(f"저장: {OUT_DIR / 'domain_shift_tsne.png'}")


def _scatter(Z, institution, events, title, save_path, xlabel, ylabel):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    colors = np.where(institution == 0, "#4C72B0", "#DD8452")
    for inst, label, c in [(0, "TCGA", "#4C72B0"), (1, "CPTAC", "#DD8452")]:
        mask = institution == inst
        axes[0].scatter(Z[mask, 0], Z[mask, 1], s=18, alpha=0.7, c=c, label=label, edgecolors="none")
    axes[0].set_title("institution")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].legend()

    for ev, label, c in [(0, "censored", "#55A868"), (1, "event(사망)", "#C44E52")]:
        mask = events == ev
        axes[1].scatter(Z[mask, 0], Z[mask, 1], s=18, alpha=0.7, c=c, label=label, edgecolors="none")
    axes[1].set_title("OS_event")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
