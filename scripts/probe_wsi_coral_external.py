"""
[CORAL 진단] WSI 브랜치의 external 성능이 "신호가 없어서" 낮은 건지 "도메인 시프트에 묻혀서"
낮은 건지 구분하기 위해, CPTAC(external) WSI 임베딩을 TCGA(train) 분포에 CORAL로 정렬한 뒤
scripts/probe_wsi_only_signal.py의 두 WSI-only 경로 (A)/(B)를 다시 평가한다.

CORAL(Sun & Saenko 2016, "Return of Frustratingly Easy Domain Adaptation"): 두 도메인의 feature
공분산을 맞추는 선형변환. 여기서는 이미 학습된 risk_head/probe를 재학습하지 않고 그대로 쓰는
상황이라, "target(CPTAC)을 whitening 후 source(TCGA train) 공분산으로 recolor"하는 방향을
쓴다 — source 분포에 맞춰 학습된 classifier에 target을 갖다 맞추는 실전형 변형.

[중요한 방법론적 주의] CORAL은 target(CPTAC)의 "라벨 없는" feature 분포(공분산)를 미리 봐야
정렬 변환을 계산할 수 있다. 즉 이 결과는 순수 zero-shot external 일반화가 아니라, "그 기관의
라벨 없는 데이터 분포는 알고 있는 상태에서의 적응형 예측"이다 — 아래 결과를 해석할 때
scripts/probe_wsi_only_signal.py의 원래(비정렬) 수치와 반드시 같이 봐야 한다.

해석 방향: CORAL 정렬 후에도 external c-index가 여전히 우연 수준이면 "WSI 표현 자체에 신호가
부족하다"는 가설이 강화되고, 정렬 후 뚜렷이 올라가면 "신호는 있는데 도메인 시프트에 묻혀
있었다"는 가설이 강화된다 — 이 갈림이 이번 진단의 목적이지, CORAL을 최종 채택 방법으로 쓰려는
게 아니다.

사용법: python scripts/probe_wsi_coral_external.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from lifelines import CoxPHFitter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, literature_guided_gene_ids_intersection
from models import ViT_PMA
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv, stage_stats_from_csv, STAGE_FIELDS
from models.rna_predictor import RNAPredictionHead
from utils.metrics import compute_survival_metrics

SEEDS = [42, 84, 126]
N_FOLDS = 5
CKPT_DIR = _ROOT / "models" / "checkpoint"
COV_EPS = 1.0  # 공분산 shrinkage — D=64에 표본이 ~120개뿐이라 역행렬 안정화에 필요


def _identity_collate(batch):
    return batch[0]


def _ckpt_path(seed: int, fold: int) -> Path:
    matches = list(CKPT_DIR.glob(
        f"survival_tcga_uni2_seed{seed}_*STG_R_DISP_COX_ADD_FOLD{fold}OF{N_FOLDS}_best_pma.pt"
    ))
    if len(matches) != 1:
        raise FileNotFoundError(f"seed={seed} fold={fold}: checkpoint {len(matches)}개 매칭됨")
    return matches[0]


def _build_model(device, rna_input_dim, age_mean, age_std, margin_stats, stage_stats, ckpt_path):
    cfg = Config()
    cfg.model.use_attn_dispersion = True
    model = ViT_PMA(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                     backbone="uni2", combine_mode="cox_add", use_margin=True, margin_stats=margin_stats,
                     use_age_sex=True, use_staging=True, stage_stats=stage_stats).to(device)
    model.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, rna_input_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def _collect(model, loader, device):
    out = []
    for patient_slides in loader:
        p = patient_slides[0]
        rna = p["rna"].to(device)
        z_rna = model.encode_rna(rna)
        comps, spatial_feats = [], []
        for slide in patient_slides:
            fwd = model(slide["coords"].to(device), features=slide["features"])
            comps.append(fwd["embed"])
            if "spatial_feat" in fwd:
                spatial_feats.append(fwd["spatial_feat"])
        patient_embed = torch.stack(comps).mean(dim=0)  # (4, D)
        spatial_feat = torch.stack(spatial_feats).mean(dim=0) if spatial_feats else None
        z_wsi, _ = model.component_coattn(patient_embed, z_rna)  # (D,) — (A) 경로용

        out.append({
            "case_id": p["case_id"], "time": float(p["OS_time"].item()), "event": int(p["OS_event"].item()),
            "h_mean": patient_embed[0].cpu().numpy(),   # (B) 경로용 — 순수 mean-pool, RNA 무관
            "z_wsi": z_wsi.cpu().numpy(),                # (A) 경로용 — RNA-guided co-attention 출력
            "z_rna_dim": z_rna.shape[0],
            "spatial_feat": spatial_feat,
        })
    return out


def _cov_mean(X, eps=COV_EPS):
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = (Xc.T @ Xc) / max(1, len(X) - 1) + eps * np.eye(X.shape[1])
    return cov, mean


def _sym_pow(C, power):
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.clip(eigvals, 1e-8, None)
    return eigvecs @ np.diag(eigvals ** power) @ eigvecs.T


def coral_align(X_target, X_source):
    """target(X_target)을 whitening 후 source 공분산으로 recolor — source로 학습된 head에 넣기 위함."""
    Cs, mean_s = _cov_mean(X_source)
    Ct, mean_t = _cov_mean(X_target)
    A = _sym_pow(Ct, -0.5) @ _sym_pow(Cs, 0.5)
    return (X_target - mean_t) @ A + mean_s


@torch.no_grad()
def _risk_A_ablation(model, X_wsi, spatial_feats, rna_dim):
    """(A) risk_head ablation — z_wsi(정렬 여부 무관하게 이미 계산된 값)로 risk 재계산."""
    risks = []
    for i in range(len(X_wsi)):
        z_wsi = torch.tensor(X_wsi[i], dtype=torch.float32, device=next(model.parameters()).device)
        fused = torch.cat([z_wsi, torch.zeros(rna_dim, device=z_wsi.device)], dim=-1)
        if spatial_feats[i] is not None:
            fused = torch.cat([fused, spatial_feats[i]], dim=-1)
        risk = model.risk_head(fused.unsqueeze(0)).view(1)
        risks.append(risk.item())
    return np.array(risks)


def _fit_probe(train_patients, n_components=8, penalizer=1.0):
    X_train = np.stack([p["h_mean"] for p in train_patients])
    times = np.array([p["time"] for p in train_patients])
    events = np.array([p["event"] for p in train_patients])
    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=n_components, random_state=0).fit(scaler.transform(X_train))
    Z_train = pca.transform(scaler.transform(X_train))
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(_to_cox_df(Z_train, times, events), duration_col="time", event_col="event")
    return scaler, pca, cph


def _to_cox_df(Z, times, events):
    import pandas as pd
    cols = {f"pc{i}": Z[:, i] for i in range(Z.shape[1])}
    cols["time"] = times
    cols["event"] = events
    return pd.DataFrame(cols)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rna_gene_ids = literature_guided_gene_ids_intersection(1500)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS["tcga"])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS["tcga"])
    stage_stats = stage_stats_from_csv(CLINICAL_PATHS["tcga"])
    ds_kwargs = dict(with_clinical=True, with_margin=True, with_staging=True, with_rna=True,
                      rna_gene_ids=rna_gene_ids, feature_backbone="uni2")

    ext_A_raw, ext_A_coral = defaultdict(list), defaultdict(list)
    ext_B_raw, ext_B_coral = defaultdict(list), defaultdict(list)
    ext_label = {}

    for seed in SEEDS:
        cfg = Config()
        cfg.data.seed = seed
        for fold in range(N_FOLDS):
            ckpt_path = _ckpt_path(seed, fold)
            model = _build_model(device, len(rna_gene_ids), age_mean, age_std, margin_stats, stage_stats, ckpt_path)

            train_ds = WSISurvivalDataset(cfg.data, dataset="tcga", split="train", fold=fold, n_folds=N_FOLDS, **ds_kwargs)
            ext_ds   = WSISurvivalDataset(cfg.data, dataset="cptac", split="all", **ds_kwargs)
            train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=_identity_collate)
            ext_loader   = DataLoader(ext_ds,   batch_size=1, shuffle=False, collate_fn=_identity_collate)

            train_p = _collect(model, train_loader, device)
            ext_p   = _collect(model, ext_loader, device)

            # ---- (A) risk_head ablation: z_wsi를 CORAL 정렬 전/후 둘 다 ----
            X_train_zwsi = np.stack([p["z_wsi"] for p in train_p])
            X_ext_zwsi   = np.stack([p["z_wsi"] for p in ext_p])
            spatial_feats = [p["spatial_feat"] for p in ext_p]

            rna_dim = ext_p[0]["z_rna_dim"]
            risk_A_raw   = _risk_A_ablation(model, X_ext_zwsi, spatial_feats, rna_dim)
            X_ext_zwsi_aligned = coral_align(X_ext_zwsi, X_train_zwsi)
            risk_A_coral = _risk_A_ablation(model, X_ext_zwsi_aligned, spatial_feats, rna_dim)

            # ---- (B) 독립 프로브: h_mean을 CORAL 정렬 전/후 둘 다, train에 fit한 probe로 예측 ----
            scaler, pca, cph = _fit_probe(train_p)
            X_train_hmean = np.stack([p["h_mean"] for p in train_p])
            X_ext_hmean   = np.stack([p["h_mean"] for p in ext_p])

            Z_ext_raw = pca.transform(scaler.transform(X_ext_hmean))
            risk_B_raw = cph.predict_log_partial_hazard(
                _to_cox_df(Z_ext_raw, np.zeros(len(ext_p)), np.zeros(len(ext_p)))
            ).to_numpy()

            X_ext_hmean_aligned = coral_align(X_ext_hmean, X_train_hmean)
            Z_ext_coral = pca.transform(scaler.transform(X_ext_hmean_aligned))
            risk_B_coral = cph.predict_log_partial_hazard(
                _to_cox_df(Z_ext_coral, np.zeros(len(ext_p)), np.zeros(len(ext_p)))
            ).to_numpy()

            for i, p in enumerate(ext_p):
                cid = p["case_id"]
                ext_A_raw[cid].append(float(risk_A_raw[i]))
                ext_A_coral[cid].append(float(risk_A_coral[i]))
                ext_B_raw[cid].append(float(risk_B_raw[i]))
                ext_B_coral[cid].append(float(risk_B_coral[i]))
                ext_label[cid] = (p["time"], p["event"])

            print(f"  [완료] seed={seed} fold={fold} (train N={len(train_p)}, external N={len(ext_p)})")

    print("\n" + "=" * 80)
    _report("(A) risk_head ablation — CORAL 정렬 전", ext_A_raw, ext_label)
    _report("(A) risk_head ablation — CORAL 정렬 후", ext_A_coral, ext_label)
    print()
    _report("(B) 독립 PCA+ridge-Cox 프로브 — CORAL 정렬 전", ext_B_raw, ext_label)
    _report("(B) 독립 PCA+ridge-Cox 프로브 — CORAL 정렬 후", ext_B_coral, ext_label)


def _report(title, patient_risks, label):
    case_ids = sorted(patient_risks.keys())
    ens = np.array([float(np.mean(patient_risks[c])) for c in case_ids])
    times = np.array([label[c][0] for c in case_ids])
    events = np.array([label[c][1] for c in case_ids])
    m = compute_survival_metrics(ens, times, events)
    print(f"--- external — {title} ---")
    print(f"  15실행 앙상블: N={len(case_ids)} c_index={m['c_index']:.4f} "
          f"HR={m['hr']:.3f} log_rank_p={m['log_rank_p']:.4f}")


if __name__ == "__main__":
    main()
