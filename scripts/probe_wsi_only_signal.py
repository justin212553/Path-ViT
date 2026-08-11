"""
[조사 1] WSI 브랜치가 RNA/clinical 없이 그 자체로 얼마나 예후 신호를 담고 있는지 격리해서 확인한다.

baseline PMA(uni2, cox_add, STG+R) checkpoint 15개(3seed x 5fold, models/checkpoint/에 이미
저장됨)를 재학습 없이 그대로 불러와, 환자당 두 가지 "WSI-only" risk를 계산한다:

(A) risk_head ablation: 학습된 가중치는 그대로 두고, risk_head에 들어가는 fused 벡터에서 z_rna
    부분만 0으로 지우고 clinical_linear 가산항도 아예 더하지 않는다. z_wsi 자체는 원래대로
    component_coattn(patient_embed, z_rna)로 계산하므로(RNA가 "4개 WSI 관점 중 뭘 볼지"
    query로는 여전히 관여) 완전한 RNA 무관은 아니지만, risk_head가 최종적으로 보는 입력에서
    RNA 항을 제거한 것 — models/vit_pma.py의 rna_gate_only 설계와 동일한 의미의 사후 ablation.
    이미 이 프로젝트에서 검증된 zero-ablation 방법론(scripts/diagnose_pma_component_reliance.py)의
    직접 확장이다. 한계: risk_head 가중치 자체가 RNA/clinical과 공동학습됐으므로, 이 확률은
    "그 가중치가 WSI만으로 낼 수 있는 최선"이지 완전히 독립적인 신호 검증은 아니다.

(B) 독립 프로브: (A)의 한계를 보완하기 위해, 학습된 risk_head를 아예 쓰지 않고 WSI mean-pooled
    임베딩(patient_embed의 "mean" 성분, RNA/attention 전혀 관여 안 함 — 가장 순수한 WSI 표현)에
    PCA(8차원)로 축소 후 별도의 ridge-정규화 CoxPHFitter를 그 fold의 train split에서만 새로
    fit해서 test/external을 예측한다. risk_head와 완전히 독립적인, WSI 표현 자체의 생존 신호
    존재 여부를 보는 깨끗한 검증.

baseline(전체 모델, RNA+clinical+WSI 다 사용) 재계산도 sanity check로 같이 낸다 — 이게 기존에
보고된 internal 0.6359 / external 0.6337 근처로 나와야 아래 ablation 수치를 믿을 수 있다.

내부(internal)는 기존 멀티시드 프로토콜과 동일하게 시드별 5-fold OOF를 pool한 뒤 환자 단위로
시드 간 평균, external은 15개 실행 예측을 환자 단위로 그대로 평균한다
(scripts/pool_multiseed_kfold_preds.py, scripts/pool_multiseed_external_preds.py와 동일 원칙).

사용법: python scripts/probe_wsi_only_signal.py
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


def _identity_collate(batch):
    return batch[0]


def _ckpt_path(seed: int, fold: int) -> Path:
    matches = list(CKPT_DIR.glob(
        f"survival_tcga_uni2_seed{seed}_*STG_R_DISP_COX_ADD_FOLD{fold}OF{N_FOLDS}_best_pma.pt"
    ))
    if len(matches) != 1:
        raise FileNotFoundError(f"seed={seed} fold={fold}: checkpoint {len(matches)}개 매칭됨 ({matches})")
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
    """환자별 (4,D) patient_embed, z_rna, clin_raw, spatial_feat, time, event, case_id."""
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

        age_years = p["age_years"].to(device)
        sex_idx = p["sex_idx"].to(device)
        margin_ord = p["margin_ord"].to(device)
        stage_ord = {f: p[f].to(device) for f in STAGE_FIELDS}
        clin_raw = model._clinical_raw(age_years, sex_idx, margin_ord, stage_ord=stage_ord)

        out.append({
            "case_id": p["case_id"], "time": float(p["OS_time"].item()), "event": int(p["OS_event"].item()),
            "patient_embed": patient_embed, "z_rna": z_rna, "spatial_feat": spatial_feat, "clin_raw": clin_raw,
        })
    return out


@torch.no_grad()
def _risk_full(model, patients):
    """sanity check용 — 원래 baseline과 동일한 전체 모델 risk."""
    risks = []
    for pd in patients:
        z_wsi, _ = model.component_coattn(pd["patient_embed"], pd["z_rna"])
        fused = torch.cat([z_wsi, pd["z_rna"]], dim=-1)
        if pd["spatial_feat"] is not None:
            fused = torch.cat([fused, pd["spatial_feat"]], dim=-1)
        risk = model.risk_head(fused.unsqueeze(0)).view(1)
        risk = risk + model.clinical_linear(pd["clin_raw"]).view(1)
        risks.append(risk.item())
    return np.array(risks)


@torch.no_grad()
def _risk_wsi_only_ablation(model, patients):
    """(A) — risk_head 입력에서 z_rna를 0으로, clinical 가산항은 아예 생략."""
    risks = []
    for pd in patients:
        z_wsi, _ = model.component_coattn(pd["patient_embed"], pd["z_rna"])  # RNA는 query로만 관여
        fused = torch.cat([z_wsi, torch.zeros_like(pd["z_rna"])], dim=-1)
        if pd["spatial_feat"] is not None:
            fused = torch.cat([fused, pd["spatial_feat"]], dim=-1)
        risk = model.risk_head(fused.unsqueeze(0)).view(1)
        risks.append(risk.item())
    return np.array(risks)


def _h_mean(patients):
    """(B)용 — 4개 성분 중 순수 mean-pool만(index 0), RNA/attention 전혀 관여 안 함."""
    return np.stack([pd["patient_embed"][0].cpu().numpy() for pd in patients])


def _fit_and_predict_probe(train_patients, eval_patients_list, n_components=8, penalizer=1.0):
    """train_patients로 StandardScaler+PCA+CoxPHFitter를 fit, eval_patients_list(여러 그룹)의 risk를 각각 예측."""
    X_train = _h_mean(train_patients)
    times = np.array([pd["time"] for pd in train_patients])
    events = np.array([pd["event"] for pd in train_patients])

    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=n_components, random_state=0).fit(scaler.transform(X_train))
    Z_train = pca.transform(scaler.transform(X_train))

    cph = CoxPHFitter(penalizer=penalizer)
    df_train = _to_cox_df(Z_train, times, events)
    cph.fit(df_train, duration_col="time", event_col="event")

    results = []
    for eval_patients in eval_patients_list:
        X_eval = _h_mean(eval_patients)
        Z_eval = pca.transform(scaler.transform(X_eval))
        df_eval = _to_cox_df(Z_eval, np.zeros(len(eval_patients)), np.zeros(len(eval_patients)))
        risk = cph.predict_log_partial_hazard(df_eval).to_numpy()
        results.append(risk)
    return results


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

    # internal: seed -> case_id -> risk (OOF, 5-fold pooled)
    internal_full = {s: {} for s in SEEDS}
    internal_ablA = {s: {} for s in SEEDS}
    internal_probeB = {s: {} for s in SEEDS}
    internal_label = {}

    # external: (seed,fold) 실행별 case_id -> risk 리스트
    ext_full = defaultdict(list)
    ext_ablA = defaultdict(list)
    ext_probeB = defaultdict(list)
    ext_label = {}

    for seed in SEEDS:
        cfg = Config()
        cfg.data.seed = seed
        for fold in range(N_FOLDS):
            ckpt_path = _ckpt_path(seed, fold)
            model = _build_model(device, len(rna_gene_ids), age_mean, age_std, margin_stats, stage_stats, ckpt_path)

            train_ds = WSISurvivalDataset(cfg.data, dataset="tcga", split="train", fold=fold, n_folds=N_FOLDS, **ds_kwargs)
            test_ds  = WSISurvivalDataset(cfg.data, dataset="tcga", split="test",  fold=fold, n_folds=N_FOLDS, **ds_kwargs)
            ext_ds   = WSISurvivalDataset(cfg.data, dataset="cptac", split="all", **ds_kwargs)

            train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=_identity_collate)
            test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, collate_fn=_identity_collate)
            ext_loader   = DataLoader(ext_ds,   batch_size=1, shuffle=False, collate_fn=_identity_collate)

            train_p = _collect(model, train_loader, device)
            test_p  = _collect(model, test_loader, device)
            ext_p   = _collect(model, ext_loader, device)

            full_test, full_ext = _risk_full(model, test_p), _risk_full(model, ext_p)
            ablA_test, ablA_ext = _risk_wsi_only_ablation(model, test_p), _risk_wsi_only_ablation(model, ext_p)
            probeB_test, probeB_ext = _fit_and_predict_probe(train_p, [test_p, ext_p])

            for i, pd_ in enumerate(test_p):
                cid = pd_["case_id"]
                internal_full[seed][cid] = float(full_test[i])
                internal_ablA[seed][cid] = float(ablA_test[i])
                internal_probeB[seed][cid] = float(probeB_test[i])
                internal_label[cid] = (pd_["time"], pd_["event"])
            for i, pd_ in enumerate(ext_p):
                cid = pd_["case_id"]
                ext_full[cid].append(float(full_ext[i]))
                ext_ablA[cid].append(float(ablA_ext[i]))
                ext_probeB[cid].append(float(probeB_ext[i]))
                ext_label[cid] = (pd_["time"], pd_["event"])

            print(f"  [완료] seed={seed} fold={fold} (test N={len(test_p)}, external N={len(ext_p)})")

    print("\n" + "=" * 80)
    _report_internal("전체 baseline (sanity check)", internal_full, internal_label)
    _report_internal("(A) risk_head ablation — RNA/clinical=0, WSI만", internal_ablA, internal_label)
    _report_internal("(B) 독립 PCA+ridge-Cox 프로브 — WSI mean-pool만", internal_probeB, internal_label)

    print("\n" + "=" * 80)
    _report_external("전체 baseline (sanity check)", ext_full, ext_label)
    _report_external("(A) risk_head ablation — RNA/clinical=0, WSI만", ext_ablA, ext_label)
    _report_external("(B) 독립 PCA+ridge-Cox 프로브 — WSI mean-pool만", ext_probeB, ext_label)


def _report_internal(title, per_seed_risks, label):
    print(f"\n--- internal — {title} ---")
    seed_c = []
    for seed in SEEDS:
        case_ids = list(per_seed_risks[seed].keys())
        risks = np.array([per_seed_risks[seed][c] for c in case_ids])
        times = np.array([label[c][0] for c in case_ids])
        events = np.array([label[c][1] for c in case_ids])
        m = compute_survival_metrics(risks, times, events)
        seed_c.append(m["c_index"])
        print(f"  seed={seed}: N={len(case_ids)} c_index={m['c_index']:.4f}")
    seed_c = np.array(seed_c)
    print(f"  seed 간: mean={seed_c.mean():.4f} std={seed_c.std():.4f}")

    common = sorted(set.intersection(*(set(per_seed_risks[s].keys()) for s in SEEDS)))
    ens = np.array([np.mean([per_seed_risks[s][c] for s in SEEDS]) for c in common])
    times = np.array([label[c][0] for c in common])
    events = np.array([label[c][1] for c in common])
    m = compute_survival_metrics(ens, times, events)
    print(f"  -> 3seed 앙상블: N={len(common)} c_index={m['c_index']:.4f}")


def _report_external(title, patient_risks, label):
    print(f"\n--- external — {title} ---")
    case_ids = sorted(patient_risks.keys())
    ens = np.array([float(np.mean(patient_risks[c])) for c in case_ids])
    times = np.array([label[c][0] for c in case_ids])
    events = np.array([label[c][1] for c in case_ids])
    m = compute_survival_metrics(ens, times, events)
    print(f"  -> 15실행 앙상블: N={len(case_ids)} c_index={m['c_index']:.4f}")


if __name__ == "__main__":
    main()
