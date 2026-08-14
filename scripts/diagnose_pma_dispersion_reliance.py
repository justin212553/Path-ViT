"""
--attn-dispersion(공간특징, models/spatial_features.py) 사후 ablation — baseline PMA
(PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD) 3seed x 5fold 체크포인트 15개를 재학습 없이
그대로 두고, 추론 시점에 spatial_feat(dispersion 값)만 0벡터로 지워서 risk_head에 넣었을 때
c-index가 얼마나 변하는지 본다. scripts/diagnose_pma_component_reliance.py와 동일한 방식
(이미 학습 끝난 체크포인트에서 한 feature만 사후에 지우는 것)을 spatial_feat 하나에 적용한
버전 — "이 feature가 아예 없었으면 어떻게 됐을까"(재학습 ablation)보다 약한 증거지만,
재학습 없이 몇 분 안에 "지금 모델이 이 feature에 얼마나 기대고 있는가"를 알 수 있다.

internal은 각 seed 안에서 5-fold pooled out-of-fold(코호트 전체 152명 커버)로 계산한 뒤
3seed 예측 평균 앙상블, external은 15개 실행(seed x fold) 예측 평균 앙상블 — 기존
scripts/pool_multiseed_{kfold,external}_preds.py와 동일한 pooling 방식.

2026-08-14: internal fold의 test 환자 목록은 WSISurvivalDataset(split="test", fold=..)로
"재구성"하지 않고, 그 checkpoint를 만든 학습 실행이 실제로 저장해 둔
.logs/kfold_preds/*.csv의 case_id를 그대로 읽어 restrict_case_ids로 필터링한다 —
_stratified_kfold_assignment의 라운드로빈 편향 수정(오프셋 추가, data/dataset.py 2026-08-14)이
이 baseline 체크포인트(2026-08-08 학습, 수정 이전 코드)의 원래 fold 배정을 더 이상 재현하지
못해(직접 검증: fold0 seed42 31명 중 5명만 원본과 일치), split="test"로 재구성하면 원래
train/val에 있던 환자가 "test"로 잘못 섞여 들어가는 leakage가 생겼던 것을 CSV 직접 참조로
우회한다.

사용법: python -m scripts.diagnose_pma_dispersion_reliance
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
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
MODEL_TAG = "PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD"


def _identity_collate(batch):
    return batch[0]


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
def _collect_patient_data(model, loader, device):
    patients = []
    for patient_slides in loader:
        p = patient_slides[0]
        rna = p["rna"].to(device)
        z_rna = model.encode_rna(rna)
        comps, spatial_feats = [], []
        for slide in patient_slides:
            out = model(slide["coords"].to(device), features=slide["features"])
            comps.append(out["embed"])
            if "spatial_feat" in out:
                spatial_feats.append(out["spatial_feat"])
        patient_embed = torch.stack(comps).mean(dim=0)
        spatial_feat = torch.stack(spatial_feats).mean(dim=0) if spatial_feats else None

        age_years = p["age_years"].to(device)
        sex_idx = p["sex_idx"].to(device)
        margin_ord = p["margin_ord"].to(device)
        stage_ord = {f: p[f].to(device) for f in STAGE_FIELDS}
        clin_raw = model._clinical_raw(age_years, sex_idx, margin_ord, stage_ord=stage_ord)

        patients.append({
            "case_id": p["case_id"], "time": float(p["OS_time"].item()), "event": int(p["OS_event"].item()),
            "patient_embed": patient_embed, "z_rna": z_rna, "spatial_feat": spatial_feat, "clin_raw": clin_raw,
        })
    return patients


@torch.no_grad()
def _risks(model, patients, ablate_dispersion: bool):
    risks_by_case = {}
    for pd in patients:
        embed = pd["patient_embed"]
        z_wsi, _ = model.component_coattn(embed, pd["z_rna"])
        fused = torch.cat([z_wsi, pd["z_rna"]], dim=-1)
        if pd["spatial_feat"] is not None:
            spatial_feat = torch.zeros_like(pd["spatial_feat"]) if ablate_dispersion else pd["spatial_feat"]
            fused = torch.cat([fused, spatial_feat], dim=-1)
        risk = model.risk_head(fused.unsqueeze(0)).view(1)
        risk = risk + model.clinical_linear(pd["clin_raw"]).view(1)
        risks_by_case[pd["case_id"]] = (risk.item(), pd["time"], pd["event"])
    return risks_by_case


def _load_fold_case_ids(seed: int, fold: int) -> set[str]:
    path = (_ROOT / ".logs" / "kfold_preds" /
            f"tcga_{MODEL_TAG}_FOLD{fold}OF{N_FOLDS}_seed{seed}_fold{fold}of{N_FOLDS}.csv")
    with open(path, newline="") as f:
        return {row["case_id"] for row in csv.DictReader(f)}


def _pool_and_score(risk_dicts: list[dict]) -> dict:
    """여러 실행(run)의 {case_id: (risk, time, event)}를 case_id 기준으로 risk 평균 앙상블."""
    common = sorted(set.intersection(*(set(d.keys()) for d in risk_dicts)))
    risks = np.array([np.mean([d[c][0] for d in risk_dicts]) for c in common])
    times = np.array([risk_dicts[0][c][1] for c in common])
    events = np.array([risk_dicts[0][c][2] for c in common])
    return compute_survival_metrics(risks, times, events)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rna_gene_ids = literature_guided_gene_ids_intersection(1500)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS["tcga"])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS["tcga"])
    stage_stats = stage_stats_from_csv(CLINICAL_PATHS["tcga"])
    ds_kwargs = dict(with_clinical=True, with_margin=True, with_staging=True, with_rna=True,
                      rna_gene_ids=rna_gene_ids, feature_backbone="uni2")

    external_ds = None  # dataset="cptac"는 seed에 무관(split="all") — 첫 로드 후 재사용
    internal_baseline_by_seed, internal_ablated_by_seed = {}, {}
    external_baseline_runs, external_ablated_runs = [], []

    for seed in SEEDS:
        cfg = Config()
        cfg.data.seed = seed
        fold_baseline_risks, fold_ablated_risks = {}, {}

        for fold in range(N_FOLDS):
            ckpt_glob = list((_ROOT / "models" / "checkpoint").glob(
                f"survival_tcga_uni2_seed{seed}_*{MODEL_TAG}_FOLD{fold}OF{N_FOLDS}_best_pma.pt"))
            if len(ckpt_glob) != 1:
                raise RuntimeError(f"seed={seed} fold={fold}: checkpoint {len(ckpt_glob)}개 매칭 (1개여야 함)")
            ckpt_path = ckpt_glob[0]
            print(f"[seed={seed} fold={fold}] {ckpt_path.name}")

            model = _build_model(device, len(rna_gene_ids), age_mean, age_std, margin_stats, stage_stats, ckpt_path)

            fold_case_ids = _load_fold_case_ids(seed, fold)
            internal_ds = WSISurvivalDataset(cfg.data, dataset="tcga", split="all",
                                              restrict_case_ids=fold_case_ids, **ds_kwargs)
            assert len(internal_ds.cases) == len(fold_case_ids), (
                f"seed={seed} fold={fold}: 필터링된 환자 수({len(internal_ds.cases)})가 "
                f"CSV의 case_id 수({len(fold_case_ids)})와 다릅니다")
            internal_loader = DataLoader(internal_ds, batch_size=1, shuffle=False,
                                          collate_fn=_identity_collate, num_workers=0)
            internal_patients = _collect_patient_data(model, internal_loader, device)
            fold_baseline_risks.update(_risks(model, internal_patients, ablate_dispersion=False))
            fold_ablated_risks.update(_risks(model, internal_patients, ablate_dispersion=True))

            if external_ds is None:
                external_ds = WSISurvivalDataset(cfg.data, dataset="cptac", split="all", **ds_kwargs)
            external_loader = DataLoader(external_ds, batch_size=1, shuffle=False,
                                          collate_fn=_identity_collate, num_workers=0)
            external_patients = _collect_patient_data(model, external_loader, device)
            external_baseline_runs.append(_risks(model, external_patients, ablate_dispersion=False))
            external_ablated_runs.append(_risks(model, external_patients, ablate_dispersion=True))

        internal_baseline_by_seed[seed] = fold_baseline_risks
        internal_ablated_by_seed[seed] = fold_ablated_risks

    print("\n=== Internal (seed별 pooled OOF, 3seed 예측 평균 앙상블) ===")
    m_base = _pool_and_score(list(internal_baseline_by_seed.values()))
    m_abl = _pool_and_score(list(internal_ablated_by_seed.values()))
    print(f"  baseline(dispersion 유지): c_index={m_base['c_index']:.4f} log_rank_p={m_base['log_rank_p']:.4f}")
    print(f"  dispersion=0 ablation   : c_index={m_abl['c_index']:.4f} log_rank_p={m_abl['log_rank_p']:.4f}")
    print(f"  delta = {m_abl['c_index'] - m_base['c_index']:+.4f}")

    print("\n=== External (15실행 예측 평균 앙상블) ===")
    m_base_ext = _pool_and_score(external_baseline_runs)
    m_abl_ext = _pool_and_score(external_ablated_runs)
    print(f"  baseline(dispersion 유지): c_index={m_base_ext['c_index']:.4f} log_rank_p={m_base_ext['log_rank_p']:.4f}")
    print(f"  dispersion=0 ablation   : c_index={m_abl_ext['c_index']:.4f} log_rank_p={m_abl_ext['log_rank_p']:.4f}")
    print(f"  delta = {m_abl_ext['c_index'] - m_base_ext['c_index']:+.4f}")


if __name__ == "__main__":
    main()
