"""
scripts/diagnose_porpoise_residual_hazard.py — collapse된 attention 안에 남아있는 옅은
patient-specific 잔차(§2-3, findings_backlog.md 2026-08-31 / paper/porpoise_investigation_
2026-08-31.md — z_attn과 z_mean의 cosine 0.9998, 그러나 차이 벡터끼리는 pairwise cosine 0.26
으로 환자마다 다름)가 진짜로 survival hazard와 관련된 신호인지, 아니면 그냥 hazard와 무관한
환자별 신호(염색 배치 효과 등)인지를 재학습 없이 직접 검증한다.

[방법] fusion/risk_head/clinical_linear는 이미 z_attn(gated-ABMIL 실제 출력)을 입력으로
학습됐다 — 가중치를 전혀 건드리지 않고, 추론 시점에만 WSI 입력을 z_attn -> z_mean(같은 patch
토큰의 무파라미터 평균)으로 바꿔치기해서 risk를 다시 계산한다.
  - baseline: risk = risk_head(fusion(z_attn, z_rna) [+ spatial_feat]) + clinical_linear(clin)
  - swapped : risk = risk_head(fusion(z_mean, z_rna) [+ spatial_feat]) + clinical_linear(clin)
같은 파라미터로 z_attn 대신 z_mean을 넣었을 때 C-index/log-rank p가 유의하게 떨어지면 ->
risk_head가 그 잔차를 실제로 예측에 활용 중(잔차 = hazard 정보 보유)이라는 뜻. 거의 안
바뀌면 -> risk_head가 그 잔차를 (있어도) 사실상 안 쓴다는 뜻 — post-hoc sharpening이 왜
안 먹혔는지(§2-3)와도 정합적인 결론이 된다.

논문 관례(2seed 84/126 x 5fold, seed42는 WSI 포함 모델에서 제외)로 internal/external을
pooled 비교한다. internal은 seed 내에서는 out-of-fold concat(fold가 서로 disjoint), seed
간에는 환자 단위 risk 평균(ensemble) — scripts/pool_multiseed_kfold_preds.py와 동일 관례.
external은 10개 체크포인트(2seed x 5fold) 전부의 예측을 환자 단위로 평균.

사용법:
    python -m scripts.diagnose_porpoise_residual_hazard
    python -m scripts.diagnose_porpoise_residual_hazard --seeds 42,84,126  # seed42도 참고용으로
    # temp=0.2 sharpening 체크포인트로 검증하려면:
    python -m scripts.diagnose_porpoise_residual_hazard \
        --ckpt-pattern "models/checkpoint/survival_tcga_uni2_seed{seed}_INT1500_SS_STG_R_PORPOISE_uni2_INT1500_SS_STG_R_T0.2_DISP_FOLD{fold}OF{n_folds}_best_porpoise.pt" \
        --label temp0.2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, literature_guided_gene_ids_intersection
from models import ViT_PORPOISE
from models.clinical_encoder import age_stats_from_csv, stage_stats_from_df, margin_stats_from_df
from utils.metrics import compute_survival_metrics
from train import _stage_ord_from_patient, _margin_ord_from_patient

DEFAULT_CKPT_PATTERN = (
    "models/checkpoint/survival_tcga_uni2_seed{seed}_INT1500_SS_STG_R_PORPOISE_uni2_"
    "INT1500_SS_STG_R_DISP_FOLD{fold}OF{n_folds}_best_porpoise.pt"
)


@torch.no_grad()
def _patient_risks(model, patient_slides, device) -> dict:
    """environment: ViT_PORPOISE.combine_with_clinical_rna와 동일 계산 그래프를 두 번(z_attn/
    z_mean) 태워 baseline/swapped risk를 함께 낸다."""
    rna = patient_slides[0]["rna"].to(device, non_blocking=True)
    z_rna = model.encode_rna(rna)  # (D,)

    attn_embeds, mean_embeds, spatial_feats = [], [], []
    for slide in patient_slides:
        coords = slide["coords"].to(device, non_blocking=True)
        features = slide.get("features")
        out = model(coords, features=features.to(device, non_blocking=True) if features is not None else None)
        attn_embeds.append(out["embed"])
        mean_embeds.append(out["meanpool_embed"])
        if "spatial_feat" in out:
            spatial_feats.append(out["spatial_feat"])
    z_attn = torch.stack(attn_embeds).mean(dim=0)
    z_mean = torch.stack(mean_embeds).mean(dim=0)
    spatial_feat = torch.stack(spatial_feats).mean(dim=0) if spatial_feats else None

    age_years = patient_slides[0]["age_years"].to(device, non_blocking=True)
    sex_idx = patient_slides[0]["sex_idx"].to(device, non_blocking=True)
    stage_ord = _stage_ord_from_patient(patient_slides, device)
    margin_ord = _margin_ord_from_patient(patient_slides, device)
    clin_raw = model._clinical_embed(age_years, sex_idx, margin_ord, stage_ord=stage_ord).squeeze(0)

    def _risk(z_wsi: torch.Tensor) -> float:
        fused = model.fusion(z_wsi, z_rna)
        if spatial_feat is not None:
            fused = torch.cat([fused, spatial_feat], dim=-1)
        r = model.risk_head(fused.unsqueeze(0)).view(1)
        r = r + model.clinical_linear(clin_raw.unsqueeze(0)).view(1)
        return float(r.item())

    return {
        "case_id": patient_slides[0]["case_id"],
        "time": float(patient_slides[0]["OS_time"].item()),
        "event": int(patient_slides[0]["OS_event"].item()),
        "baseline_risk": _risk(z_attn),
        "swapped_risk": _risk(z_mean),
        "resid_norm": float((z_attn - z_mean).norm().item() / (z_attn.norm().item() + 1e-12)),
    }


def _load_model(ckpt_path, cfg, rna_gene_ids, age_mean, age_std, stage_stats, margin_stats,
                 backbone, device):
    model = ViT_PORPOISE(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=len(rna_gene_ids),
        backbone=backbone, use_staging=True, stage_stats=stage_stats,
        use_margin=True, margin_stats=margin_stats, use_attn_dispersion=True,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    assert not missing, f"필수 파라미터 누락: {missing}"
    model.eval()
    return model


def _report(label: str, risks: np.ndarray, times: np.ndarray, events: np.ndarray) -> dict:
    m = compute_survival_metrics(risks, times, events)
    print(f"  [{label:8s}] N={len(risks)} events={int(events.sum())} | c_index={m['c_index']:.4f} | "
          f"HR={m['hr']:.3f} [{m['hr_ci_lower']:.3f}, {m['hr_ci_upper']:.3f}] | "
          f"log_rank_p={m['log_rank_p']:.4f}")
    return m


def _residual_isolation_report(label: str, b_ens: np.ndarray, s_ens: np.ndarray,
                                times: np.ndarray, events: np.ndarray):
    """잔차(Δrisk = baseline - swapped)를 그 자체로 하나의 독립된 risk score로 취급해
    hazard와의 관련성을 직접 잰다 — '너무 작아서 안 보이는 것'과 'hazard와 orthogonal한 것'을
    분리하기 위한 진단. c_index는 방향(부호)에 민감하므로 max(C, 1-C)도 같이 본다(부호는
    baseline-swapped라는 정의상 임의적 — Δrisk가 hazard와 반대 방향으로 정렬돼 있어도
    '관련은 있다'는 신호이므로). log_rank_p(중앙값 이분화)는 부호와 무관하게 유의성만 본다."""
    resid = b_ens - s_ens
    resid_std_ratio = float(np.std(resid) / (np.std(b_ens) + 1e-12))
    m = compute_survival_metrics(resid, times, events)
    c_abs = max(m["c_index"], 1 - m["c_index"]) if not np.isnan(m["c_index"]) else float("nan")
    print(f"  [잔차 단독] Δrisk의 std / baseline risk의 std = {resid_std_ratio:.4f} "
          f"(1이면 같은 규모, 작을수록 '너무 작아서'에 가까움)")
    print(f"  [잔차 단독] Δrisk를 risk score로 쓸 때: c_index={m['c_index']:.4f} "
          f"(방향 무관 max(C,1-C)={c_abs:.4f}) | log_rank_p={m['log_rank_p']:.4f} "
          f"(부호와 무관하게 유의성만 봄 — 낮으면 orthogonal이 아니라는 뜻)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=str, default="84,126",
                         help="콤마 구분 seed 목록 (기본: 84,126 — paper 관례, WSI 모델은 seed42 제외)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="tcga")
    parser.add_argument("--external-dataset", type=str, default="cptac")
    parser.add_argument("--rna-n-genes", type=int, default=1500)
    parser.add_argument("--backbone", type=str, default="uni2")
    parser.add_argument("--ckpt-pattern", type=str, default=DEFAULT_CKPT_PATTERN,
                         help="{seed}/{fold}/{n_folds} placeholder를 담은 체크포인트 경로 패턴. "
                              "기본은 no_aux 표준 PORPOISE 레시피.")
    parser.add_argument("--label", type=str, default="no_aux_baseline")
    parser.add_argument("--bootstrap", type=int, default=2000,
                         help="baseline vs swapped paired bootstrap로 delta C-index 유의성 검정. 0이면 생략.")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config()
    rna_gene_ids = literature_guided_gene_ids_intersection(args.rna_n_genes)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.dataset])
    stage_stats = stage_stats_from_df(pd.read_csv(CLINICAL_PATHS[args.dataset]))
    margin_stats = margin_stats_from_df(pd.read_csv(CLINICAL_PATHS[args.dataset])[["residual_disease"]])
    ds_kwargs = dict(
        with_clinical=True, with_staging=True, with_margin=True, with_rna=True,
        feature_backbone=args.backbone, rna_gene_ids=rna_gene_ids,
    )

    print(f"=== {args.label} — internal({args.dataset}) out-of-fold, seeds={seeds} ===")
    per_seed_internal = {}  # seed -> {case_id: record}
    for seed in seeds:
        cfg.data.seed = seed
        seed_records = {}
        for fold in range(args.n_folds):
            ckpt_path = args.ckpt_pattern.format(seed=seed, fold=fold, n_folds=args.n_folds)
            if not Path(ckpt_path).exists():
                print(f"  [SKIP] seed={seed} fold={fold}: {ckpt_path} 없음")
                continue
            model = _load_model(ckpt_path, cfg, rna_gene_ids, age_mean, age_std, stage_stats,
                                 margin_stats, args.backbone, device)
            ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="test",
                                     fold=fold, n_folds=args.n_folds, **ds_kwargs)
            for i in range(len(ds)):
                rec = _patient_risks(model, ds[i], device)
                seed_records[rec["case_id"]] = rec
        per_seed_internal[seed] = seed_records
        risks_b = np.array([r["baseline_risk"] for r in seed_records.values()])
        risks_s = np.array([r["swapped_risk"] for r in seed_records.values()])
        times = np.array([r["time"] for r in seed_records.values()])
        events = np.array([r["event"] for r in seed_records.values()])
        print(f" seed={seed} (N={len(seed_records)}):")
        _report("baseline", risks_b, times, events)
        _report("swapped", risks_s, times, events)

    common = set.intersection(*[set(v.keys()) for v in per_seed_internal.values()])
    common = sorted(common)
    print(f"\n=== {args.label} — internal seed간 앙상블(N={len(common)}) ===")
    b_ens, s_ens, times, events = [], [], [], []
    for cid in common:
        b_ens.append(np.mean([per_seed_internal[s][cid]["baseline_risk"] for s in seeds]))
        s_ens.append(np.mean([per_seed_internal[s][cid]["swapped_risk"] for s in seeds]))
        ref = per_seed_internal[seeds[0]][cid]
        times.append(ref["time"])
        events.append(ref["event"])
    b_ens, s_ens = np.array(b_ens), np.array(s_ens)
    times, events = np.array(times), np.array(events)
    m_b = _report("baseline", b_ens, times, events)
    m_s = _report("swapped", s_ens, times, events)
    print(f"  -> baseline - swapped C-index delta: {m_b['c_index'] - m_s['c_index']:+.4f}")
    _residual_isolation_report(args.label, b_ens, s_ens, times, events)

    if args.bootstrap > 0:
        rng = np.random.RandomState(0)
        n = len(common)
        deltas = []
        for _ in range(args.bootstrap):
            idx = rng.randint(0, n, n)
            cb = compute_survival_metrics(b_ens[idx], times[idx], events[idx])["c_index"]
            cs = compute_survival_metrics(s_ens[idx], times[idx], events[idx])["c_index"]
            if not (np.isnan(cb) or np.isnan(cs)):
                deltas.append(cb - cs)
        deltas = np.array(deltas)
        p_two_sided = float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))
        print(f"  -> paired bootstrap({args.bootstrap}) delta 95% CI: "
              f"[{np.percentile(deltas, 2.5):.4f}, {np.percentile(deltas, 97.5):.4f}], p={p_two_sided:.4f}")

    if args.external_dataset:
        print(f"\n=== {args.label} — external({args.external_dataset}) 전체, {len(seeds)}seed x "
              f"{args.n_folds}fold 앙상블 ===")
        ext_records_per_ckpt = []
        for seed in seeds:
            for fold in range(args.n_folds):
                ckpt_path = args.ckpt_pattern.format(seed=seed, fold=fold, n_folds=args.n_folds)
                if not Path(ckpt_path).exists():
                    continue
                model = _load_model(ckpt_path, cfg, rna_gene_ids, age_mean, age_std, stage_stats,
                                     margin_stats, args.backbone, device)
                ext_ds = WSISurvivalDataset(cfg.data, dataset=args.external_dataset, split="all", **ds_kwargs)
                recs = {}
                for i in range(len(ext_ds)):
                    rec = _patient_risks(model, ext_ds[i], device)
                    recs[rec["case_id"]] = rec
                ext_records_per_ckpt.append(recs)

        common_ext = set.intersection(*[set(r.keys()) for r in ext_records_per_ckpt])
        common_ext = sorted(common_ext)
        b_ens, s_ens, times, events = [], [], [], []
        for cid in common_ext:
            b_ens.append(np.mean([r[cid]["baseline_risk"] for r in ext_records_per_ckpt]))
            s_ens.append(np.mean([r[cid]["swapped_risk"] for r in ext_records_per_ckpt]))
            ref = ext_records_per_ckpt[0][cid]
            times.append(ref["time"])
            events.append(ref["event"])
        b_ens, s_ens = np.array(b_ens), np.array(s_ens)
        times, events = np.array(times), np.array(events)
        m_b = _report("baseline", b_ens, times, events)
        m_s = _report("swapped", s_ens, times, events)
        print(f"  -> baseline - swapped C-index delta: {m_b['c_index'] - m_s['c_index']:+.4f}")
        _residual_isolation_report(args.label, b_ens, s_ens, times, events)

        if args.bootstrap > 0:
            rng = np.random.RandomState(0)
            n = len(common_ext)
            deltas = []
            for _ in range(args.bootstrap):
                idx = rng.randint(0, n, n)
                cb = compute_survival_metrics(b_ens[idx], times[idx], events[idx])["c_index"]
                cs = compute_survival_metrics(s_ens[idx], times[idx], events[idx])["c_index"]
                if not (np.isnan(cb) or np.isnan(cs)):
                    deltas.append(cb - cs)
            deltas = np.array(deltas)
            p_two_sided = float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))
            print(f"  -> paired bootstrap({args.bootstrap}) delta 95% CI: "
                  f"[{np.percentile(deltas, 2.5):.4f}, {np.percentile(deltas, 97.5):.4f}], p={p_two_sided:.4f}")


if __name__ == "__main__":
    main()
