"""
scripts/diagnose_porpoise_reliance.py — ViT_PORPOISE(--PORPOISE) 체크포인트를 재학습 없이 그대로
써서 "이 개선이 어디서 왔는가"를 직접 조사한다. scripts/diagnose_wsi_reliance.py(2026-07-21,
co-attention entropy 붕괴를 처음 발견한 바로 그 스크립트)를 ViT_PORPOISE 계산 그래프에 맞게
재구성한 것.

[배경] PORPOISE 파일럿(seed84/fold0, internal C=0.7063)이 co-attention 계열(M4/M4A/MCAT,
findings_backlog.md 2026-08-31 최상위 발견 — co-attention entropy 0.999~1.000 붕괴, query
개수·gradient와 무관)보다 확연히 좋게 나왔다. ViT_PORPOISE는 기존 대비 **두 가지**를 동시에
바꿨다 — (a) attn_pool을 RNA-guided co-attention에서 RNA 무관 plain gated-ABMIL로 교체,
(b) risk_head 직전 결합을 concat에서 BilinearFusion(Kronecker product)으로 교체. 이 스크립트
하나로는 (a)/(b) 중 뭐가 원인인지 분리 못 하지만(그러려면 (a)만 바꾼 버전, (b)만 바꾼 버전을
따로 학습해야 함 — 별도 ablation), 최소한 **(b)의 전제 조건**은 확인할 수 있다 — plain ABMIL이
RNA 간섭 없이 patch를 실제로 구별하기 시작했는지(그럼 attention 자체가 이미 좋아진 것) 아니면
여전히 uniform인지(그럼 attention과 무관하게 BilinearFusion만으로 이득이 났다는 뜻)를 직접 보면
된다.

두 가지를 계산한다:
  (A) 브랜치별 ablation — WSI/Clinical/RNA 세 모달리티를 각각 (a) 0벡터로 치환 (b) 환자 간
      무작위로 셔플했을 때 C-index가 baseline 대비 어떻게 변하는지. ViT_PORPOISE는
      diagnose_wsi_reliance.py(ViT_PMA, [z_wsi,z_clinical,z_rna] concat)와 계산 그래프가
      달라 ablation 방식도 다르다 — WSI/RNA는 BilinearFusion(z_wsi, z_rna) 이후 risk_head를,
      Clinical은 별도의 Cox 가산항(_clinical_embed → clinical_linear)을 각각 재실행한다.
  (B) plain gated-ABMIL의 patch attention 정규화 엔트로피 — 1에 가까우면 co-attention 때와
      마찬가지로 여전히 거의 균등(RNA를 안 봐도 patch를 못 고른다는 뜻), 0에 가까우면 소수
      patch에 뾰족하게 집중(RNA 간섭이 없어지자 이제 patch를 구별하기 시작했다는 뜻).

사용법:
    python -m scripts.diagnose_porpoise_reliance --ckpt models/checkpoint/survival_..._best_porpoise.pt
    # 여러 체크포인트(예: dispersion/aux ablation 3종) 한 번에 비교 — --labels로 표에 붙일 이름 지정
    python -m scripts.diagnose_porpoise_reliance \
        --ckpt ckpt_full.pt ckpt_no_disp.pt ckpt_no_aux.pt ckpt_no_disp_no_aux.pt \
        --labels full no_dispersion no_aux no_disp_no_aux \
        --attn-dispersion 1 0 1 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
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

N_PERM_TRIALS = 20
BRANCHES = ["wsi", "clinical", "rna"]


def _c_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    comparable = (time[:, None] < time[None, :]) & event[:, None].astype(bool)
    concordant = comparable & (risk[:, None] > risk[None, :])
    tied = comparable & (risk[:, None] == risk[None, :])
    n = int(comparable.sum())
    return float((concordant.sum() + 0.5 * tied.sum()) / n) if n > 0 else float("nan")


def _entropy(p: torch.Tensor) -> float:
    """p: (N,) 합=1인 attention 분포. 정규화 엔트로피(0~1, 1=완전균등)."""
    n = p.shape[0]
    if n <= 1:
        return 1.0
    ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum()
    return float((ent / np.log(n)).item())


@torch.no_grad()
def _patient_forward(model, patient_slides, device):
    """ViT_PORPOISE 계산 그래프(train.py::_patient_risk와 동일 순서, 단 결합만 다름) —
    risk_head/BilinearFusion/clinical_linear 직전 중간값을 그대로 노출한다."""
    rna = patient_slides[0]["rna"].to(device, non_blocking=True)
    z_rna = model.encode_rna(rna)  # (D,)

    slide_embeds = []
    patch_entropies = []
    spatial_feats = []
    for slide in patient_slides:
        coords = slide["coords"].to(device, non_blocking=True)
        features = slide.get("features")
        out = model(coords, features=features.to(device, non_blocking=True) if features is not None else None)
        slide_embeds.append(out["embed"])
        patch_entropies.append(_entropy(out["attn_weights"]))
        if "spatial_feat" in out:
            spatial_feats.append(out["spatial_feat"])
    z_wsi = torch.stack(slide_embeds).mean(dim=0)  # (D,) — 환자 단위 평균 풀링
    # ViT_PORPOISE.combine_with_clinical_rna와 동일 — use_attn_dispersion=True인 체크포인트는
    # fusion 출력에 이 1차원을 이어붙인 뒤 risk_head를 통과한다(risk_head 입력 차원이 mmhid+1).
    spatial_feat = torch.stack(spatial_feats).mean(dim=0) if spatial_feats else None

    age_years = patient_slides[0]["age_years"].to(device, non_blocking=True)
    sex_idx = patient_slides[0]["sex_idx"].to(device, non_blocking=True)
    stage_ord = _stage_ord_from_patient(patient_slides, device)
    margin_ord = _margin_ord_from_patient(patient_slides, device)
    clin_raw = model._clinical_embed(age_years, sex_idx, margin_ord, stage_ord=stage_ord)  # (1, raw_dim)

    return {
        "z_wsi": z_wsi.float().cpu(),
        "z_rna": z_rna.float().cpu(),
        "clin_raw": clin_raw.squeeze(0).float().cpu(),  # (raw_dim,)
        "spatial_feat": spatial_feat.float().cpu() if spatial_feat is not None else None,  # (1,) or None
        "patch_attn_entropy": float(np.mean(patch_entropies)),
        "case_id": patient_slides[0]["case_id"],
        "time": float(patient_slides[0]["OS_time"].item()),
        "event": int(patient_slides[0]["OS_event"].item()),
    }


def _collect(model, dataset, device) -> list[dict]:
    return [_patient_forward(model, dataset[i], device) for i in range(len(dataset))]


@torch.no_grad()
def _ablation_report(model, records: list[dict], device, rng: np.random.Generator) -> dict:
    parts = {
        "wsi": torch.stack([r["z_wsi"] for r in records]).to(device),
        "rna": torch.stack([r["z_rna"] for r in records]).to(device),
        "clinical": torch.stack([r["clin_raw"] for r in records]).to(device),
    }
    spatial_feats = (
        torch.stack([r["spatial_feat"] for r in records]).to(device)
        if records[0]["spatial_feat"] is not None else None
    )
    times = np.array([r["time"] for r in records])
    events = np.array([r["event"] for r in records])
    n = len(records)

    def _batch_risk(p: dict[str, torch.Tensor]) -> np.ndarray:
        # WSI/RNA는 BilinearFusion(Kronecker product) 이후 risk_head, Clinical은 별도 Cox
        # 가산항(clinical_linear) — 둘 다 배치 축이 없는 모듈이라 환자별로 루프 돈다(N이
        # 작아 비용 무시 가능).
        risks = []
        for i in range(n):
            fused = model.fusion(p["wsi"][i], p["rna"][i])  # (mmhid,)
            if spatial_feats is not None:
                fused = torch.cat([fused, spatial_feats[i]], dim=-1)
            risk = model.risk_head(fused.unsqueeze(0)).view(1)
            risk = risk + model.clinical_linear(p["clinical"][i].unsqueeze(0)).view(1)
            risks.append(risk.item())
        return np.array(risks)

    baseline_risk = _batch_risk(parts)
    baseline_metrics = compute_survival_metrics(baseline_risk, times, events)

    branch_reports = {}
    for branch in BRANCHES:
        zero_parts = dict(parts)
        zero_parts[branch] = torch.zeros_like(parts[branch])
        zero_metrics = compute_survival_metrics(_batch_risk(zero_parts), times, events)

        perm_cs = []
        for _ in range(N_PERM_TRIALS):
            perm = rng.permutation(n)
            perm_parts = dict(parts)
            perm_parts[branch] = parts[branch][perm]
            perm_cs.append(_c_index(_batch_risk(perm_parts), times, events))

        branch_reports[branch] = {
            "zero_c": zero_metrics["c_index"],
            "perm_c_mean": float(np.mean(perm_cs)),
            "perm_c_std": float(np.std(perm_cs)),
        }

    patch_entropy = np.mean([r["patch_attn_entropy"] for r in records])
    patch_entropy_std = np.std([r["patch_attn_entropy"] for r in records])

    return {
        "n": n,
        "baseline_c": baseline_metrics["c_index"],
        "baseline_hr": baseline_metrics["hr"],
        "baseline_p": baseline_metrics["log_rank_p"],
        "branches": branch_reports,
        "patch_attn_entropy": float(patch_entropy),
        "patch_attn_entropy_std": float(patch_entropy_std),
    }


def _print_report(label: str, rep: dict):
    print(f"\n--- {label} (n={rep['n']}) ---")
    print(f"  baseline           : C={rep['baseline_c']:.4f}  HR={rep['baseline_hr']:.3f}  logrank_p={rep['baseline_p']:.4f}")
    for branch in BRANCHES:
        br = rep["branches"][branch]
        print(f"  [{branch:8s}] zero-ablation : C={br['zero_c']:.4f}  (baseline 대비 {br['zero_c']-rep['baseline_c']:+.4f})")
        print(f"  [{branch:8s}] perm-ablation : C={br['perm_c_mean']:.4f} +/- {br['perm_c_std']:.4f}  "
              f"(baseline 대비 {br['perm_c_mean']-rep['baseline_c']:+.4f})")
    print(f"  plain-ABMIL patch attention 정규화 엔트로피: {rep['patch_attn_entropy']:.4f} "
          f"+/- {rep['patch_attn_entropy_std']:.4f}  (1=완전균등[co-attention 때와 동일 붕괴], "
          f"0=소수 patch에 집중[RNA 간섭 없어지자 구별 시작])")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, nargs="+", required=True, help="ViT_PORPOISE 체크포인트 경로(들).")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                         help="--ckpt 각각에 붙일 이름(기본: ckpt 파일명). 개수가 --ckpt와 같아야 함.")
    parser.add_argument("--attn-dispersion", type=int, nargs="+", default=None,
                         help="--ckpt 각각이 --attn-dispersion으로 학습됐는지(1/0). 기본: 전부 0. "
                              "risk_head 입력 차원이 달라지므로 반드시 학습 때와 맞춰야 로드된다.")
    parser.add_argument("--dataset", type=str, default="tcga", choices=["tcga", "cptac"])
    parser.add_argument("--external-dataset", type=str, default=None, choices=[None, "tcga", "cptac"],
                         help="지정하면 internal(--dataset의 test fold)뿐 아니라 external(이 코호트 "
                              "전체, split='all')도 같이 진단한다.")
    parser.add_argument("--rna-n-genes", type=int, default=1500)
    parser.add_argument("--backbone", type=str, default="uni2")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--perm-seed", type=int, default=0)
    args = parser.parse_args()

    labels = args.labels or [Path(c).stem for c in args.ckpt]
    if len(labels) != len(args.ckpt):
        raise ValueError("--labels 개수는 --ckpt 개수와 같아야 합니다.")
    dispersions = args.attn_dispersion or [0] * len(args.ckpt)
    if len(dispersions) != len(args.ckpt):
        raise ValueError("--attn-dispersion 개수는 --ckpt 개수와 같아야 합니다.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.perm_seed)

    cfg = Config()
    rna_gene_ids = literature_guided_gene_ids_intersection(args.rna_n_genes)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.dataset])
    stage_stats = stage_stats_from_df(__import__("pandas").read_csv(CLINICAL_PATHS[args.dataset]))
    margin_stats = margin_stats_from_df(
        __import__("pandas").read_csv(CLINICAL_PATHS[args.dataset])[["residual_disease"]]
    )

    ds_kwargs = dict(
        with_clinical=True, with_staging=True, with_margin=True, with_rna=True,
        feature_backbone=args.backbone, rna_gene_ids=rna_gene_ids,
        fold=args.fold, n_folds=args.n_folds,
    )
    print("데이터셋 준비 중...")
    internal_ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="test", **ds_kwargs)
    print(f"internal({args.dataset} fold{args.fold}/{args.n_folds} test) 환자 수: {len(internal_ds)}")
    external_ds = None
    if args.external_dataset:
        external_ds = WSISurvivalDataset(cfg.data, dataset=args.external_dataset, split="all", **ds_kwargs)
        print(f"external({args.external_dataset} 전체) 환자 수: {len(external_ds)}")

    for ckpt_path, label, use_disp in zip(args.ckpt, labels, dispersions):
        print(f"\n{'='*70}\n{label}  ({ckpt_path})\n{'='*70}")
        model = ViT_PORPOISE(
            cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=len(rna_gene_ids),
            backbone=args.backbone, use_staging=True, stage_stats=stage_stats,
            use_margin=True, margin_stats=margin_stats, use_attn_dispersion=bool(use_disp),
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # strict=False: --rna-aux-weight로 학습된 체크포인트는 rna_aux_head 가중치도 갖고
        # 있는데, 이 진단은 attn_pool/fusion/risk_head만 보므로 그 헤드는 재구성하지 않는다.
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        assert not missing, f"필수 파라미터 누락: {missing}"
        model.eval()

        internal_records = _collect(model, internal_ds, device)
        internal_rep = _ablation_report(model, internal_records, device, rng)
        _print_report(f"{label} — internal({args.dataset} test)", internal_rep)

        if external_ds is not None:
            external_records = _collect(model, external_ds, device)
            external_rep = _ablation_report(model, external_records, device, rng)
            _print_report(f"{label} — external({args.external_dataset})", external_rep)


if __name__ == "__main__":
    main()
