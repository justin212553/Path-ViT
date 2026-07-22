"""
학습된 PMA_EX_SS_AUX 체크포인트(seed42/84/126, RNA 수정판, models/checkpoint/
survival_tcga_EX_SS_AUX_best_pma_seed{seed}.pt)를 재학습 없이 그대로 써서, "WSI 브랜치를
실제로 얼마나 쓰고 있는가" + "그 attention이 internal(tcga held-out)과 external(cptac)에서
어떻게 다르게 행동하는가"를 직접 조사한다.

배경: findings_backlog.md — RNA 수정/risk head/tile-fusion/인코더 비율/앙상블까지 다 시도해도
external에서 WSI+RNA+Clinical fusion 모델이 RNA+Clinical만 쓰는 M7_EX을 못 넘는 패턴이
계속 재현됐다. "어떤 아키텍처를 더 시도할까"가 아니라 "지금 학습된 모델이 WSI를 어떻게
쓰고 있길래 이런 결과가 나오는가"를 직접 들여다보는 진단 스크립트.

네 가지를 계산한다:
  (A) 브랜치별 ablation — risk_head 입력 [z_wsi, z_clinical, z_rna] 중 한 브랜치씩만 (a) 0벡터로
      치환 (b) 환자 간 무작위로 셔플(patient-specific 신호만 제거, 분포는 그대로) 했을 때 C-index가
      baseline 대비 internal/external에서 각각 얼마나/어느 방향으로 변하는지. 나머지 두 브랜치는
      건드리지 않고 risk_head만 다시 통과시키므로 재학습이 전혀 필요 없다 — WSI뿐 아니라
      Clinical/RNA도 같은 방식으로 비교해, 세 브랜치의 상대적 기여도를 나란히 잰다.
  (B) RNA-guided co-attention이 4개 통계적 관점(mean/std/attention-weighted/top-k-mean) 중
      무엇을 고르는지 — 거의 uniform(0.25씩)이면 사실상 관점을 구분 못 하고 있다는 뜻.
  (C) 패치 단위 ABMIL attention(MultiComponentPooling 내부)의 정규화 엔트로피 — 1에 가까우면
      거의 모든 패치를 동등하게 취급(수렴 못 함), 0에 가까우면 소수 패치에 뾰족하게 집중.

internal(tcga split="test", 학습에 안 쓰인 held-out)과 external(cptac 전체)을 나란히 비교해,
"external에서만 유독 WSI 의존이 해로운가"를 직접 확인하는 게 핵심 질문이다.

사용법:
    python -m scripts.diagnose_wsi_reliance --seeds 42 84 126
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
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, literature_guided_gene_ids
from models import ViT_PMA
from models.clinical_encoder import age_stats_from_csv
from models.rna_predictor import RNAPredictionHead
from utils.metrics import compute_survival_metrics

COMPONENT_NAMES = ["mean", "std", "attn", "topk"]
N_PERM_TRIALS = 20


def _c_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """compute_survival_metrics()와 같은 공식이지만 HR/log-rank(lifelines, 느림)는 건너뛴
    C-index 전용 버전 — permutation ablation처럼 반복 호출하는 곳에 쓴다."""
    comparable = (time[:, None] < time[None, :]) & event[:, None].astype(bool)
    concordant = comparable & (risk[:, None] > risk[None, :])
    tied = comparable & (risk[:, None] == risk[None, :])
    n = int(comparable.sum())
    return float((concordant.sum() + 0.5 * tied.sum()) / n) if n > 0 else float("nan")


def _entropy(p: torch.Tensor) -> float:
    """p: (K,) 합=1인 분포. 정규화 엔트로피(0~1, 1=완전균등)를 반환한다."""
    k = p.shape[0]
    if k <= 1:
        return 1.0
    ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum()
    return float((ent / np.log(k)).item())


@torch.no_grad()
def _patient_forward(model, patient_slides, device):
    """_patient_risk(train.py)의 PMA 경로를 재구현하되, risk_head 직전 중간값
    (z_wsi/z_clinical/z_rna/co-attention 가중치/패치 attention 엔트로피)을 그대로 노출한다."""
    rna = patient_slides[0]["rna"].to(device, non_blocking=True)
    z_rna = model.encode_rna(rna)  # (rna_dim,)

    components_per_slide = []
    patch_entropies = []
    for slide in patient_slides:
        coords = slide["coords"].to(device, non_blocking=True)
        out = model(coords, features=slide["features"])
        components_per_slide.append(out["embed"])  # (4, D)
        patch_entropies.append(_entropy(out["attn_weights"]))

    patient_components = torch.stack(components_per_slide).mean(dim=0)  # (4, D), 슬라이드 평균

    age_years = patient_slides[0]["age_years"].to(device, non_blocking=True)
    sex_idx = patient_slides[0]["sex_idx"].to(device, non_blocking=True)
    z_clinical = model.clinical_encoder(age_years.unsqueeze(0), sex_idx.unsqueeze(0)).squeeze(0)

    z_wsi, coattn_weights = model.component_coattn(patient_components, z_rna)  # (D,), (4,)

    return {
        "z_wsi": z_wsi.float().cpu(),
        "z_clinical": z_clinical.float().cpu(),
        "z_rna": z_rna.float().cpu(),
        "coattn_weights": coattn_weights.float().cpu(),
        "patch_attn_entropy": float(np.mean(patch_entropies)),
        "case_id": patient_slides[0]["case_id"],
        "time": float(patient_slides[0]["OS_time"].item()),
        "event": int(patient_slides[0]["OS_event"].item()),
    }


def _risk_from_parts(model, z_wsi: torch.Tensor, z_clinical: torch.Tensor, z_rna: torch.Tensor) -> torch.Tensor:
    """vit_pma.py::ViT_PMA.combine_with_clinical_rna와 동일한 concat 순서([z_wsi, z_clinical,
    z_rna])로 risk_head만 재실행한다 — WSI 인코딩을 다시 태우지 않아 ablation이 매우 저렴하다."""
    combined = torch.cat([z_wsi, z_clinical, z_rna], dim=-1).unsqueeze(0)
    return model.risk_head(combined).view(1)


def _collect(model, dataset, device) -> list[dict]:
    records = []
    for i in range(len(dataset)):
        records.append(_patient_forward(model, dataset[i], device))
    return records


BRANCHES = ["wsi", "clinical", "rna"]


@torch.no_grad()
def _ablation_report(model, records: list[dict], device, rng: np.random.Generator) -> dict:
    parts = {
        "wsi": torch.stack([r["z_wsi"] for r in records]).to(device),
        "clinical": torch.stack([r["z_clinical"] for r in records]).to(device),
        "rna": torch.stack([r["z_rna"] for r in records]).to(device),
    }
    times = np.array([r["time"] for r in records])
    events = np.array([r["event"] for r in records])
    n = len(records)

    def _batch_risk(p: dict[str, torch.Tensor]) -> np.ndarray:
        combined = torch.cat([p["wsi"], p["clinical"], p["rna"]], dim=-1)
        return model.risk_head(combined).view(-1).cpu().numpy()

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

    coattn = torch.stack([r["coattn_weights"] for r in records]).mean(dim=0)  # (4,)
    coattn_entropy = np.mean([_entropy(r["coattn_weights"]) for r in records])
    patch_entropy = np.mean([r["patch_attn_entropy"] for r in records])

    return {
        "n": n,
        "baseline_c": baseline_metrics["c_index"],
        "baseline_hr": baseline_metrics["hr"],
        "baseline_p": baseline_metrics["log_rank_p"],
        "branches": branch_reports,
        "coattn_weights": coattn.numpy(),
        "coattn_entropy": float(coattn_entropy),
        "patch_attn_entropy": float(patch_entropy),
    }


def _print_report(label: str, rep: dict):
    print(f"\n--- {label} (n={rep['n']}) ---")
    print(f"  baseline           : C={rep['baseline_c']:.4f}  HR={rep['baseline_hr']:.3f}  logrank_p={rep['baseline_p']:.4f}")
    for branch in BRANCHES:
        br = rep["branches"][branch]
        print(f"  [{branch:8s}] zero-ablation : C={br['zero_c']:.4f}  (baseline 대비 {br['zero_c']-rep['baseline_c']:+.4f})")
        print(f"  [{branch:8s}] perm-ablation : C={br['perm_c_mean']:.4f} +/- {br['perm_c_std']:.4f}  "
              f"(baseline 대비 {br['perm_c_mean']-rep['baseline_c']:+.4f})")
    coattn_str = ", ".join(f"{n}={w:.3f}" for n, w in zip(COMPONENT_NAMES, rep["coattn_weights"]))
    print(f"  co-attention 4-관점 평균 가중치: {coattn_str}  (균등=0.25씩, 엔트로피={rep['coattn_entropy']:.3f}/1.0)")
    print(f"  패치 attention 정규화 엔트로피 : {rep['patch_attn_entropy']:.3f}/1.0  (1=완전균등, 0=소수 패치에 집중)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 84, 126])
    parser.add_argument("--ckpt-tag", type=str, default="tcga_EX_SS_AUX")
    parser.add_argument("--perm-seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.perm_seed)

    cfg = Config()
    rna_gene_ids = literature_guided_gene_ids(1500)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS["tcga"])

    print("데이터셋 준비 중...")
    internal_ds = WSISurvivalDataset(
        cfg.data, dataset="tcga", split="test",
        with_clinical=True, with_rna=True, rna_gene_ids=rna_gene_ids,
    )
    external_ds = WSISurvivalDataset(
        cfg.data, dataset="cptac", split="all",
        with_clinical=True, with_rna=True, rna_gene_ids=rna_gene_ids,
    )
    print(f"internal(tcga held-out test) 환자 수: {len(internal_ds)}")
    print(f"external(cptac 전체) 환자 수: {len(external_ds)}")

    ckpt_dir = Path(__file__).resolve().parent.parent / "models" / "checkpoint"

    seed_reports = {"internal": [], "external": []}
    for seed in args.seeds:
        ckpt_path = ckpt_dir / f"survival_{args.ckpt_tag}_best_pma_seed{seed}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"체크포인트 없음: {ckpt_path}")

        model = ViT_PMA(cfg.model, age_mean=age_mean, age_std=age_std,
                         rna_input_dim=len(rna_gene_ids)).to(device)
        model.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, len(rna_gene_ids)).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        print(f"\n{'='*70}\nseed={seed}\n{'='*70}")

        internal_records = _collect(model, internal_ds, device)
        external_records = _collect(model, external_ds, device)

        internal_rep = _ablation_report(model, internal_records, device, rng)
        external_rep = _ablation_report(model, external_records, device, rng)
        seed_reports["internal"].append(internal_rep)
        seed_reports["external"].append(external_rep)

        _print_report(f"seed={seed} internal(tcga held-out)", internal_rep)
        _print_report(f"seed={seed} external(cptac)", external_rep)

    print(f"\n{'='*70}\n{len(args.seeds)}시드 평균\n{'='*70}")
    for label, reps in seed_reports.items():
        avg = {
            "n": reps[0]["n"],
            "baseline_c": np.mean([r["baseline_c"] for r in reps]),
            "baseline_hr": np.mean([r["baseline_hr"] for r in reps]),
            "baseline_p": np.mean([r["baseline_p"] for r in reps]),
            "branches": {
                branch: {
                    "zero_c": np.mean([r["branches"][branch]["zero_c"] for r in reps]),
                    "perm_c_mean": np.mean([r["branches"][branch]["perm_c_mean"] for r in reps]),
                    "perm_c_std": np.mean([r["branches"][branch]["perm_c_std"] for r in reps]),
                }
                for branch in BRANCHES
            },
            "coattn_weights": np.mean([r["coattn_weights"] for r in reps], axis=0),
            "coattn_entropy": np.mean([r["coattn_entropy"] for r in reps]),
            "patch_attn_entropy": np.mean([r["patch_attn_entropy"] for r in reps]),
        }
        _print_report(f"{label} (3시드 평균)", avg)


if __name__ == "__main__":
    main()
