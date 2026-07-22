"""
같은 아키텍처(PMA_EX_SS_AUX)를 여러 시드로 학습한 체크포인트들을 모아, external test(cptac)에서
환자별 risk score를 평균 낸 앙상블 성능을 확인한다 — 이 프로젝트 전체에서 시드 간 편차가 매우
컸던 것에 대한 대응(재학습 없이 기존 체크포인트만으로 시도 가능).

사용법:
    python -m scripts.ensemble_eval --dataset tcga --seeds 42 84 126
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
from train import _patient_risk
from utils.metrics import compute_survival_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="tcga", choices=["tcga", "cptac"],
                         help="학습에 쓴 코호트(external 방향의 train 쪽) — external은 반대 코호트.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 84, 126])
    parser.add_argument("--ckpt-tag", type=str, default="tcga_EX_SS_AUX",
                         help="models/checkpoint/survival_{ckpt-tag}_best_pma_seed{seed}.pt 패턴의 태그.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    external_dataset = {"tcga": "cptac", "cptac": "tcga"}[args.dataset]

    cfg = Config()
    rna_gene_ids = literature_guided_gene_ids(1500)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.dataset])

    external_ds = WSISurvivalDataset(
        cfg.data, dataset=external_dataset, split="all",
        with_clinical=True, with_rna=True, rna_gene_ids=rna_gene_ids,
    )
    print(f"external({external_dataset}) 환자 수: {len(external_ds)}")

    ckpt_dir = Path(__file__).resolve().parent.parent / "models" / "checkpoint"
    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad()

    case_ids = None
    times = None
    events = None
    per_seed_risks = []

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

        risks, seed_case_ids, seed_times, seed_events = [], [], [], []
        with torch.no_grad():
            for i in range(len(external_ds)):
                patient_slides = external_ds[i]
                risk, _, _ = _patient_risk(model, patient_slides, device, amp_ctx, transform=None, chunk_size=64)
                risks.append(risk.float().item())
                seed_case_ids.append(patient_slides[0]["case_id"])
                seed_times.append(float(patient_slides[0]["OS_time"].item()))
                seed_events.append(int(patient_slides[0]["OS_event"].item()))

        if case_ids is None:
            case_ids, times, events = seed_case_ids, seed_times, seed_events
        else:
            assert seed_case_ids == case_ids, "시드 간 환자 순서가 다릅니다 — external_ds가 seed에 의존하면 안 됨"

        risks = np.array(risks)
        per_seed_risks.append(risks)
        metrics = compute_survival_metrics(risks, np.array(times), np.array(events))
        print(f"seed={seed:4d} | external_c_index={metrics['c_index']:.4f} | "
              f"HR={metrics['hr']:.3f} | logrank_p={metrics['log_rank_p']:.4f}")

    times_arr, events_arr = np.array(times), np.array(events)

    print("\n=== 앙상블(risk score 단순 평균) ===")
    ensemble_risk = np.mean(per_seed_risks, axis=0)
    metrics = compute_survival_metrics(ensemble_risk, times_arr, events_arr)
    print(f"external_c_index={metrics['c_index']:.4f} | HR={metrics['hr']:.3f} "
          f"[{metrics['hr_ci_lower']:.3f}, {metrics['hr_ci_upper']:.3f}] | logrank_p={metrics['log_rank_p']:.4f}")

    print("\n=== 참고: 단일 시드 평균(앙상블 아님, 그냥 3개 숫자의 평균) ===")
    single_seed_cs = [compute_survival_metrics(r, times_arr, events_arr)["c_index"] for r in per_seed_risks]
    print(f"평균 C={np.mean(single_seed_cs):.4f} (개별: {[f'{c:.4f}' for c in single_seed_cs]})")


if __name__ == "__main__":
    main()
