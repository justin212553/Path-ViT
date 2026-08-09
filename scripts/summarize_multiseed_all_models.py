"""
M1_POOL/M2_POOL/M3/M5/M6/M7(+참고용 PMA)을 한 번에 훑어서, 모델마다 internal(seed간 pooled +
prediction 앙상블)과 external(seed x fold prediction 앙상블)을 계산하고, 결과를 표+로그
파일로 남긴다. scripts/pool_multiseed_kfold_preds.py + pool_multiseed_external_preds.py
(단일 모델 전용, 그대로 남겨둠)의 로직을 여러 모델에 한 번에 적용하는 버전이다.

2026-08-08: M1_POOL/M2_POOL/M3(UNI2)는 HPC에서, M5/M6/M7은 로컬에서 학습하기로 해서, 두 군데의
데이터가 준비되는 시점이 다르다 — 이 스크립트는 CSV가 없는 모델은 크래시 대신 [SKIP]으로
표시하고 넘어가므로, 어느 쪽에서 실행해도 그 시점에 있는 데이터만으로 부분 요약이 가능하다.

사용법:
    python -m scripts.summarize_multiseed_all_models
    python -m scripts.summarize_multiseed_all_models --out .logs/my_summary.log
    python -m scripts.summarize_multiseed_all_models --seeds 42,84,126 --n-folds 5
"""
import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.metrics import compute_survival_metrics

DEFAULT_SEEDS = [42, 84, 126]
DEFAULT_N_FOLDS = 5
DEFAULT_BOOTSTRAP = 2000

# (라벨, internal dataset, external dataset, model_prefix) — model_prefix는 train.py/
# train_light.py가 실제로 만드는 model_prefix와 정확히 같아야 한다(그래야 kfold_preds/
# external_preds 파일명이 맞아떨어짐).
MODELS = [
    ("M1_POOL (WSI 단독, self-attn, UNI2)",           "tcga", "cptac", "M1_POOL_uni2_SS_DISP"),
    ("M2_POOL (WSI+Clinical co-attn, UNI2)",          "tcga", "cptac", "M2_POOL_uni2_SS_DISP"),
    ("M3 (WSI+RNA, UNI2)",                            "tcga", "cptac", "PMA_uni2_INT1500_SS_AUX_NOCLINICAL_DISP"),
    ("M5 (Clinical, STG+R)",                          "tcga", "cptac", "M5_STG_R"),
    ("M6 (RNA, INT1500)",                             "tcga", "cptac", "M6_INT1500"),
    ("M7 (Clinical+RNA, STG+R, cox_add)",             "tcga", "cptac", "M7_INT1500_STG_R_COX_ADD"),
    ("PMA (WSI+RNA+Clinical, STG+R, UNI2, cox_add)",  "tcga", "cptac", "PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD"),
]


def _load_csv(path: Path) -> dict:
    preds = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            preds[row["case_id"]] = (float(row["risk"]), float(row["OS_time"]), int(row["OS_event"]))
    return preds


def _bootstrap_ci(risks, times, events, n_boot=DEFAULT_BOOTSTRAP, seed=0):
    rng = np.random.RandomState(seed)
    n = len(risks)
    boot_c = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        m = compute_survival_metrics(risks[idx], times[idx], events[idx])
        if not np.isnan(m["c_index"]):
            boot_c.append(m["c_index"])
    boot_c = np.array(boot_c)
    lo, hi = np.percentile(boot_c, [2.5, 97.5])
    return lo, hi, float(boot_c.std())


def summarize_internal(dataset: str, model: str, seeds: list[int], n_folds: int, n_boot: int):
    pred_dir = _ROOT / ".logs" / "kfold_preds"
    per_seed_preds, per_seed_c, lines = {}, [], []

    for seed in seeds:
        preds = {}
        for fold in range(n_folds):
            path = pred_dir / f"{dataset}_{model}_FOLD{fold}OF{n_folds}_seed{seed}_fold{fold}of{n_folds}.csv"
            if not path.exists():
                path = pred_dir / f"{dataset}_{model}_seed{seed}_fold{fold}of{n_folds}.csv"
            if not path.exists():
                return None, [f"  [SKIP] seed={seed} fold={fold} internal CSV 없음: {path.name}"]
            fold_preds = _load_csv(path)
            overlap = set(fold_preds) & set(preds)
            if overlap:
                return None, [f"  [ERROR] seed={seed} 내 fold 간 case_id 중복 {len(overlap)}명 — fold 분할이 겹쳤을 가능성"]
            preds.update(fold_preds)
        per_seed_preds[seed] = preds
        risks  = np.array([v[0] for v in preds.values()])
        times  = np.array([v[1] for v in preds.values()])
        events = np.array([v[2] for v in preds.values()])
        m = compute_survival_metrics(risks, times, events)
        per_seed_c.append(m["c_index"])
        lines.append(f"  seed={seed}: N={len(preds)}, events={int(events.sum())} | c_index={m['c_index']:.4f} | "
                      f"HR={m['hr']:.3f} [{m['hr_ci_lower']:.3f},{m['hr_ci_upper']:.3f}] | log_rank_p={m['log_rank_p']:.4f}")

    per_seed_c = np.array(per_seed_c)
    lines.append(f"  -> seed 간 pooled c-index: mean={per_seed_c.mean():.4f}, std={per_seed_c.std():.4f}")

    case_sets = [set(p.keys()) for p in per_seed_preds.values()]
    common = sorted(set.intersection(*case_sets))
    if any(len(cs) != len(common) for cs in case_sets):
        lines.append(f"  [경고] seed 간 case 집합이 다름 — 교집합({len(common)}명)만 앙상블에 사용")

    ens_risks, times, events = [], [], []
    for cid in common:
        seed_risks, ref = [], None
        for seed in seeds:
            r, t, e = per_seed_preds[seed][cid]
            seed_risks.append(r)
            ref = ref or (t, e)
        ens_risks.append(float(np.mean(seed_risks)))
        times.append(ref[0]); events.append(ref[1])
    ens_risks, times, events = np.array(ens_risks), np.array(times), np.array(events)
    m = compute_survival_metrics(ens_risks, times, events)
    lo, hi, std = _bootstrap_ci(ens_risks, times, events, n_boot)
    lines.append(f"  === {len(seeds)}-seed 예측 평균 앙상블 (N={len(common)}, events={int(events.sum())}) ===")
    lines.append(f"  c_index={m['c_index']:.4f} | HR={m['hr']:.3f} [{m['hr_ci_lower']:.3f},{m['hr_ci_upper']:.3f}] | "
                  f"log_rank_p={m['log_rank_p']:.4f} | bootstrap95%CI=[{lo:.4f},{hi:.4f}] std={std:.4f}")
    return {"ensemble_c": m["c_index"], "per_seed_mean": float(per_seed_c.mean()),
            "per_seed_std": float(per_seed_c.std()), "ci": (lo, hi)}, lines


def summarize_external(external_dataset: str, model: str, seeds: list[int], n_folds: int, n_boot: int):
    pred_dir = _ROOT / ".logs" / "external_preds"
    run_c = []
    patient_risks: dict[str, list[float]] = defaultdict(list)
    patient_label: dict[str, tuple[float, int]] = {}
    lines = []

    for seed in seeds:
        for fold in range(n_folds):
            path = pred_dir / f"{external_dataset}_{model}_FOLD{fold}OF{n_folds}_seed{seed}_fold{fold}of{n_folds}.csv"
            if not path.exists():
                path = pred_dir / f"{external_dataset}_{model}_seed{seed}_fold{fold}of{n_folds}.csv"
            if not path.exists():
                return None, [f"  [SKIP] seed={seed} fold={fold} external CSV 없음: {path.name}"]
            preds = _load_csv(path)
            risks  = np.array([v[0] for v in preds.values()])
            times  = np.array([v[1] for v in preds.values()])
            events = np.array([v[2] for v in preds.values()])
            m = compute_survival_metrics(risks, times, events)
            run_c.append(m["c_index"])
            for cid, (r, t, e) in preds.items():
                patient_risks[cid].append(r)
                patient_label[cid] = (t, e)

    run_c = np.array(run_c)
    lines.append(f"  {len(run_c)}개 실행(seed x fold) external c-index: mean={run_c.mean():.4f}, std={run_c.std():.4f}")

    case_ids = sorted(patient_risks.keys())
    ens_risks = np.array([float(np.mean(patient_risks[c])) for c in case_ids])
    times  = np.array([patient_label[c][0] for c in case_ids])
    events = np.array([patient_label[c][1] for c in case_ids])
    m = compute_survival_metrics(ens_risks, times, events)
    lo, hi, std = _bootstrap_ci(ens_risks, times, events, n_boot)
    lines.append(f"  === {len(seeds)*n_folds}개 예측 평균 앙상블 (N={len(case_ids)}, events={int(events.sum())}) ===")
    lines.append(f"  c_index={m['c_index']:.4f} | HR={m['hr']:.3f} [{m['hr_ci_lower']:.3f},{m['hr_ci_upper']:.3f}] | "
                  f"log_rank_p={m['log_rank_p']:.4f} | bootstrap95%CI=[{lo:.4f},{hi:.4f}] std={std:.4f}")
    return {"ensemble_c": m["c_index"], "run_mean": float(run_c.mean()), "run_std": float(run_c.std()),
            "ci": (lo, hi)}, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=None,
                         help="로그 파일 경로(기본: .logs/multiseed_summary_<타임스탬프>.log)")
    parser.add_argument("--seeds", type=str, default="42,84,126")
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out_path = Path(args.out) if args.out else _ROOT / ".logs" / f"multiseed_summary_{datetime.now():%Y%m%d_%H%M%S}.log"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_lines = [f"=== Multi-seed({len(seeds)}seed x {args.n_folds}fold) internal/external 요약 "
                 f"— {datetime.now():%Y-%m-%d %H:%M} ===\n"]
    summary_rows = []

    for label, dataset, ext_dataset, model in MODELS:
        all_lines.append(f"\n{'='*70}\n{label}  (model_prefix={model})\n{'='*70}")
        all_lines.append("[Internal]")
        int_result, int_lines = summarize_internal(dataset, model, seeds, args.n_folds, args.bootstrap)
        all_lines.extend(int_lines)
        all_lines.append("[External]")
        ext_result, ext_lines = summarize_external(ext_dataset, model, seeds, args.n_folds, args.bootstrap)
        all_lines.extend(ext_lines)
        summary_rows.append((label, int_result, ext_result))

    all_lines.append(f"\n\n{'='*70}\n요약표\n{'='*70}")
    header = f"{'모델':<48} {'Internal 앙상블':>18} {'External 앙상블':>18}"
    all_lines.append(header)
    for label, int_result, ext_result in summary_rows:
        int_s = f"{int_result['ensemble_c']:.4f}" if int_result else "N/A(SKIP)"
        ext_s = f"{ext_result['ensemble_c']:.4f}" if ext_result else "N/A(SKIP)"
        all_lines.append(f"{label:<48} {int_s:>18} {ext_s:>18}")

    text = "\n".join(all_lines)
    print(text)
    out_path.write_text(text, encoding="utf-8")
    print(f"\n\n로그 저장: {out_path}")


if __name__ == "__main__":
    main()
