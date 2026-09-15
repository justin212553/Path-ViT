"""
MMP(Song et al., ICML 2024, mahmoodlab/MMP) BRCA 재현(sbatch/run_mmp_brca_survival_array_hpc.sh,
2seed x 5fold = 10런) 결과를 이 프로젝트 관례(pool_porpoise_official_kfold.py와 동일한 정신)로
집계한다.

MMP 자체 main_survival.py는 fold마다 {results_dir}/{split}_results.pkl을 저장한다
(training/trainer.py::validate_survival, dumps 딕셔너리: sample_ids, all_risk_scores,
all_censorships, all_event_times). PORPOISE의 patient_results(dict)와 달리 병렬 배열
형태라, sample_ids와 zip해서 {case_id: (risk, time, event)}로 바꾼다.

**주의(극성, 미검증)**: os_censorship을 저희 자체 CSV에서 1-OS_event로 채워 넣었으니(scripts/
prepare_mmp_brca_data.py), MMP 파이프라인이 그 값을 그대로 통과시킨다면 이 스크립트도
PORPOISE와 동일하게 event=1-censorship로 변환하면 맞다 — 다만 MMP 자체 risk score의 부호가
"높을수록 나쁜 예후"인지는 실제 결과가 나오기 전엔 코드만으로 100% 확신하기 어렵다. 처음
돌린 뒤 이 스크립트가 내는 c-index가 0.5 근처에서 한참 벗어나 0.2~0.3대로 나오면 부호가
뒤집힌 것이니 risk에 -1을 곱해서 다시 확인할 것.

internal은 5-fold pooled out-of-fold(각 seed 안에서 fold별 held-out test를 이어붙임) +
seed 간 예측 평균 앙상블, external은 이 연구 다른 모델들과 동일한 관례대로 10개(seed x fold)
체크포인트의 A2/AR/E9 예측을 전부 평균한다(각 fold가 서로 다른 train split으로 학습됐으므로
fold 자체도 독립 추정치로 취급).

사용법:
    python scripts/pool_mmp_brca_kfold.py --results-root mmp/src/results \\
        --exp-prefix BRCA_OWNPROTOCOL_officialRNA --feat extracted-vit_large_patch16_224.dinov2.uni_mass100k \\
        --seeds 84,126 --n-folds 5 --bootstrap 2000
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.metrics import compute_survival_metrics


def _load_dump(results_root: Path, exp_prefix: str, feat: str, seed: int, fold: int, split: str) -> dict:
    run_dir = results_root / f"{exp_prefix}_seed{seed}_k{fold}::PANTHER_default::{feat}"
    pkl_path = run_dir / f"{split}_results.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"{pkl_path} 없음 — 이 (seed={seed}, fold={fold}) 런이 아직 안 끝났을 수 있음")
    with open(pkl_path, "rb") as f:
        dump = pickle.load(f)
    ids = dump["sample_ids"]
    risks = np.asarray(dump["all_risk_scores"]).reshape(-1)
    censorships = np.asarray(dump["all_censorships"]).reshape(-1)
    times = np.asarray(dump["all_event_times"]).reshape(-1)
    events = 1.0 - censorships  # PORPOISE 관례와 동일, 위 주의사항 참조
    return {cid: (float(r), float(t), float(e)) for cid, r, t, e in zip(ids, risks, times, events)}


def _pool_internal(results_root: Path, exp_prefix: str, feat: str, seeds: list[int], n_folds: int) -> dict:
    """seed별로 5-fold의 test(held-out) 예측을 이어붙여 pooled out-of-fold를 만든다."""
    per_seed_preds = {}
    for seed in seeds:
        preds = {}
        for fold in range(n_folds):
            fold_preds = _load_dump(results_root, exp_prefix, feat, seed, fold, "test")
            overlap = set(fold_preds) & set(preds)
            if overlap:
                raise ValueError(f"seed={seed}: fold {fold}가 이전 fold와 case_id {len(overlap)}개 겹침")
            preds.update(fold_preds)
        per_seed_preds[seed] = preds
    return per_seed_preds


def _ensemble_across_seeds(per_seed_preds: dict) -> tuple:
    case_sets = [set(p.keys()) for p in per_seed_preds.values()]
    common = set.intersection(*case_sets)
    if any(cs != common for cs in case_sets):
        missing = set.union(*case_sets) - common
        print(f"  [경고] 일부 case가 모든 seed에 있지 않음({len(missing)}명) — 교집합({len(common)}명)만 사용")
    common = sorted(common)
    risks, times, events = [], [], []
    for cid in common:
        seed_risks, ref_t, ref_e = [], None, None
        for seed, preds in per_seed_preds.items():
            r, t, e = preds[cid]
            seed_risks.append(r)
            if ref_t is None:
                ref_t, ref_e = t, e
            elif abs(t - ref_t) > 1e-6 or e != ref_e:
                raise ValueError(f"case_id={cid} 라벨이 seed마다 다름")
        risks.append(np.mean(seed_risks))
        times.append(ref_t)
        events.append(ref_e)
    return np.array(risks), np.array(times), np.array(events), common


def _bootstrap_ci(risks, times, events, n_boot, seed=0):
    rng = np.random.RandomState(seed)
    n = len(risks)
    boot_c = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        m = compute_survival_metrics(risks[idx], times[idx], events[idx])
        if not np.isnan(m["c_index"]):
            boot_c.append(m["c_index"])
    boot_c = np.array(boot_c)
    return np.percentile(boot_c, [2.5, 97.5]), boot_c.std()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=str, required=True, help="예: mmp/src/results")
    ap.add_argument("--exp-prefix", type=str, default="BRCA_OWNPROTOCOL_officialRNA")
    ap.add_argument("--feat", type=str, default="extracted-vit_large_patch16_224.dinov2.uni_mass100k")
    ap.add_argument("--seeds", type=str, default="84,126")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=0)
    args = ap.parse_args()

    results_root = Path(args.results_root)
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"=== MMP BRCA internal (pooled out-of-fold, seed별) — {results_root} ===")
    per_seed_preds = _pool_internal(results_root, args.exp_prefix, args.feat, seeds, args.n_folds)
    for seed, preds in per_seed_preds.items():
        risks = np.array([v[0] for v in preds.values()])
        times = np.array([v[1] for v in preds.values()])
        events = np.array([v[2] for v in preds.values()])
        m = compute_survival_metrics(risks, times, events)
        print(f"  seed={seed}: N={len(preds)}, events={int(events.sum())} | c_index={m['c_index']:.4f} | "
              f"HR={m['hr']:.3f} [{m['hr_ci_lower']:.3f}, {m['hr_ci_upper']:.3f}] | log_rank_p={m['log_rank_p']:.4f}")

    risks, times, events, cases = _ensemble_across_seeds(per_seed_preds)
    m = compute_survival_metrics(risks, times, events)
    print(f"\n=== Internal, seed 간 예측 평균 앙상블 (N={len(cases)}, events={int(events.sum())}) ===")
    print(f"  c_index={m['c_index']:.4f} | HR={m['hr']:.3f} [{m['hr_ci_lower']:.3f}, {m['hr_ci_upper']:.3f}] | "
          f"log_rank_p={m['log_rank_p']:.4f}")
    if args.bootstrap > 0:
        (lo, hi), std = _bootstrap_ci(risks, times, events, args.bootstrap)
        print(f"  bootstrap 95% CI (n={args.bootstrap}) = [{lo:.4f}, {hi:.4f}], std={std:.4f}")

    print(f"\n=== MMP BRCA external (A2/AR/E9, {len(seeds)}seed x {args.n_folds}fold = "
          f"{len(seeds) * args.n_folds}개 체크포인트 평균) ===")
    per_run_preds = []
    for seed in seeds:
        for fold in range(args.n_folds):
            per_run_preds.append(_load_dump(results_root, args.exp_prefix, args.feat, seed, fold, "external"))
    case_sets = [set(p.keys()) for p in per_run_preds]
    common = sorted(set.intersection(*case_sets))
    if any(cs != set(common) for cs in case_sets):
        print(f"  [경고] 일부 external case가 모든 런에 있지 않음 — 교집합({len(common)}명)만 사용")
    ext_risks, ext_times, ext_events = [], [], []
    for cid in common:
        run_risks, ref_t, ref_e = [], None, None
        for preds in per_run_preds:
            r, t, e = preds[cid]
            run_risks.append(r)
            if ref_t is None:
                ref_t, ref_e = t, e
        ext_risks.append(np.mean(run_risks))
        ext_times.append(ref_t)
        ext_events.append(ref_e)
    ext_risks, ext_times, ext_events = np.array(ext_risks), np.array(ext_times), np.array(ext_events)
    m = compute_survival_metrics(ext_risks, ext_times, ext_events)
    print(f"  N={len(common)}, events={int(ext_events.sum())} | c_index={m['c_index']:.4f} | "
          f"HR={m['hr']:.3f} [{m['hr_ci_lower']:.3f}, {m['hr_ci_upper']:.3f}] | log_rank_p={m['log_rank_p']:.4f}")
    if args.bootstrap > 0:
        (lo, hi), std = _bootstrap_ci(ext_risks, ext_times, ext_events, args.bootstrap)
        print(f"  bootstrap 95% CI (n={args.bootstrap}) = [{lo:.4f}, {hi:.4f}], std={std:.4f}")


if __name__ == "__main__":
    main()
