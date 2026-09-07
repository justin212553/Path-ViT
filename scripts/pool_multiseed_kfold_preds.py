"""
여러 seed로 반복한 --fold(k-fold) internal 결과를 모아, seed별 pooled out-of-fold internal
지표와 함께, 환자 단위로 risk score를 seed 간 평균 낸(prediction ensembling) 최종 지표까지 계산한다.

pool_kfold_preds.py(단일 seed)의 다중 seed 확장판 — 같은 fold 번호라도 seed마다 fold 배정 자체가
달라지므로(_stratified_kfold_assignment가 seed로 셔플), 환자 한 명은 seed마다 다른(그 환자를 한
번도 학습에 쓰지 않은) 모델의 예측을 받는다. 그 예측들을 환자 단위로 평균 내면, k-fold checkpoint
5개를 그대로 평균하는 것(같은 seed 안에서는 한 환자가 4/5 fold의 train/val에 이미 들어가 있어
held-out 원칙이 깨짐)과 달리, 각 예측이 항상 "그 환자를 한 번도 학습에 안 쓴 모델"에서만 나온다는
원칙을 지키면서 seed 간 노이즈를 줄일 수 있다.

사용법:
    python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
        --model PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD --seeds 42,84,126 --n-folds 5
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.metrics import compute_survival_metrics


#  train.py가 "best checkpoint"(기본, 항상 저장)와 별개로 진단용 변형 예측을 같은 model_prefix
#  기반 파일명에 이 태그들만 끼워 추가로 저장하는 경우가 있다(예: --report-final-epoch-metrics의
#  "_FINALEPOCH_", SWA soup의 "_SOUP_", --full-train의 "_FULLTRAIN") — 와일드카드 매칭이 이걸
#  같이 주워서 "여러 파일이 걸림" 모호성 에러를 내거나(2026-09-06 실사용 중 발견) 잘못 섞여
#  풀링될 수 있다. --model 문자열 자체가 그 태그를 명시적으로 포함하지 않는 한 기본적으로 제외한다.
_SPECIAL_INFIXES = ("FINALEPOCH", "SOUP", "FULLTRAIN")


def _find_pred_path(pred_dir: Path, dataset: str, model: str, seed: int, fold: int, n_folds: int) -> Path:
    """model_prefix에 _FOLD{f}OF{n} 접미사가 붙는지 여부(train.py의 여러 조건부 접미사 순서를
    손으로 재현하다 보면 붙는 자리를 놓치기 쉽다)와 무관하게, 접두사/접미사만 고정하고 그
    사이는 와일드카드로 찾는다 — 정확한 model_prefix 문자열을 매번 손으로 완벽히 재현해야
    하는 취약함을 없앤다. 단 _SPECIAL_INFIXES(진단용 변형 태그)는 --model이 명시적으로
    요청하지 않는 한 매칭에서 제외한다."""
    suffix = f"_seed{seed}_fold{fold}of{n_folds}.csv"
    matches = sorted(pred_dir.glob(f"{dataset}_{model}*{suffix}"))
    matches = [
        p for p in matches
        if all(tag in model or tag not in p.name for tag in _SPECIAL_INFIXES)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"seed={seed} fold={fold}: '{dataset}_{model}*{suffix}' 패턴에 여러 파일이 걸림 "
            f"(model 태그가 다른 실행과 접두사를 공유해 모호함) — {[p.name for p in matches]}"
        )
    # 못 찾았을 때: 같은 dataset+seed+fold의 다른 태그 파일이 있는지 보여줘서 "아예 안 돎"과
    # "태그 문자열이 틀림"을 구분할 수 있게 한다.
    nearby = sorted(pred_dir.glob(f"{dataset}_*{suffix}"))
    hint = f" (같은 seed/fold의 다른 태그 후보: {[p.name for p in nearby]})" if nearby else " (같은 seed/fold 파일 자체가 없음 — 그 array task가 아직 안 끝났거나 실패했을 가능성)"
    raise FileNotFoundError(f"seed={seed} fold={fold} 예측 파일을 못 찾음: {pred_dir}/{dataset}_{model}*{suffix}{hint}")


def _load_seed_predictions(pred_dir: Path, dataset: str, model: str, seed: int, n_folds: int) -> dict:
    """seed 하나의 n_folds개 fold CSV를 모아 {case_id: (risk, OS_time, OS_event)}로 반환한다."""
    preds = {}
    for fold in range(n_folds):
        path = _find_pred_path(pred_dir, dataset, model, seed, fold, n_folds)
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                cid = row["case_id"]
                if cid in preds:
                    raise ValueError(f"seed={seed} 내에서 case_id 중복: {cid} — fold 분할이 겹쳤을 가능성")
                preds[cid] = (float(row["risk"]), float(row["OS_time"]), int(row["OS_event"]))
    return preds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True, choices=["tcga", "cptac", "both", "brca"])
    parser.add_argument("--model", type=str, required=True,
                         help="model_prefix (예: PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD)")
    parser.add_argument("--seeds", type=str, default="42,84,126",
                         help="콤마로 구분한 seed 목록 (기본: 42,84,126 — 이 프로젝트 표준 3시드)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--bootstrap", type=int, default=0,
        help="주어지면(예: 2000) 최종 앙상블 지표에 환자 단위 resample bootstrap 95%% CI를 추가로 "
             "계산한다. 0(기본)이면 생략.",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    pred_dir = Path(__file__).parent.parent / ".logs" / "kfold_preds"

    per_seed_preds = {}
    per_seed_c = []
    print(f"=== seed별 pooled out-of-fold internal — {args.dataset} {args.model} ({args.n_folds}-fold) ===")
    for seed in seeds:
        preds = _load_seed_predictions(pred_dir, args.dataset, args.model, seed, args.n_folds)
        per_seed_preds[seed] = preds
        case_ids = list(preds.keys())
        risks  = np.array([preds[c][0] for c in case_ids])
        times  = np.array([preds[c][1] for c in case_ids])
        events = np.array([preds[c][2] for c in case_ids])
        m = compute_survival_metrics(risks, times, events)
        per_seed_c.append(m["c_index"])
        print(f"  seed={seed}: N={len(case_ids)}, events={int(events.sum())} | "
              f"c_index={m['c_index']:.4f} | HR={m['hr']:.3f} [{m['hr_ci_lower']:.3f}, {m['hr_ci_upper']:.3f}] | "
              f"log_rank_p={m['log_rank_p']:.4f}")

    per_seed_c = np.array(per_seed_c)
    print(f"  -> seed 간 pooled c-index: mean={per_seed_c.mean():.4f}, std={per_seed_c.std():.4f} "
          f"(이게 반복측정으로 본 '이 정도 흔들린다'는 불확실성 폭)")

    # seed 간 case_id 집합이 동일한지(같은 코호트를 매번 전부 커버했는지) 확인
    case_sets = [set(p.keys()) for p in per_seed_preds.values()]
    common_cases = set.intersection(*case_sets)
    if any(cs != common_cases for cs in case_sets):
        missing = set.union(*case_sets) - common_cases
        print(f"  [경고] 일부 case가 모든 seed에 있지 않음({len(missing)}명) — "
              f"교집합({len(common_cases)}명)만 앙상블에 사용")
    common_cases = sorted(common_cases)

    # 환자 단위 risk score를 seed 간 평균 (각 seed의 예측은 항상 그 환자를 학습에 안 쓴 모델의 것 —
    # held-out 원칙을 유지한 채 평균)
    ensembled_risks, times, events = [], [], []
    for cid in common_cases:
        seed_risks = []
        ref_time, ref_event = None, None
        for seed in seeds:
            r, t, e = per_seed_preds[seed][cid]
            seed_risks.append(r)
            if ref_time is None:
                ref_time, ref_event = t, e
            elif abs(t - ref_time) > 1e-6 or e != ref_event:
                raise ValueError(f"case_id={cid}의 OS_time/OS_event가 seed마다 다름 — 라벨 불일치 의심")
        ensembled_risks.append(float(np.mean(seed_risks)))
        times.append(ref_time)
        events.append(ref_event)

    ensembled_risks = np.array(ensembled_risks)
    times = np.array(times)
    events = np.array(events)
    m = compute_survival_metrics(ensembled_risks, times, events)
    print(f"\n=== seed 간 예측 평균 앙상블 (N={len(common_cases)}, events={int(events.sum())}) ===")
    print(f"  c_index={m['c_index']:.4f} | HR={m['hr']:.3f} [{m['hr_ci_lower']:.3f}, {m['hr_ci_upper']:.3f}] | "
          f"log_rank_p={m['log_rank_p']:.4f}")

    if args.bootstrap > 0:
        rng = np.random.RandomState(0)
        n = len(common_cases)
        boot_c = []
        for _ in range(args.bootstrap):
            idx = rng.randint(0, n, n)
            bm = compute_survival_metrics(ensembled_risks[idx], times[idx], events[idx])
            if not np.isnan(bm["c_index"]):
                boot_c.append(bm["c_index"])
        boot_c = np.array(boot_c)
        lo, hi = np.percentile(boot_c, [2.5, 97.5])
        print(f"  bootstrap 95% CI (환자 단위 resample, n={args.bootstrap}) = "
              f"[{lo:.4f}, {hi:.4f}], std={boot_c.std():.4f}")


if __name__ == "__main__":
    main()
