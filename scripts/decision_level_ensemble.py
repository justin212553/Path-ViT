"""
Decision-level 앙상블 — M6(RNA 단독)/M5(Clinical 단독)/M1_POOL(WSI 단독)처럼 완전히 독립적으로
학습된 단일 모달리티 모델들의 예측을 사후에(risk score 평균) 합친다. PMA/M7 같은 joint fusion
(하나의 risk_head가 여러 모달리티를 같이 최적화)은 강한 신호(RNA)가 약한 신호(clinical/WSI)와
gradient를 공유하며 희석되는 문제가 이 세션에서 반복 확인됐다(PMA/M7이 M6 단독보다도 낮음) —
decision-level 앙상블은 각 모델을 독립적으로 학습한 뒤 예측값만 사후에 섞으므로, 강한 모델의
신호가 학습 단계에서 애초에 희석될 일이 없다.

이미 저장된 M5/M6/M1_POOL의 3seed x 5fold(internal)·15실행(external) 예측 CSV를 그대로
재사용한다 — 새로 학습할 필요가 없다.

[스케일 문제] 모델마다 risk_head 구조/loss 스케일이 달라 raw risk score의 분산이 다르다.
raw 평균을 내면 분산이 큰 모델이 앙상블을 사실상 지배해버릴 수 있어, 모델별로 z-score
표준화(평균 0, 표준편차 1로 맞춤)한 뒤 평균한다.

사용법: python scripts/decision_level_ensemble.py
"""
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.metrics import compute_survival_metrics

SEEDS = [42, 84, 126]
N_FOLDS = 5

MODELS = {
    "M6(RNA)": "M6_INT1500",
    "M5(Clinical)": "M5_STG_R",
    "M1_POOL(WSI)": "M1_POOL_uni2_SS_DISP",
}


def _load_internal_ensemble(model_prefix: str) -> dict:
    """3seed x 5fold pooled OOF를 시드 간 평균 — pool_multiseed_kfold_preds.py와 동일 로직."""
    pred_dir = _ROOT / ".logs" / "kfold_preds"
    per_seed = {}
    for seed in SEEDS:
        preds = {}
        for fold in range(N_FOLDS):
            path = pred_dir / f"tcga_{model_prefix}_FOLD{fold}OF{N_FOLDS}_seed{seed}_fold{fold}of{N_FOLDS}.csv"
            if not path.exists():
                path = pred_dir / f"tcga_{model_prefix}_seed{seed}_fold{fold}of{N_FOLDS}.csv"
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    preds[row["case_id"]] = (float(row["risk"]), float(row["OS_time"]), int(row["OS_event"]))
        per_seed[seed] = preds
    common = sorted(set.intersection(*(set(p.keys()) for p in per_seed.values())))
    out = {}
    for cid in common:
        risks = [per_seed[s][cid][0] for s in SEEDS]
        _, t, e = per_seed[SEEDS[0]][cid]
        out[cid] = (float(np.mean(risks)), t, e)
    return out


def _load_external_ensemble(model_prefix: str) -> dict:
    """15실행(3seed x 5fold) 예측을 환자 단위로 평균 — pool_multiseed_external_preds.py와 동일 로직."""
    pred_dir = _ROOT / ".logs" / "external_preds"
    patient_risks, patient_label = {}, {}
    for seed in SEEDS:
        for fold in range(N_FOLDS):
            path = pred_dir / f"cptac_{model_prefix}_FOLD{fold}OF{N_FOLDS}_seed{seed}_fold{fold}of{N_FOLDS}.csv"
            if not path.exists():
                path = pred_dir / f"cptac_{model_prefix}_seed{seed}_fold{fold}of{N_FOLDS}.csv"
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    cid = row["case_id"]
                    patient_risks.setdefault(cid, []).append(float(row["risk"]))
                    patient_label[cid] = (float(row["OS_time"]), int(row["OS_event"]))
    return {cid: (float(np.mean(rs)), *patient_label[cid]) for cid, rs in patient_risks.items()}


def _zscore(risks: np.ndarray) -> np.ndarray:
    return (risks - risks.mean()) / risks.std()


def _ensemble_c_index(model_scores: dict, combo: tuple, bootstrap: int = 0):
    """combo에 속한 모델들의 z표준화 risk를 평균해 c-index 계산."""
    common = sorted(set.intersection(*(set(model_scores[m].keys()) for m in combo)))
    times = np.array([model_scores[combo[0]][cid][1] for cid in common])
    events = np.array([model_scores[combo[0]][cid][2] for cid in common])

    z_stack = []
    for m in combo:
        raw = np.array([model_scores[m][cid][0] for cid in common])
        z_stack.append(_zscore(raw))
    ensembled = np.mean(z_stack, axis=0)

    m = compute_survival_metrics(ensembled, times, events)
    result = {"c_index": m["c_index"], "hr": m["hr"], "log_rank_p": m["log_rank_p"], "n": len(common)}
    if bootstrap > 0:
        rng = np.random.RandomState(0)
        boot_c = []
        for _ in range(bootstrap):
            idx = rng.randint(0, len(common), len(common))
            bm = compute_survival_metrics(ensembled[idx], times[idx], events[idx])
            if not np.isnan(bm["c_index"]):
                boot_c.append(bm["c_index"])
        lo, hi = np.percentile(boot_c, [2.5, 97.5])
        result["ci"] = (lo, hi)
    return result


def main():
    print("=== 모델별 예측 로드 ===")
    internal_scores, external_scores = {}, {}
    for label, prefix in MODELS.items():
        internal_scores[label] = _load_internal_ensemble(prefix)
        external_scores[label] = _load_external_ensemble(prefix)
        print(f"  {label}: internal N={len(internal_scores[label])}, external N={len(external_scores[label])}")

    names = list(MODELS.keys())
    combos = []
    for r in range(1, len(names) + 1):
        combos.extend(itertools.combinations(names, r))

    print("\n=== internal (3seed 앙상블 기준 조합) ===")
    for combo in combos:
        res = _ensemble_c_index(internal_scores, combo, bootstrap=2000 if len(combo) > 1 else 0)
        ci_str = f" CI=[{res['ci'][0]:.4f},{res['ci'][1]:.4f}]" if "ci" in res else ""
        print(f"  {' + '.join(combo):40s} c_index={res['c_index']:.4f} HR-logp={res['log_rank_p']:.4f}{ci_str}")

    print("\n=== external (15실행 앙상블 기준 조합) ===")
    for combo in combos:
        res = _ensemble_c_index(external_scores, combo, bootstrap=2000 if len(combo) > 1 else 0)
        ci_str = f" CI=[{res['ci'][0]:.4f},{res['ci'][1]:.4f}]" if "ci" in res else ""
        print(f"  {' + '.join(combo):40s} c_index={res['c_index']:.4f} HR-logp={res['log_rank_p']:.4f}{ci_str}")


if __name__ == "__main__":
    main()
