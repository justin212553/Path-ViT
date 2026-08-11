"""
Internal(OOF) 예측에서 c-index를 흔드는 이상치/고레버리지 환자를 감사한다.

pool_multiseed_kfold_preds.py와 동일하게 3시드(42/84/126) OOF risk를 환자 단위로 평균 낸
앙상블 예측을 기준으로, 두 가지 지표를 계산한다:

1) leave-one-out delta: 환자 한 명을 통째로 빼고 c-index를 다시 계산했을 때 원래 c-index와의
   차이. delta = c_full - c_without_p. delta가 큰 음수면("이 환자를 빼면 c-index가 오히려
   오른다") 그 환자가 성능을 깎아먹는 고레버리지/이상치 후보다. 반대로 큰 양수면 그 환자
   혼자서 성능을 상당 부분 떠받치고 있다는 뜻이라 이것도 감사 대상이다(과대적합/우연 신호 의심).

2) discordant fraction: 그 환자가 관여하는 모든 comparable pair(Harrell's c-index 정의,
   utils/metrics.py::compute_survival_metrics와 동일한 comparability 규칙) 중 risk 순서가
   실제 생존 순서와 어긋난 쌍의 비율. involvement(관여 pair 수)가 충분히 큰데 discordant
   비율도 높은 환자가 진짜 이상치다 — involvement가 작으면 우연히 비율이 튈 수 있어 걸러야 함.

c-index 정의(comparable = time_i < time_j & event_i)상 "이른 시점에 죽은 환자(case, row)"와
"그 시점까지 살아있던 걸로 관찰된 환자(control, col)" 두 역할로 각각 몇 번 등장하는지 나눠서
계산한다 — 한 환자가 case로도, control로도 여러 번 나올 수 있어서 역할별로 안 나누면 해석이
어려움.

clinical CSV(ajcc_stage/tumor_grade)를 붙여서, 고레버리지 환자들이 특정 병기/등급에 쏠려
있는지도 같이 보여준다.

사용법:
    python scripts/audit_leverage_patients.py --dataset tcga \
        --model PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD --seeds 42,84,126 --n-folds 5 --top-n 20
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


def _load_seed_predictions(pred_dir: Path, dataset: str, model: str, seed: int, n_folds: int) -> dict:
    preds = {}
    for fold in range(n_folds):
        path = pred_dir / f"{dataset}_{model}_FOLD{fold}OF{n_folds}_seed{seed}_fold{fold}of{n_folds}.csv"
        if not path.exists():
            path = pred_dir / f"{dataset}_{model}_seed{seed}_fold{fold}of{n_folds}.csv"
        if not path.exists():
            raise FileNotFoundError(f"seed={seed} fold={fold} 예측 파일을 못 찾음: {path}")
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                cid = row["case_id"]
                if cid in preds:
                    raise ValueError(f"seed={seed} 내에서 case_id 중복: {cid}")
                preds[cid] = (float(row["risk"]), float(row["OS_time"]), int(row["OS_event"]))
    return preds


def _load_clinical(dataset: str) -> dict:
    path = _ROOT / "data" / f"clinical_{dataset}.csv"
    out = {}
    if not path.exists():
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["case_id"]] = {
                "ajcc_stage": row.get("ajcc_stage", ""),
                "tumor_grade": row.get("tumor_grade", ""),
                "residual_disease": row.get("residual_disease", ""),
            }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default="tcga", choices=["tcga", "cptac"])
    parser.add_argument("--model", type=str, default="PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD")
    parser.add_argument("--seeds", type=str, default="42,84,126")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    pred_dir = _ROOT / ".logs" / "kfold_preds"

    per_seed_preds = {seed: _load_seed_predictions(pred_dir, args.dataset, args.model, seed, args.n_folds)
                       for seed in seeds}
    common_cases = sorted(set.intersection(*(set(p.keys()) for p in per_seed_preds.values())))

    risks, times, events = [], [], []
    for cid in common_cases:
        seed_risks = [per_seed_preds[s][cid][0] for s in seeds]
        _r, t, e = per_seed_preds[seeds[0]][cid]
        risks.append(float(np.mean(seed_risks)))
        times.append(t)
        events.append(e)
    risks = np.array(risks)
    times = np.array(times)
    events = np.array(events, dtype=bool)
    n = len(common_cases)

    clinical = _load_clinical(args.dataset)

    m_full = compute_survival_metrics(risks, times, events)
    print(f"=== {args.dataset} {args.model} — 3시드 앙상블 OOF (N={n}, events={int(events.sum())}) ===")
    print(f"  전체 c_index={m_full['c_index']:.4f}\n")

    # comparable[i,j]: i(case, 더 이른 시점에 사망)와 j(control, 그 시점까지 관찰됨)가 comparable
    comparable = (times[:, None] < times[None, :]) & events[:, None]
    concordant = comparable & (risks[:, None] > risks[None, :])
    tied = comparable & (risks[:, None] == risks[None, :])
    # discordant = comparable & ~concordant & ~tied

    total_comparable = int(comparable.sum())
    total_concordant = int(concordant.sum())
    total_tied = int(tied.sum())
    c_full = (total_concordant + 0.5 * total_tied) / total_comparable
    assert abs(c_full - m_full["c_index"]) < 1e-9, "재계산 c-index가 compute_survival_metrics와 불일치"

    # leave-one-out: 환자 p를 행/열에서 모두 제외했을 때의 c-index (대각선은 항상 False라 이중차감 없음)
    row_comp = comparable.sum(axis=1)
    col_comp = comparable.sum(axis=0)
    row_conc = concordant.sum(axis=1)
    col_conc = concordant.sum(axis=0)
    row_tied = tied.sum(axis=1)
    col_tied = tied.sum(axis=0)

    loo_comparable = total_comparable - row_comp - col_comp
    loo_concordant = total_concordant - row_conc - col_conc
    loo_tied = total_tied - row_tied - col_tied
    with np.errstate(invalid="ignore", divide="ignore"):
        loo_c = (loo_concordant + 0.5 * loo_tied) / loo_comparable
    delta = c_full - loo_c  # 양수: 그 환자가 c-index를 떠받침 / 음수: 빼면 오히려 c-index가 오름(발목잡는 환자)

    involvement = row_comp + col_comp
    disc_as_case = row_comp - row_conc - row_tied
    disc_as_control = col_comp - col_conc - col_tied
    total_discordant_of_p = disc_as_case + disc_as_control
    with np.errstate(invalid="ignore", divide="ignore"):
        disc_frac = np.where(involvement > 0, total_discordant_of_p / involvement, np.nan)

    risk_z = (risks - risks.mean()) / risks.std()

    def fmt_row(idx):
        cid = common_cases[idx]
        cl = clinical.get(cid, {})
        return (f"{cid:15s} risk={risks[idx]:7.3f}(z={risk_z[idx]:+.2f}) "
                f"time={times[idx]:6.0f} event={int(events[idx])} "
                f"involve={involvement[idx]:3d} disc_frac={disc_frac[idx]:.3f} "
                f"delta={delta[idx]:+.5f} "
                f"stage={cl.get('ajcc_stage','?'):10s} grade={cl.get('tumor_grade','?')}")

    print(f"--- 이 환자를 빼면 c-index가 오히려 오르는(=발목 잡는) 상위 {args.top_n}명 (delta 오름차순) ---")
    order = np.argsort(delta)  # 가장 음수부터
    for idx in order[:args.top_n]:
        print("  " + fmt_row(idx))

    print(f"\n--- 이 환자 혼자 c-index를 크게 떠받치는(과대적합/우연 신호 의심) 상위 {args.top_n}명 (delta 내림차순) ---")
    for idx in order[::-1][:args.top_n]:
        print("  " + fmt_row(idx))

    print(f"\n--- involvement>={n//4} (충분히 많은 쌍에 관여) 중 discordant 비율 상위 {args.top_n}명 ---")
    min_involve = max(5, n // 4)
    mask = involvement >= min_involve
    idxs = np.where(mask)[0]
    idxs = idxs[np.argsort(-disc_frac[idxs])]
    for idx in idxs[:args.top_n]:
        print("  " + fmt_row(idx))

    print(f"\n--- risk score 원값 기준 |z|>2.5 이상치 ---")
    outlier_idxs = np.where(np.abs(risk_z) > 2.5)[0]
    if len(outlier_idxs) == 0:
        print("  없음")
    else:
        for idx in outlier_idxs[np.argsort(-np.abs(risk_z[outlier_idxs]))]:
            print("  " + fmt_row(idx))

    # 최악 상위 5/10/20명을 "한꺼번에" 빼면 (개별 delta 합과 달리 상호작용까지 반영) c-index가 얼마나 바뀌는지
    print(f"\n--- 최악(delta 최하위) 상위 K명을 한꺼번에 제외했을 때 c-index 변화 (상호작용 포함, 개별 delta 합과 다를 수 있음) ---")
    for k in (1, 3, 5, 10, 20):
        if k > n:
            break
        drop_idx = order[:k]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[drop_idx] = False
        m_k = compute_survival_metrics(risks[keep_mask], times[keep_mask], events[keep_mask])
        print(f"  최악 {k:2d}명 제외 -> N={int(keep_mask.sum())}, c_index={m_k['c_index']:.4f} "
              f"(전체 대비 {m_k['c_index']-c_full:+.4f})")

    # censoring 여부가 고레버리지군에 쏠려있는지 베이스라인과 비교
    worst_k = order[:max(10, args.top_n)]
    base_event_rate = events.mean()
    worst_event_rate = events[worst_k].mean()
    print(f"\n--- censoring 편중 체크 ---")
    print(f"  전체 event rate={base_event_rate:.3f} | 최악 {len(worst_k)}명 event rate={worst_event_rate:.3f}")


if __name__ == "__main__":
    main()
