"""
scripts/final_eval_external_ckpt_sweep.py의 internal 버전 — 2026-08-30, M4_noaux/M4_nodisp의
internal kfold_preds CSV를 실수로 지운 사고 복구용으로 만들었다. train.py --eval-internal-ckpt
(2026-08-30 추가, --eval-external-ckpt와 동일 관례)로 이미 저장된 checkpoint를 다시 읽어 internal
held-out fold 예측만 재추출한다(재학습 없음). CONFIGS는 external 스윕 스크립트 것을 그대로
재사용 — 어차피 같은 checkpoint를 대상으로 하므로 중복 정의하지 않는다.

HPC에서 실행:
    python -m scripts.final_eval_internal_ckpt_sweep --only M4_noaux,M4_nodisp
    python -m scripts.final_eval_internal_ckpt_sweep          # --only 생략하면 CONFIGS 전체
"""
import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.final_eval_external_ckpt_sweep import CONFIGS, N_FOLDS, SEEDS, _find_ckpt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", type=str, default=None,
                         help="콤마로 구분한 label 목록(CONFIGS의 첫 항목, 예: M4_noaux,M4_nodisp). "
                              "생략하면 전체 CONFIGS를 순서대로 실행.")
    args = parser.parse_args()

    configs = CONFIGS
    if args.only:
        wanted = set(args.only.split(","))
        known = {c[0] for c in CONFIGS}
        unknown = wanted - known
        if unknown:
            raise ValueError(f"CONFIGS에 없는 label: {sorted(unknown)} (있는 것: {sorted(known)})")
        configs = [c for c in CONFIGS if c[0] in wanted]

    n_ok, n_skip, n_fail = 0, 0, 0
    for label, train_args, include, exclude, suffix in configs:
        print(f"\n########## {label} ##########")
        for seed in SEEDS:
            for fold in range(N_FOLDS):
                ckpt = _find_ckpt(seed, fold, include, exclude, suffix)
                if ckpt is None:
                    n_skip += 1
                    continue
                cmd = [sys.executable, "train.py", *train_args,
                       "--dataset", "tcga", "--external", "--seed", str(seed),
                       "--fold", str(fold), "--n-folds", str(N_FOLDS),
                       "--eval-internal-ckpt", str(ckpt)]
                print(f"  seed={seed} fold={fold}: {ckpt.name}")
                result = subprocess.run(cmd, cwd=_ROOT)
                if result.returncode == 0:
                    n_ok += 1
                else:
                    n_fail += 1
                    print(f"  [FAIL] seed={seed} fold={fold} (exit {result.returncode})")
    print(f"\n=== 완료: 성공 {n_ok} / 실패 {n_fail} / 체크포인트 못 찾음(skip) {n_skip} (전체 {len(configs) * len(SEEDS) * N_FOLDS}) ===")


if __name__ == "__main__":
    main()
