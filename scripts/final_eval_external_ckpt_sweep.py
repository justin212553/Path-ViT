"""
sbatch/final_m1m4_3seed_kfold_hpc/의 7개 학습 스크립트(M1, M2 baseline/hybrid, M3 baseline/
hybrid, M4 baseline/hybrid)가 --external로 학습은 했지만, train.py의 일반 --fold 학습
경로는 external 지표를 화면에 출력만 하고 CSV로 저장하지 않는다(--eval-external-ckpt 단독
실행 때만 저장) — 그래서 external_preds CSV가 하나도 없다. 이미 저장된 체크포인트를
--eval-external-ckpt로 다시 읽어 external만 재평가/CSV 저장한다(재학습 없음).

HPC에서 실행: python -m scripts.final_eval_external_ckpt_sweep
"""
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = _ROOT / "models" / "checkpoint"
SEEDS = [42, 84, 126]
N_FOLDS = 5

# (label, train.py construction args, ckpt 파일명에 반드시 포함될 substring,
#  ckpt 파일명에 있으면 안 되는 substring(baseline이 hybrid로 오매칭되는 것 방지), ckpt suffix)
CONFIGS = [
    ("M1", ["--M1_POOL", "--backbone", "uni2native", "--attn-dispersion", "--patch-keep-frac", "0.8"],
     "M1_POOL_uni2native_SS_DISP", None, "m1_pool"),
    ("M2_baseline", ["--M2_POOL", "--backbone", "uni2native", "--attn-dispersion", "--patch-keep-frac", "0.8"],
     "M2_POOL_uni2native_SS_DISP", "XMLP", "m2_pool"),
    ("M2_hybrid", ["--M2_POOL", "--backbone", "uni2native", "--attn-dispersion", "--patch-keep-frac", "0.8",
                   "--wsi-extra-mlp", "--clinical-lr-mult", "20.0"],
     "M2_POOL_uni2native_SS_DISP_XMLP_CLR20", None, "m2_pool"),
    ("M3_baseline", ["--PMA", "--no-clinical", "--rna-genes", "literature_1500_intersection",
                      "--backbone", "uni2native", "--patch-keep-frac", "0.8", "--attn-dispersion",
                      "--rna-aux-weight", "1.0"],
     "PMA_uni2native_INT1500_SS_AUX_NOCLINICAL_DISP", "XMLP", "pma"),
    ("M3_hybrid", ["--PMA", "--no-clinical", "--rna-genes", "literature_1500_intersection",
                    "--backbone", "uni2native", "--patch-keep-frac", "0.8", "--attn-dispersion",
                    "--rna-aux-weight", "1.0", "--wsi-extra-mlp", "--rna-lr-mult", "20.0"],
     "PMA_uni2native_INT1500_SS_AUX_NOCLINICAL_DISP_XMLP_RLR20", None, "pma"),
    ("M4_baseline", ["--PMA", "--rna-genes", "literature_1500_intersection", "--backbone", "uni2native",
                      "--clinical-staging", "--clinical-margin", "--patch-keep-frac", "0.8",
                      "--attn-dispersion", "--rna-aux-weight", "1.0", "--combine-mode", "cox_add"],
     "PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD", "XMLP", "pma"),
    ("M4_hybrid", ["--PMA", "--rna-genes", "literature_1500_intersection", "--backbone", "uni2native",
                    "--clinical-staging", "--clinical-margin", "--patch-keep-frac", "0.8",
                    "--attn-dispersion", "--rna-aux-weight", "1.0", "--combine-mode", "cox_add",
                    "--wsi-extra-mlp", "--clinical-lr-mult", "20.0", "--rna-lr-mult", "20.0"],
     "PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD_XMLP_CLR20_RLR20", None, "pma"),
]


def _find_ckpt(seed: int, fold: int, include: str, exclude: str | None, suffix: str) -> Path | None:
    pattern = f"*_FOLD{fold}OF{N_FOLDS}_best_{suffix}.pt"
    candidates = [
        p for p in CKPT_DIR.glob(pattern)
        if f"seed{seed}_" in p.name and include in p.name and (exclude is None or exclude not in p.name)
    ]
    if len(candidates) == 0:
        print(f"  [SKIP] seed={seed} fold={fold} include={include!r}: 매칭 0개")
        return None
    if len(candidates) > 1:
        # 이전 세션(예: uni2native 파일럿, paper notes 5번 항목)에 같은 태그로 이미 돌려둔 옛날
        # 체크포인트가 남아있어 겹치는 경우 — mtime이 가장 최근(오늘 새로 학습한 것)을 택한다.
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"  [WARN] seed={seed} fold={fold} include={include!r}: {len(candidates)}개 매칭 -> "
              f"최신 파일 선택: {candidates[0].name} (mtime={candidates[0].stat().st_mtime:.0f})")
    return candidates[0]


def main():
    n_ok, n_skip, n_fail = 0, 0, 0
    for label, train_args, include, exclude, suffix in CONFIGS:
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
                       "--eval-external-ckpt", str(ckpt)]
                print(f"  seed={seed} fold={fold}: {ckpt.name}")
                result = subprocess.run(cmd, cwd=_ROOT)
                if result.returncode == 0:
                    n_ok += 1
                else:
                    n_fail += 1
                    print(f"  [FAIL] seed={seed} fold={fold} (exit {result.returncode})")
    print(f"\n=== 완료: 성공 {n_ok} / 실패 {n_fail} / 체크포인트 못 찾음(skip) {n_skip} (전체 {len(CONFIGS) * len(SEEDS) * N_FOLDS}) ===")


if __name__ == "__main__":
    main()
