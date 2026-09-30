"""
2026-09-28 일회성 로컬 검증 스크립트 — 리뷰 피드백(전처리 leakage) 영향 크기를 seed=84/fold=0
하나로 빠르게 확인하기 위한 것. data.dataset.RNA_PATHS를 fold-safe CSV(이 fold의 train-only
통계로 재정규화된 값)로 몽키패치한 뒤 train.py를 그대로 호출한다. cluster centroids는
--cluster-centroids-path로 별도 전달(코드 수정 불필요, 기존 인자 재사용).

결과가 유의미하면 이후 정식 CLI 플래그(예: --fold-safe-rna)로 옮길 것 — 지금은 이 결과 하나만
빠르게 보기 위한 임시 스크립트.

사용법: python scripts/run_train_foldsafe_test.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import data.dataset as dataset_mod

dataset_mod.RNA_PATHS["tcga"] = "data/rna_tcga_foldsafe_seed84_fold0.csv"
dataset_mod.RNA_PATHS["cptac"] = "data/rna_cptac_foldsafe_seed84_fold0.csv"

sys.argv = [
    "train.py",
    "--PMA", "--cluster-pool",
    "--cluster-centroids-path", "data/ccu_fs.pt",
    "--combine-mode", "cox_add",
    "--rna-genes", "pdac_consistency_1500",
    "--dataset", "tcga", "--external",
    "--seed", "84",
    "--backbone", "uni2native",
    "--clinical-margin", "--clinical-staging",
    "--clinical-lr-mult", "100", "--use-cnv", "--clinical-mutation",
    "--patch-keep-frac", "0.8",
    "--surv-loss", "both", "--nll-n-bins", "4", "--nll-cox-weight", "1.0",
    "--fold", "0", "--n-folds", "5",
    "--group-ts", "foldsafe_leakage_test_seed84_fold0",
]

import train  # noqa: E402

train.main()
