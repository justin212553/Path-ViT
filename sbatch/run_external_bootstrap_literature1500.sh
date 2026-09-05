#!/bin/bash
# 2026-09-05: M4(literature_1500+CNV+mutation+STG+R+CLR100, uni2native) external(CPTAC)
# bootstrap. GPU 불필요 — 로그인 노드에서 바로 실행해도 된다(sbatch 불필요).
#
# 주의: --model 태그는 "_EX"(literature_1500 자체는 train.py model_prefix 로직상 유전자 수를
# 태그에 안 남기고 그냥 "_EX"만 붙음, "_EXT1500"이 아님, 2026-09-05 train.py 코드로 직접 확인).
# 실제 .logs/kfold_preds/, .logs/external_preds/ 파일명과 다르면 그 파일명 기준으로 --model 값을
# 맞출 것.
#
# 실행: bash sbatch/run_external_bootstrap_literature1500.sh

cd /pub/wonseukl/Path-ViT/
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

python scripts/pool_multiseed_external_preds.py --dataset cptac \
    --model M4_uni2native_EX_CNV_STG_R_MUT_COX_ADD_CLR100_LRMW10 \
    --seeds 84,126 --n-folds 5 --bootstrap 2000
