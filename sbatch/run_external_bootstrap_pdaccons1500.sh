#!/bin/bash
# 2026-09-05: M4(pdac_consistency_1500+CNV+mutation+STG+R+CLR100, uni2native) external(CPTAC)
# bootstrap. GPU 불필요(예측 CSV는 이미 train.py --external이 학습 중 저장해둠, 여기선 그 CSV들을
# 읽어서 부트스트랩 통계만 냄) — 로그인 노드에서 바로 실행해도 된다(sbatch 불필요).
#
# 실행: bash sbatch/run_external_bootstrap_pdaccons1500.sh

cd /pub/wonseukl/Path-ViT/
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

python scripts/pool_multiseed_external_preds.py --dataset cptac \
    --model M4_uni2native_PDACCONS1500_CNV_STG_R_MUT_COX_ADD_CLR100_LRMW10 \
    --seeds 84,126 --n-folds 5 --bootstrap 2000
