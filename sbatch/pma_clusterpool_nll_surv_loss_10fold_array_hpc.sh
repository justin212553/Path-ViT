#!/bin/bash
#SBATCH --job-name=PVT-PMA-clusterpool-nllsurv-10fold
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_clusterpool_nll_surv_loss_10fold_array_%a.log

# 2026-09-06: sbatch/pma_nll_surv_loss_10fold_array_hpc.sh와 완전히 동일하지만 --cluster-pool만
# 추가 — models/vit_pma.py::ViT_PMA cluster_pool=True(2026-09-05 도입, Nystrom+ABMIL을 사전계산
# unsupervised 군집 대표값(K=10)으로 대체). 별도 모델 클래스가 아니라 --PMA의 플래그 하나라
# 나머지 레시피(rna-genes/combine-mode/CLR100/CNV/mutation/nll_surv 등)는 그대로 유지.
#
# --cluster-centroids-path 기본값(None -> data/cluster_centroids_{backbone}.pt)을 그대로
# 쓴다 — backbone=uni2native면 data/cluster_centroids_uni2native_k10_tcgacptac.pt(로컬 확인
# 완료, CPU 생성 테스트로 실제 로딩까지 검증됨).
#
# 2026-09-06 수정: 최초 제출본은 --attn-dispersion을 베이스 PMA 레시피에서 그대로 물려받았는데,
# cluster_pool=True에선 attn_pool(patch-level attention) 자체가 없어(forward()가 cluster
# 대표값만 만들고 조기 return) attn_dispersion이 필요로 하는 attn_weights가 존재하지 않는다 —
# risk_head는 여전히 그 차원을 기대해서 LayerNorm shape 에러로 10개 fold 전부 시작하자마자
# 크래시했다(모델 초기화만 하고 학습은 한 스텝도 못 감). models/vit_pma.py에 이 조합을 막는
# 명시적 검증도 추가했다 — 여기서는 --attn-dispersion을 아예 뺐다(cluster_pool은 애초에
# patch-level attention을 안 쓰므로 "attention 분산"이라는 개념 자체가 성립 안 함).
#
# [2026-09-06 수정] "10개 fold를 한번에"를 1seed x 10fold(fold당 테스트 ~15명)로 잘못 해석했다
# — 이 프로젝트의 실제 "10개" 관례는 **2seed(84,126) x 5fold**(seed42는 WSI 포함 모델에서 유독
# 튀는 값이 나와 최종 집계에서 제외해온 관례, paper/final_results_summary.md). 5fold(fold당
# ~30명)로 되돌리고 시드를 2개로 늘려 검정력을 확보한다. sbatch/porpoise_no_aux_multiseed_
# kfold_array_hpc.sh와 동일한 SEED_IDX/FOLD 인덱싱 관례(3seed 대신 2seed라 --array=0-9는
# 그대로, 의미만 5fold*2seed로 바뀜).
#
# 완료 후: 정확한 모델 태그는 `ls .logs/kfold_preds/tcga_PMA*CLUSTERPOOL*NLLSURV4*`로 확인 후
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model <확인한 태그> --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/pma_clusterpool_nll_surv_loss_10fold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(84 126)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

log=".logs/train_tcga_seed${SEED}_PMA_uni2native_INT1500_CNV_SS_AUX_STG_R_MUT_CLUSTERPOOL_NLLSURV4_kfold5_fold${FOLD}.log"

echo "=== PMA+ClusterPool nll_surv loss(uni2native,cox_add,CLR100,CNV,MUT) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 --rna-aux-weight 1.0 \
    --cluster-pool \
    --surv-loss nll_surv --nll-n-bins 4 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0906pma_clusterpool_nllsurv_2seed_kfold5_array 2>&1 | tee "${log}"
echo "=== PMA+ClusterPool nll_surv loss(uni2native,cox_add,CLR100,CNV,MUT) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
