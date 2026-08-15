#!/bin/bash
#SBATCH --job-name=PVT-M1-novit-seed42-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m1_novit_seed42_kfold_array_%a.log

# 2026-08-14: M1~M4 사다리를 M4-NOVIT(단일 gated ABMIL + patch-mixing transformer 제거)과
# 같은 WSI 아키텍처로 통일하는 작업의 M1(WSI 단독). M1(ViT_M1, 기본 모델)은 원래부터 단일
# gated ABMIL이고 skip_patch_vit도 이미 지원했지만, train.py의 기본(--M1) 생성 코드가
# skip_patch_vit=args.skip_patch_vit를 안 넘겨서 실질적으로 못 쓰고 있던 걸 오늘 고쳤다.
#
# 레시피: M4-NOVIT에서 SS(patch-keep-frac)/AUX(rna-aux-weight)/FiLM(RNA 조건화, M1엔 RNA가
# 없으니 자동으로 없음 — self ABMIL)을 뺀 나머지(skip-patch-vit, attn-dispersion)만 유지.
# SS/AUX는 RNA가 있는 모델(M3/M4)에서만 의미가 있는 조합(findings_backlog.md 9번 항목)이라
# RNA가 아예 없는 M1엔 애초에 안 맞는다.
#
# 완료 후(5개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 5개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M1_uni2_DISP_NOVIT --seeds 42 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/m1_novit_seed42_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=42
N_FOLDS=5
FOLD=$SLURM_ARRAY_TASK_ID

log=".logs/train_tcga_seed${SEED}_M1_uni2_DISP_NOVIT_kfold5_fold${FOLD}.log"

echo "=== M1(uni2,skip-patch-vit,DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M1 --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --attn-dispersion \
    --skip-patch-vit \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0814m1_novit_seed42_kfold5_array 2>&1 | tee "${log}"
echo "=== M1(uni2,skip-patch-vit,DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
