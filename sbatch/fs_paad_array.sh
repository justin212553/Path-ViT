#!/bin/bash
#SBATCH --job-name=PVT-FS-PAAD
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-24
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/fs_paad_%x_%a.log

# 2026-09-29: 리뷰 대응 재실행 배치(PAAD) — paper/TODO_reviewer_feedback.md 1~2절.
# 모든 모델 공통:
#   --fold-safe                     RNA/CNV z-score, clinical 통계, ClusterPool centroid를 fold train만으로
#   --restrict-cohort-file n110     M1~M7 전부 같은 코호트(TCGA 110 + CPTAC 136)
#   5 seeds x 5 folds = 25 array task
# 모델은 MODEL 환경변수로 고른다 — 직접 제출하지 말고 sbatch/submit_fs_batch.sh를 쓸 것
# (모델별 job 이름·자원을 맞춰 제출함). 예: sbatch --export=ALL,MODEL=M4 sbatch/fs_paad_array.sh
#
# 선행 조건: data/cohort_n110.txt, data/rna_{tcga,cptac}_raw_log2.csv가 HPC에 있어야 함
# (로컬에서 생성 — 없으면 fold-safe가 FileNotFoundError).
#
# 완료 확인: python scripts/check_pred_completeness.py --tag _FS_COHN110 (또는 _FS)

cd /pub/wonseukl/Path-ViT/
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(84 126 42 168 210)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$((IDX / N_FOLDS))]}
FOLD=$((IDX % N_FOLDS))

COMMON="--dataset tcga --external --seed ${SEED} --fold ${FOLD} --n-folds ${N_FOLDS} \
  --fold-safe --restrict-cohort-file data/cohort_n110.txt \
  --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 --group-ts 0929fs_paad_${MODEL}"
WSI="--cluster-pool --backbone uni2native --patch-keep-frac 0.8"
RNA="--rna-genes pdac_consistency_1500"
CLIN="--clinical-staging --clinical-margin --clinical-lr-mult 100"

case "$MODEL" in
  # --- 기본 M1~M7 ---
  M1)        CMD="train.py --M1 $WSI" ;;
  M2)        CMD="train.py --M2 $WSI $CLIN" ;;
  M3)        CMD="train.py --PMA $WSI --no-clinical $RNA --use-cnv" ;;
  M4)        CMD="train.py --PMA $WSI --combine-mode cox_add $RNA --use-cnv $CLIN --clinical-mutation" ;;
  M5)        CMD="train_light.py --M5 --raw-linear --clinical-margin --clinical-staging" ;;
  M6)        CMD="train_light.py --M6 $RNA --use-cnv" ;;
  M7)        CMD="train_light.py --M7 $RNA --use-cnv --combine-mode cox_add $CLIN --clinical-mutation" ;;
  # --- co-attention 분리 (입력 동일, fusion만 self-attention) ---
  M3SA)      CMD="train.py --PMA $WSI --self-attn-fusion --no-clinical $RNA --use-cnv" ;;
  M4SA)      CMD="train.py --PMA $WSI --self-attn-fusion --combine-mode cox_add $RNA --use-cnv $CLIN --clinical-mutation" ;;
  # --- RNA-seq / CNV 분리 [정찬권 #1,#2] ---
  M3NOCNV)   CMD="train.py --PMA $WSI --no-clinical $RNA" ;;
  M4NOCNV)   CMD="train.py --PMA $WSI --combine-mode cox_add $RNA $CLIN --clinical-mutation" ;;
  M6NOCNV)   CMD="train_light.py --M6 $RNA" ;;
  M7NOCNV)   CMD="train_light.py --M7 $RNA --combine-mode cox_add $CLIN --clinical-mutation" ;;
  # --- mutation 분리 [정찬권 #1,#6, 결정 A] ---
  M4NOMUT)   CMD="train.py --PMA $WSI --combine-mode cox_add $RNA --use-cnv $CLIN" ;;
  # --- 진단(2026-10-04): fold-safe 유지, centroid만 기존 고정 파일 — co-attention 이득 감소 원인 분리 ---
  M4FIXC)    CMD="train.py --PMA $WSI --fs-fixed-centroids --combine-mode cox_add $RNA --use-cnv $CLIN --clinical-mutation" ;;
  M4SAFIXC)  CMD="train.py --PMA $WSI --fs-fixed-centroids --self-attn-fusion --combine-mode cox_add $RNA --use-cnv $CLIN --clinical-mutation" ;;
  *) echo "unknown MODEL=$MODEL"; exit 1 ;;
esac

echo "=== FS PAAD ${MODEL} seed=${SEED} fold=${FOLD} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
echo "python -u ./${CMD} ${COMMON}"
python -u ./${CMD} ${COMMON}
echo "=== FS PAAD ${MODEL} seed=${SEED} fold=${FOLD} Complete: $(date) ==="
