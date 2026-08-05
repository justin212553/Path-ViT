#!/bin/bash
#SBATCH --job-name=PVT-M3-aug-kfold
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/train_m3_aug_kfold.log

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

# 2026-08-05: train_pma_aug_kfold_hpc.sh와 동일한 패턴(GPU 1개짜리 SLURM job 안에서 5개 fold를
# 백그라운드 프로세스로 동시 실행) — train_m3_kfold_array_hpc.sh(SLURM job array, GPU 5개 필요)
# 대신 GPU 1개만 쓴다. WSI backbone(CNN+ViT) forward가 비용의 ~99%를 차지해 M1/M2/M3/PMA가
# GPU 메모리 사용량이 거의 같고, PMA(RNA+clinical 인코더까지 얹은 가장 무거운 모델)가 A30
# 하나에 5개 동시 실행이 들어간 걸 이미 확인했으니 M3(WSI+RNA, clinical 제외)도 들어갈 것으로
# 예상.
# --rna-genes literature_1500_intersection: train_m3_kfold_array_hpc.sh와 동일하게 leakage-free
# TCGA-only/CPTAC-only 교집합 유전자셋(문헌 curated 163개+교집합 1337개=1500개) 사용 — 예전
# both-결합 literature_1500(leaky)은 안 쓴다.
# --tile-decode-workers 4: 프로세스당(위 --cpus-per-task=20을 5개 프로세스가 나눠 쓰는 셈).
# M3(WSI+RNA, ViT_PMA --no-clinical) — RNA가 있으니 SS+AUX+AUG+DISP 전부.
#
# 완료 후 집계(model_prefix에 --no-clinical의 _NOCLINICAL 태그가 들어간다는 점 주의):
#   python scripts/summarize_kfold.py --dataset tcga --seed 84 --n-folds 5 --model PMA_INT1500_SS_AUX_AUG_NOCLINICAL_DISP
#
# 제출: sbatch scripts/train_m3_aug_kfold_hpc.sh
for fold in 0 1 2 3 4; do
    python -u ./train.py --PMA --no-clinical --rna-genes literature_1500_intersection --dataset tcga --external --seed 84 \
        --tile-augment --image --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
        --tile-decode-workers 4 --cache-val-tiles --cache-external-tiles \
        --fold "$fold" --n-folds 5 "$@" \
        > ".logs/m3_aug_kfold_fold${fold}.log" 2>&1 &
done
wait
echo "=== 5개 fold 전부 종료 ==="
