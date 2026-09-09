#!/bin/bash
# 2026-09-09: M1~M7 ablation 테이블 최종 레시피 재학습을 HPC array 대신 로컬에서 순차 실행.
# 사용자 결정 — 대규모(6모델 x 2seed x 5fold = 60회) 학습을 HPC array로 올리면 어느 seed/fold가
# preemption으로 끊길지 예측 불가능하고 결과 취합도 흩어져 어려워지므로, 로컬 GPU(RTX 4060)에서
# 한 번에 하나씩 느긋하게 순차 실행한다 — sbatch/{m1,m2,pma_clusterpool_pdaccons_m3,m5,m6,m7}_*
# _hpc.sh의 python 커맨드를 그대로 재사용(SLURM_ARRAY_TASK_ID 대신 bash for문으로 seed/fold를
# 돈다). M4(=PMA ClusterPool 확정 레시피)는 이미 완료돼 있어 여기서 제외.
#
# 순서: 모델 단위로 seed x fold 10개를 전부 끝낸 뒤 다음 모델로 넘어간다(fold 단위로 모델을
# 섞지 않음) — 빠른 WSI-free 모델(M5/M6/M7)부터 돌려 새로 이식한 코드 경로(surv_n_classes,
# --surv-loss CLI 배선)를 빨리 검증하고, 한 모델이 끝날 때마다 바로 pooling/bootstrap이
# 가능하게 한다. 느린 WSI 모델(M1/M2/M3)은 뒤로 미뤄 밤새 돌아가게 한다.
#
# 재실행 가능(resumable) — 각 실행의 로그 파일이 이미 있고 "Complete:"로 끝나 있으면 건너뛴다.
# 중간에 컴퓨터를 끄거나 스크립트를 다시 실행해도 이미 끝난 조합은 다시 안 돈다. 실패한 실행은
# 스크립트 전체를 멈추지 않고 다음 조합으로 넘어간다(밤새 하나 실패했다고 나머지가 안 돌아가면
# 안 되므로) — 실패 목록은 마지막에 요약해서 출력한다.
#
# 로컬 데이터 확인(2026-09-09): data/patches_{tcga,cptac}/tiles/{slide}/features_uni2native.pt가
# 이미 로컬에 있음(TCGA 466 슬라이드, CPTAC 564 슬라이드) — WSI feature 전송 불필요.
#
# conda env: PathViT-ray(로컬 관례, HPC의 Path-ViT와 다름 — feedback_pathvit_conda_env 메모).
#
# 실행: bash scripts/run_all_ablations_local_sequential.sh
#   (백그라운드로 밤새 돌리려면: nohup bash scripts/run_all_ablations_local_sequential.sh > .logs/run_all_ablations_local.out 2>&1 &)

cd "$(dirname "$0")/.."
set -o pipefail  # run_if_needed의 성공 판정이 tee가 아니라 실제 python 종료코드를 보게 함

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PathViT-ray

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

mkdir -p .logs

SEEDS=(84 126)
N_FOLDS=5
FAILED=()
SKIPPED=0
RAN=0

run_if_needed() {
    # $1 = log 파일 경로, 나머지 인자 = 실행할 커맨드
    local log_file="$1"; shift
    if [ -f "$log_file" ] && grep -q "Complete:" "$log_file"; then
        echo "[SKIP, 이미 완료] $log_file"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi
    echo "[RUN] $log_file  ($(date))"
    if "$@" 2>&1 | tee "$log_file"; then
        echo "=== Complete: $(date) ===" >> "$log_file"
        RAN=$((RAN + 1))
    else
        echo "[FAIL] $log_file"
        FAILED+=("$log_file")
    fi
}

echo "############################################################"
echo "# M1~M7(M4 제외, 이미 완료) 확정 레시피 로컬 순차 실행 시작: $(date)"
echo "############################################################"

echo ">>> M5(ClinicalOnly, both-loss) 시작: $(date)"
for SEED in "${SEEDS[@]}"; do
  for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    log=".logs/train_tcga_m5_final_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"
    run_if_needed "$log" python -u ./train_light.py --M5 --raw-linear --dataset tcga --external --seed "${SEED}" \
        --clinical-margin --clinical-staging \
        --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0909m5_final_2seed_kfold5_local
  done
done

echo ">>> M6(RNAOnly, pdac_consistency_1500, both-loss) 시작: $(date)"
for SEED in "${SEEDS[@]}"; do
  for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    log=".logs/train_tcga_m6_pdaccons_final_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"
    run_if_needed "$log" python -u ./train_light.py --M6 --rna-genes pdac_consistency_1500 --dataset tcga --external --seed "${SEED}" \
        --use-cnv \
        --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0909m6_pdaccons_final_2seed_kfold5_local
  done
done

echo ">>> M7(ClinicalRNAOnly, pdac_consistency_1500, both-loss) 시작: $(date)"
for SEED in "${SEEDS[@]}"; do
  for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    log=".logs/train_tcga_m7_pdaccons_final_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"
    run_if_needed "$log" python -u ./train_light.py --M7 --rna-genes pdac_consistency_1500 --dataset tcga --external --seed "${SEED}" \
        --use-cnv --clinical-margin --clinical-staging --clinical-mutation --combine-mode cox_add \
        --clinical-lr-mult 100 \
        --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0909m7_pdaccons_final_2seed_kfold5_local
  done
done

echo ">>> M1(WSI only, ClusterPool) 시작: $(date)"
for SEED in "${SEEDS[@]}"; do
  for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    log=".logs/train_tcga_m1_clusterpool_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"
    run_if_needed "$log" python -u ./train.py --M1 --cluster-pool --dataset tcga --external --seed "${SEED}" \
        --backbone uni2native \
        --patch-keep-frac 0.8 \
        --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0909m1_clusterpool_final_2seed_kfold5_local
  done
done

echo ">>> M2(WSI+Clinical, ClusterPool) 시작: $(date)"
for SEED in "${SEEDS[@]}"; do
  for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    log=".logs/train_tcga_m2_clusterpool_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"
    run_if_needed "$log" python -u ./train.py --M2 --cluster-pool --dataset tcga --external --seed "${SEED}" \
        --backbone uni2native \
        --clinical-staging --clinical-margin --clinical-lr-mult 100 \
        --patch-keep-frac 0.8 \
        --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0909m2_clusterpool_final_2seed_kfold5_local
  done
done

echo ">>> M3(PMA ClusterPool, no-clinical) 시작: $(date)"
for SEED in "${SEEDS[@]}"; do
  for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    log=".logs/train_tcga_m3_pma_clusterpool_noclinical_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"
    run_if_needed "$log" python -u ./train.py --PMA --cluster-pool --no-clinical --rna-genes pdac_consistency_1500 --dataset tcga --external --seed "${SEED}" \
        --backbone uni2native \
        --use-cnv \
        --patch-keep-frac 0.8 \
        --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0909m3_pma_clusterpool_noclinical_2seed_kfold5_local
  done
done

echo "############################################################"
echo "# 전체 완료: $(date)"
echo "# 새로 실행: ${RAN}개 | 건너뜀(이미 완료): ${SKIPPED}개 | 실패: ${#FAILED[@]}개"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "# 실패한 로그:"
    for f in "${FAILED[@]}"; do echo "#   $f"; done
fi
echo "############################################################"
