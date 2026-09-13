#!/bin/bash
#SBATCH --job-name=PVT-porpoise-mmf-eval-cptac
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/eval_porpoise_official_paad_mmf_external_cptac_array_%a.log

# 2026-09-12: 순정 PORPOISE 공식 레시피(run_porpoise_official_paad_mmf_hpc.sh /
# run_porpoise_official_paad_mmf_seed84_hpc.sh, results_true_resnet50_mmf)를 CPTAC-PDAC에
# 평가하는 스크립트 — 지금까지 porpoise/eval_external.py는 own-RNA 재학습 체크포인트
# (results_ownrna_mmf, porpoise_ownrna_mmf_eval_cptac_5seed_array_hpc.sh)에만 배선돼 있었고,
# 순정 공식 레시피용 external 평가는 존재하지 않았다(2026-09-12 리포지토리 재조사로 확인된
# 공백). eval_external.py 자체는 건드리지 않는다 — 순정 레시피가 own-RNA 기본값과 다른 부분은
# --results-dir과 --tcga-split-dir 두 개뿐이라 CLI 인자로 오버라이드하는 것으로 충분하다.
#
# 오버라이드 근거(porpoise/main.py 직접 대조):
#   - run_porpoise_official_paad_mmf_hpc.sh는 --split_dir tcga_paad로 학습했다. main.py:247이
#     이걸 './splits/5foldcv/tcga_paad'로 바꾸므로, eval_external.py의 기본값
#     'splits/5foldcv/tcga_paad_ownrna'(own-RNA 전용 split)는 여기 맞지 않는다 —
#     --tcga-split-dir splits/5foldcv/tcga_paad로 명시.
#   - --results-dir도 학습이 쓴 실제 경로(results_true_resnet50_mmf)로 맞춘다(own-RNA 기본값
#     results_ownrna_mmf가 아님).
#   - --tcga-csv, --cptac-csv, --cptac-data-root는 eval_external.py 기본값을 그대로 쓴다 —
#     main.py:214(csv_path = './%s/%s_all_clean.csv.zip' % (args.dataset_path, study),
#     --split_dir tcga_paad일 때 study='tcga_paad')와 eval_external.py 기본값
#     'datasets_csv/tcga_paad_all_clean.csv.zip'이 정확히 일치함을 직접 코드 대조로 확인했다.
#   - 모델 하이퍼파라미터(fusion=bilinear, gate_path/gate_omic/skip=True, dropinput=0.10)는
#     run_porpoise_official_paad_mmf_hpc.sh의 학습 커맨드와 eval_external.py의 기본값이 이미
#     동일하므로 별도로 지정하지 않는다.
#
# **미확인 위험(스크립트 자체 안전장치로 방어됨)**: --cptac-csv 기본값
# (datasets_csv/cptac_paad_external_clean.csv.zip)이 순정 레시피의 유전자 universe
# (tcga_paad_all_clean.csv.zip 기준)와 실제로 같은 컬럼 순서/구성인지는 로컬에 두 CSV가 없어
# 직접 대조하지 못했다 — own-RNA 전용으로 준비됐을 가능성을 배제 못 한다. 다만 이 경우
# eval_external.py 자신의 검증 로직(line 105-111, missing gene 발견 시 ValueError로 즉시
# 실패하며 어떤 유전자가 빠졌는지 출력)이 조용한 오류 없이 바로 잡아낸다. 이 에러가 나면
# scripts/prepare_porpoise_cptac_external_data.py를 순정 레시피의 유전자 universe로 다시
# 돌려야 한다는 뜻이다.
#
# 선행 조건:
#   1) run_porpoise_official_paad_mmf_hpc.sh(seed=1) 완료 확인
#        ls porpoise/results_true_resnet50_mmf/5foldcv/*/tcga_paad_s1/s_*_checkpoint.pt | wc -l   # 5
#   2) run_porpoise_official_paad_mmf_seed84_hpc.sh(seed=84, 2026-09-12 기준 아직 제출된 적
#      없음 — 이 배열 잡보다 먼저 제출해서 완료를 기다려야 함) 완료 확인
#        ls porpoise/results_true_resnet50_mmf/5foldcv/*/tcga_paad_s84/s_*_checkpoint.pt | wc -l  # 5
#
# 완료 후 pooling(이 프로젝트의 기존 external pooling 스크립트를 그대로 재사용, 새 스크립트
# 불필요 — model 태그만 다르게):
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PORPOISE_MMF --seeds 1,84 --n-folds 5 --bootstrap 2000
#
# internal(5fold, 2seed pooled)은 이미 있는 스크립트로:
#   python scripts/pool_porpoise_official_kfold.py --results-dir porpoise/results_true_resnet50_mmf \
#       --seeds 1,84 --bootstrap 2000
#
# 제출(두 학습 array 완료 확인 후): sbatch sbatch/eval_porpoise_official_paad_mmf_external_cptac_hpc.sh

cd /pub/wonseukl/Path-ViT/porpoise

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

SEEDS=(1 84)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

OUT_CSV="/pub/wonseukl/Path-ViT/.logs/external_preds/cptac_PORPOISE_MMF_seed${SEED}_fold${FOLD}of${N_FOLDS}.csv"

echo "=== PORPOISE 공식 레시피(순정) MMF CPTAC external eval seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u eval_external.py \
    --seed "${SEED}" --fold "${FOLD}" \
    --results-dir results_true_resnet50_mmf \
    --tcga-split-dir splits/5foldcv/tcga_paad \
    --tcga-data-root /pub/wonseukl/Path-ViT/porpoise/data_root_true_resnet50 \
    --cptac-data-root /pub/wonseukl/Path-ViT/data/porpoise_style_features/cptac \
    --out-csv "${OUT_CSV}"
echo "=== PORPOISE 공식 레시피(순정) MMF CPTAC external eval seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
