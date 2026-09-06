#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-nllsurv-10fold
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_nll_surv_loss_10fold_array_%a.log

# 2026-09-06: "우리 아키텍처는 그대로 두고 loss 함수만 PORPOISE 걸로 바꾸면 어떻게 되나"
# 실험 — sbatch/porpoise_no_aux_multiseed_kfold_array_hpc.sh(지금까지 PAAD에서 internal C가
# 가장 높았던 "no_aux" 레시피, seed84/fold0 단일 C=0.7119, 3seed x 5fold 논문 사양 pooled로는
# M7과 통계적으로 구분 안 됨 — paper/porpoise_investigation_2026-08-31.md §1-3/§5-1b)에서
# 딱 두 가지만 바꿨다:
#   1. --backbone uni2 -> uni2native (사용자 확인: uni2는 구형 1024px@1.0MPP 리사이즈 방식,
#      uni2native가 이 세션 전체에서 쓴 정식 256px@0.5MPP 파이프라인 — 오늘의 uni2official/
#      uni2native 혼동을 피하려고 명시적으로 재확인 후 진행)
#   2. --surv-loss cox(default) -> nll_surv — PORPOISE 원조 discretized-time NLL
#      (utils/losses.py::nll_surv_loss, train.py 2026-09-06 신규 추가). risk_head가 스칼라
#      대신 --nll-n-bins(기본 4)개 시간-구간별 hazard logit을 뱉도록 바뀐다
#      (models/vit_porpoise.py::ViT_PORPOISE surv_n_classes). 시간-구간 경계는 PORPOISE 원본과
#      달리(원본은 전체 코호트로 fit) 이 fold의 train split만으로 fit한다 — RNA 유전자 선정
#      leakage 전례(findings_backlog.md)를 피하려는 의도적 차이.
#
# 추가로 사용자 지시(2026-09-06)에 따라 3개 clinical/omics 옵션을 켰다(레퍼런스 no_aux
# 레시피엔 전부 꺼져 있었음) — --clinical-lr-mult 100(CLR100, clinical 브랜치 lr 100배),
# --use-cnv(pathway8 범위 CNV 8차원을 RNA 벡터에 concat), --clinical-mutation(PDAC 4대
# driver gene mutation status). 마지막 것(--clinical-mutation)은 지금까지 --M4에만 이식돼
# 있어서(2026-09-03) --PORPOISE에서 쓰려면 models/vit_porpoise.py::ViT_PORPOISE에
# use_mutation/mutation_stats 배선을 추가해야 했다(2026-09-06, ViT_M4를 상속하고
# _clinical_embed를 오버라이드 안 해서 이식 자체는 안전).
#
# **주의**: --rna-genes literature_1500_intersection은 PAAD에서 leaky한 것으로 확인된 유전자셋
# (findings_backlog.md, 폴드 간 ~60% 누출) — "RNA는 그대로 두라"는 사용자 지시로 일부러 안
# 바꿨다. 절대 수치(internal C)는 두 loss 버전 다 똑같이 낙관적으로 부풀려져 있지만, "loss
# 함수 차이"라는 이 실험의 관심사(두 버전의 상대 비교) 자체는 그 누출이 양쪽에 동일하게 걸려
# 있는 한 여전히 유효하다.
#
# 3seed x 5fold(15개 array) 대신 1seed x 10fold(10개 array, seed=84 — 이 세션에서 단일 시드
# 파일럿에 계속 써온 기본값)로 간소화 — "그냥 10폴드 한번에" 사용자 지시.
#
# 완료 후(10개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 10개): 정확한 모델 태그는
# 로그 초반 "Experiment"/체크포인트 경로나 `ls .logs/kfold_preds/tcga_PORPOISE*NLLSURV4*`로
# 직접 확인할 것(CLR/CNV/MUT 접미사가 여러 곳에서 조건부로 붙어 미리 문자열로 예측하면 틀리기
# 쉬움) — 그 태그로 아래 커맨드의 --model을 채운다:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model <위에서 확인한 태그> --seeds 84 --n-folds 10 --bootstrap 2000
#
# 제출: sbatch sbatch/porpoise_nll_surv_loss_10fold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=84
N_FOLDS=10
FOLD=$SLURM_ARRAY_TASK_ID

log=".logs/train_tcga_seed${SEED}_PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_NLLSURV4_kfold10_fold${FOLD}.log"

echo "=== PORPOISE nll_surv loss(uni2native,CLR100,CNV,MUT,STG+R) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PORPOISE --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 --attn-dispersion \
    --surv-loss nll_surv --nll-n-bins 4 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0906porpoise_nllsurv_10fold_array 2>&1 | tee "${log}"
echo "=== PORPOISE nll_surv loss(uni2native,CLR100,CNV,MUT,STG+R) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
