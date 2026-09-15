#!/bin/bash
#SBATCH --job-name=PVT-mmp-brca-surv
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/run_mmp_brca_survival_array_%a.log

# 2026-09-14: MMP(Song et al., ICML 2024, mahmoodlab/MMP) BRCA 재현, 2단계(본 학습).
# 원본 mmp.sh(src/scripts/survival/mmp.sh)를 그대로 옮기되 다음을 이 연구 프로토콜에 맞춘다:
#   - split_dir: 공식 TCGA_BRCA_overall_survival_k=* 대신 이 연구 자체 TCGA_BRCA_k{0..4}_
#     seed{84,126}(scripts/prepare_mmp_brca_data.py 산출물) — 우리 M1~M7과 정확히 같은 case가
#     같은 fold에 들어간다. 폴더명은 여전히 "TCGA_BRCA_..."라 main_survival.py가 유도하는
#     cancer_type이 'BRCA'로 나오고, 공식 RNA 폴더(data_csvs/rna/hallmarks/BRCA/)를 그대로
#     찾아간다.
#   - target_col: 원본 스크립트는 dss_survival_days를 쓰지만, main_survival.py의 실제 기본값
#     자체가 os_survival_days/os_censorship라(코드 직접 확인, 2026-09-14) 이 연구가 쓰는
#     OS(overall survival)와 그대로 맞아떨어진다 — target_col을 os_survival_days로 지정.
#   - RNA: 이 연구 자체 패널로 바꾸는 초안을 한 차례 만들었다가(scripts/prepare_mmp_brca_
#     data_ownrna_bak.py에 백업) 뒤집었다 — MMP repo에 BRCA 공식 RNA 데이터(4,241유전자
#     Hallmark 패널, data_csvs/rna/hallmarks/BRCA/rna_clean.csv)가 이미 있고 우리 코호트
#     1095명 중 1093명을 그대로 커버해서(직접 대조 확인), PORPOISE 때와 같은 원칙으로
#     원저자 공식 데이터를 그대로 쓴다. WSI만 이 연구 자체 UNI v1 feature로 대체하는데,
#     이건 원저자와 동일한 백본(uni_mass100k)이라 바꿔치기가 아니다.
#   - seed: 원본은 고정 seed=1, 이 연구는 84/126 두 개를 각각 돈다.
#
# 선행 조건: run_mmp_brca_prototype_array_hpc.sh(같은 array index)가 먼저 끝나 있어야 함
#   (해당 fold의 prototypes_*.pkl이 splits/{split_dir}/prototypes/ 에 있어야 함).
#
# external(A2/AR/E9) 평가는 별도 스크립트 없이 이 학습 실행 안에서 끝난다 — 코드 직접 확인
# 결과(main_survival.py::build_datasets, trainer.py::train) csv_splits.keys()를 그대로
# 순회하며 각 split마다 validate_survival()을 돌리는 구조라, --split_names에 "external"만
# 추가하면 된다. omics scaler와 prototype 둘 다 train에서만 fit되고 나머지 split엔 적용만
# 되므로(build_datasets: "if k=='train': scaler=...", --load_proto는 이미 fold별 train 전용
# prototype을 씀) leakage 없이 그대로 맞다. 결과는 {results_dir}/summary.csv 등 MMP 자체
# 요약 파일에 split별로 이미 나뉘어 저장된다(체크포인트 재로딩 불필요).
#
# 완료 후 checkpoint(참고용, 재평가 안 해도 됨): early_stopping=0이라 모든 fold가
# {results_dir}/s_checkpoint.pth에 마지막 epoch의 순수 model.state_dict()를 그대로 저장한다
# (trainer.py:139, 'model' 같은 wrapper key 없음 — early_stopping=1일 때만 {'model':...}
# 형태로 감싸는 별도 코드 경로가 따로 있으니 혼동 주의).
#
# 제출: sbatch sbatch/run_mmp_brca_survival_array_hpc.sh

MMP_ROOT="/pub/wonseukl/Path-ViT/mmp/src"
MMP_DATAROOT="/pub/wonseukl/Path-ViT/data/mmp_brca_dataroot"

cd "${MMP_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

SEEDS=(84 126)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

SPLIT_DIR="survival/TCGA_BRCA_k${FOLD}_seed${SEED}"
FEAT='extracted-vit_large_patch16_224.dinov2.uni_mass100k'
FEAT_DIR="${MMP_DATAROOT}/extracted_mag20x_patch256_fp/${FEAT}/feats_pt"
EXP_CODE="BRCA_OWNPROTOCOL_officialRNA_seed${SEED}_k${FOLD}::PANTHER_default::${FEAT}"
RESULTS_DIR="results/${EXP_CODE}"
PROTO_PATH="splits/${SPLIT_DIR}/prototypes/prototypes_c16_${FEAT}_faiss_num_1.0e+05.pkl"

echo "=== MMP BRCA survival train seed=${SEED} fold=${FOLD} Start: $(date) (job ${SLURM_JOB_ID}) ==="
python -u -m training.main_survival \
    --data_source "${FEAT_DIR}" \
    --results_dir "${RESULTS_DIR}" \
    --split_dir "${SPLIT_DIR}" \
    --split_names train,val,test,external \
    --task BRCA_OWNPROTOCOL_officialRNA_survival \
    --target_col os_survival_days \
    --model_histo_type PANTHER \
    --model_histo_config PANTHER_default \
    --n_fc_layers 0 \
    --in_dim 1024 \
    --opt adamW \
    --lr 0.0001 \
    --lr_scheduler cosine \
    --accum_steps 1 \
    --wd 0.00001 \
    --warmup_epochs 1 \
    --max_epochs 50 \
    --train_bag_size -1 \
    --batch_size 64 \
    --seed "${SEED}" \
    --num_workers 8 \
    --em_iter 1 \
    --tau 1.0 \
    --n_proto 16 \
    --out_type allcat \
    --loss_fn cox \
    --nll_alpha 0.5 \
    --n_label_bins 4 \
    --early_stopping 0 \
    --ot_eps 1 \
    --fix_proto \
    --num_coattn_layers 1 \
    --model_mm_type coattn \
    --append_embed random \
    --histo_agg mean \
    --net_indiv \
    --load_proto \
    --proto_path "${PROTO_PATH}"
echo "=== MMP BRCA survival train seed=${SEED} fold=${FOLD} Complete: $(date) ==="
