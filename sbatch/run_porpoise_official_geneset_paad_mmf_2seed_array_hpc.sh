#!/bin/bash
#SBATCH --job-name=PVT-porpoise-mmf-officialgene-2seed
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/run_porpoise_official_geneset_paad_mmf_2seed_array_%a.log

# 2026-09-14: run_porpoise_official_paad_mmf_2seed_array_hpc.sh(results_true_resnet50_mmf)가
# 사실은 PORPOISE의 공식 유전자 세트를 쓰고 있지 않았다는 게 밝혀져 새로 만든 스크립트다.
#
# 문제 원인(직접 코드 대조로 확인, 2026-09-14): main.py의 csv_path는
# './%s/%s_all_clean.csv.zip' % (args.dataset_path, study)이고, args.dataset_path는
# utils/utils.py::get_custom_exp_code()가 --apply_mutsig 플래그 여부로 'datasets_csv' 또는
# 'datasets_csv_mutsig'를 고른다(기본은 --apply_mutsig 없이 'datasets_csv'). 그런데 공식
# PORPOISE repo(mahmoodlab/PORPOISE)엔애초에 PAAD용 CSV가 'datasets_csv/' 폴더엔 아예 없고
# 'datasets_csv_mutsig/tcga_paad_all_clean.csv.zip'(166환자, 1553 RNA-seq유전자[symbol+
# _rnaseq]+82 CNV유전자[symbol+_cnv]+14 mutation유전자[symbol+_mut], 700KB)에만 있다.
# 이전 스크립트는 --apply_mutsig를 안 줬기 때문에 존재하지도 않는 'datasets_csv/tcga_paad_
# all_clean.csv.zip'을 다른 무언가가 대신 채워넣은 상태로 돌아갔다 — 실제로 그 자리에 있던
# 파일은 이 프로젝트 자체 GDC 파이프라인이 만든 57MB짜리 19,962개 전체 protein-coding
# 유전자(Ensembl ID, CNV/mutation 없음) CSV였다(누가 언제 왜 그 경로에 놨는지는 불명 —
# 아마 --apply_mutsig를 안 준 상태에서 main.py가 죽지 않게 하려고 예전에 급조된 placeholder로
# 추정). apply_sig(coattn 모드 전용, mcat류)와는 별개 플래그라 --mode pathomic(porpoise_mmf)엔
# 영향 없음 — main.py:69 `if 'omic' in args.mode`가 'pathomic'에도 매칭되어 그냥
# genomic_features.shape[1]을 그대로 omic_input_dim으로 쓰기 때문에, 컬럼이 뭐든 조용히
# 돌아간 것.
#
# 고친 내용: --apply_mutsig 추가 + 공식 파일을 porpoise/datasets_csv_mutsig/tcga_paad_
# all_clean.csv.zip에 그대로 배치(mahmoodlab/PORPOISE에서 직접 가져옴, md5 다름 확인 완료) +
# --results_dir을 results_official_geneset_mmf로 분리(기존 results_true_resnet50_mmf는
# 잘못된 유전자셋으로 학습된 것이니 보존은 하되 더 이상 "공식 재현"으로 쓰지 않는다).
#
# 나머지 설정(fusion=bilinear, gate_path/gate_omic/skip, dropinput=0.10, seed 84/126)은
# 기존 스크립트와 동일 — 공식 PORPOISE 아키텍처/학습 레시피 자체는 원래도 맞게 돌아가고
# 있었고, 문제는 오직 genomic input CSV 하나였다.
#
# 완료 후(두 seed 다 s_0~s_4_checkpoint.pt 5개씩 있는지 확인):
#   python scripts/pool_porpoise_official_kfold.py --results-dir porpoise/results_official_geneset_mmf \
#       --seeds 84,126 --bootstrap 2000
#
# external(CPTAC) 평가는 별도 스크립트 필요 — 공식 유전자 세트(1553 RNA-seq + 82 CNV + 14
# mutation, 전부 gene symbol 기준)로 CPTAC용 CSV를 새로 만들어야 하는데, mutation 14개 중
# 우리가 이미 gene-level로 갖고 있는 건 KRAS/TP53/SMAD4/CDKN2A 4개뿐이고(data/mutations_cptac.csv),
# 나머지 10개(ARID1A/CSMD2/GNAS/MUC16/OBSCN/PCDH15/RAS/RNF43/RYR1/TTN)는 raw mutation call이
# 없다. CNV 82개는 data/cnv_cptac.csv(163유전자, gene-level, Ensembl ID)에서 커버될 가능성이
# 높지만 심볼 매핑 후 실제 교집합 확인 필요. 이 gap을 어떻게 처리할지(누락 10개를 raw MAF에서
# 새로 받아올지, unmutated로 근사할지, 아니면 이 엄격한 공식-재현판은 external 없이 internal만
# 보고할지) 사용자 결정 대기 중 — 결정되면 scripts/prepare_porpoise_cptac_official_geneset.py
# 작성 예정.
#
# 제출: sbatch sbatch/run_porpoise_official_geneset_paad_mmf_2seed_array_hpc.sh

cd /pub/wonseukl/Path-ViT/porpoise

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

DATA_ROOT="/pub/wonseukl/Path-ViT/porpoise/data_root_true_resnet50"
PT_FILES_DIR="/pub/wonseukl/Path-ViT/data/porpoise_style_features/tcga/pt_files"
mkdir -p "${DATA_ROOT}/tcga_paad_20x_features"
ln -sfn "${PT_FILES_DIR}" "${DATA_ROOT}/tcga_paad_20x_features/pt_files"

SEEDS=(84 126)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== 슬라이드 존재 여부로 CSV 필터링: $(date) ==="
python -u filter_available_slides.py --pt-files-dir "${PT_FILES_DIR}"

echo "=== PORPOISE 공식 코드, 공식 유전자 세트(datasets_csv_mutsig), 진짜 ResNet50(1024d) feature, MMF, seed=${SEED} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u main.py \
    --data_root_dir "${DATA_ROOT}" \
    --which_splits 5foldcv --split_dir tcga_paad \
    --mode pathomic --model_type porpoise_mmf --bag_loss nll_surv --reg_type pathomic \
    --apply_mutsig \
    --fusion bilinear --gate_path --gate_omic --skip --dropinput 0.10 \
    --results_dir ./results_official_geneset_mmf --seed "${SEED}" --overwrite
echo "=== PORPOISE 공식 코드, 공식 유전자 세트, 진짜 ResNet50, MMF, seed=${SEED} Complete: $(date) ==="
