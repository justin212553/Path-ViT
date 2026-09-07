#!/bin/bash
#SBATCH --job-name=PVT-porpoise-mmf
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/run_porpoise_official_paad_mmf_%j.log

# 2026-09-06: 지금까지 재현해온 porpoise_amil(WSI-only)은 애초에 논문 헤드라인 수치(PAAD
# c-index 0.653)를 낼 수 있는 실험이 아니었다 — 논문 원문(PMC10397370, Results) 직접 확인:
# "PAAD ... c-Index of 0.653 ... using MMF(멀티모달), compared to 0.580 ... using AMIL and
# 0.593 ... using SNN" — 0.653은 WSI+genomics를 결합한 MMF(PorpoiseMMF, model_porpoise.py)
# 전용 수치이고, AMIL 단독으로는 논문 자신도 0.580이 천장이다. 이번엔 실제로 그 MMF를 돌린다.
#
# genomics 표현 방식(1번=flat 벡터 전체 vs 2번=6-family gene signature 분리, apply_sig=True)
# 질문에 대한 답: 공식 docs/Commands.md의 어떤 커맨드에도 --apply_sig가 등장하지 않는다 —
# 즉 논문 헤드라인 수치는 전부 1번(flat 벡터, apply_sig=False 기본값) 기준이다. 2번 모드는
# 애초에 저희 CSV(RNA만 있고 _mut/_cnv 없음, signatures.csv도 없음)로는 쓸 수도 없었지만,
# 그거와 별개로 진짜 논문 수치 자체가 2번을 안 썼다 — 추가 파일 준비 없이 지금 CSV 그대로
# 진행 가능.
#
# **중요 함정(코드 직접 대조로 발견)**: PorpoiseMMF 클래스 자체의 기본값은
# fusion='bilinear', gate_path=1, gate_omic=1, skip=True, dropinput=0.10 인데, main.py CLI
# 인자 기본값은 --fusion(문자열 'None'->None 변환), --gate_path/--gate_omic/--skip(모두
# store_true, 기본 False), --dropinput(기본 0.0)으로 전부 다르다 — CLI 인자가 클래스 기본값을
# 덮어쓰므로, 아래 플래그들을 명시적으로 안 주면 진짜 PORPOISE 아키텍처(gated bilinear fusion)가
# 아니라 genomics 브랜치 자체가 아예 안 만들어진(fusion=None) 반쪽 모델이 되어 forward()에서
# 크래시하거나 다른 구성이 된다. 그래서 아래 커맨드에 전부 명시적으로 박아 넣는다.
#
# 선행 조건: sbatch/extract_porpoise_style_features_hpc.sh(uni2native 타일 재사용 버전) 완료
# 확인됨(data/porpoise_style_features/tcga/pt_files/*.pt).
#
# 2026-09-06 수정: 이전 실행(TCGA 성분추출이 아직 덜 끝난 상태, 152->124케이스로 줄어든 CSV로
# 돌아간 버전 — 폐기하기로 함)이 같은 exp_code로 이미 summary_latest.csv를 남겨놔서, main.py가
# "Exp Code <...> already exists! Exiting script."로 바로 종료해버렸다(porpoise/main.py:242-243,
# args.overwrite 기본 False). rm 없이 --overwrite만 추가해서 그 이전 결과를 덮어쓰게 한다
# (per-fold pkl 재생성 스킵 로직도 같은 플래그로 풀림, porpoise/main.py:54).
#
# 완료 후: porpoise/results_true_resnet50_mmf/5foldcv/.../summary_latest.csv 5-fold val
# c-index. 논문 MMF 0.653, 지금까지 나온 AMIL 재현치(run_porpoise_official_paad_amil_hpc.sh)와
# 비교.
#
# 제출: sbatch sbatch/run_porpoise_official_paad_mmf_hpc.sh

cd /pub/wonseukl/Path-ViT/porpoise

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

DATA_ROOT="/pub/wonseukl/Path-ViT/porpoise/data_root_true_resnet50"
PT_FILES_DIR="/pub/wonseukl/Path-ViT/data/porpoise_style_features/tcga/pt_files"
mkdir -p "${DATA_ROOT}/tcga_paad_20x_features"
ln -sfn "${PT_FILES_DIR}" "${DATA_ROOT}/tcga_paad_20x_features/pt_files"

# run_porpoise_official_paad_amil_hpc.sh와 동일한 안전장치 — uni2native 리타일링에서 슬라이드
# 하나가 조용히 실패했을 가능성 대비, 실제 존재하는 .pt만 남기도록 CSV 필터링(이미 대부분
# 377개 다 있을 것으로 예상, no-op에 가까울 것).
echo "=== 슬라이드 존재 여부로 CSV 필터링: $(date) ==="
python -u filter_available_slides.py --pt-files-dir "${PT_FILES_DIR}"

echo "=== PORPOISE 공식 코드, 진짜 ResNet50(1024d) feature, MMF(WSI+genomics, gated bilinear fusion) Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u main.py \
    --data_root_dir "${DATA_ROOT}" \
    --which_splits 5foldcv --split_dir tcga_paad \
    --mode pathomic --model_type porpoise_mmf --bag_loss nll_surv --reg_type pathomic \
    --fusion bilinear --gate_path --gate_omic --skip --dropinput 0.10 \
    --results_dir ./results_true_resnet50_mmf --seed 1 --overwrite
echo "=== PORPOISE 공식 코드, 진짜 ResNet50(1024d) feature, MMF Complete: $(date) ==="
