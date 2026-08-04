#!/bin/bash
#SBATCH --job-name=PVT-PMA-tileaug-disp
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_ex_ss_aux_tileaugment_dispersion_ext.log

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 2026-08-04: 컴퓨트 노드가 외부 인터넷이 막혀/느려서 매 epoch wandb.log()가 동기화를 기다리며
# 멈추는 것으로 의심되는 정지 현상(GPU util이 짧게 튀었다가 15~20분씩 0으로 죽는 패턴) 대응 —
# 로그를 로컬(.wandb/)에만 쌓고 클라우드 동기화를 아예 안 한다. 결과 확인하려면 로그인 노드
# 등 인터넷 되는 곳에서 `wandb sync .wandb/<run-dir>` (또는 `wandb sync --sync-all`)로 나중에
# 올린다.
export WANDB_MODE=offline

# scripts/_pma_ex_ss_aux_tileaugment_ext.sh(2026-07-21, dispersion 없음) + --attn-dispersion 추가.
# 그 스크립트는 이미 결과가 나온 실험 기록이라 그대로 두고, DISP를 더한 버전을 새로 만든다 —
# train_m1/m2/m3_hpc.sh가 이미 쓰고 있는 "SS+AUG+DISP(+AUX)" 레시피를 PMA(clinical 포함, M4
# 슬롯)에도 맞춘 것.
#
# --tile-decode-workers 8: models/vit_m1.py::_patch_tokens의 타일 디코딩+증강 스레드풀 크기 +
# data/patch_utils.py::build_tile_cache의 프리로드 스레드 수(2026-08-04부터 여기도 병렬화됨).
# 이 작업은 forward() 안에서 도는 별도 스레드풀이라 DataLoader num_workers와는 무관하다 — 위
# --cpus-per-task=8을 실제로 다 쓰려면 이 값도 맞춰야 한다(2026-08-03, 그 전까지 4로 하드코딩).
#
# --cache-val-tiles: 2026-08-04 — evaluate()가 train_eval/val 둘 다 tile_cache 없이 매 epoch
# 디스크에서 새로 디코딩하고 있던 게 이 클러스터(hpc3-gpu-24-06)에서 epoch 1도 못 넘기는 진짜
# 병목이었다(py-spy로 MainThread가 PIL Image.open() 안에서 멈춘 걸 직접 확인). train_eval은
# train_ds의 tile_cache를 재사용하도록 고쳤고(추가 메모리 0), val은 이 플래그로 별도 캐시를
# 만든다 — 128GB 예산이면 val(31명) 추가 캐싱 여유 충분(로컬 32GB에서는 끄는 게 안전).
#
# 제출: sbatch scripts/_pma_ex_ss_aux_tileaugment_dispersion_ext.sh

LogDir=".logs"
Seeds=(42 84 126)
GroupTs="0803pma_tileaugment_disp_ext"

for seed in "${Seeds[@]}"; do
    echo "=== PMA_EX_SS_AUX_AUG_DISP seed=${seed} Start: $(date) ==="
    log="${LogDir}/train_tcga_seed${seed}_PMA_EX_SS_AUX_AUG_DISP_ext.log"
    python -u ./train.py --dataset tcga --seed "${seed}" --PMA --rna-genes literature_1500 \
        --patch-keep-frac 0.8 --rna-aux-weight 1.0 --image --tile-augment --attn-dispersion \
        --tile-decode-workers 8 --cache-val-tiles --cache-external-tiles \
        --external --group-ts "${GroupTs}" 2>&1 | tee "${log}"
    echo "=== PMA_EX_SS_AUX_AUG_DISP seed=${seed} Complete: $(date) ==="
done

echo "=== ALL PMA_EX_SS_AUX_AUG_DISP EXTERNAL RUNS COMPLETE: $(date) ==="
