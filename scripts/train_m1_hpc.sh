#!/bin/bash
#SBATCH --job-name=PVT-M1-tcga
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/train_m1.log

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 2026-08-04: 이 클러스터 컴퓨트 노드가 외부 인터넷이 막혀/느려서 매 epoch wandb.log()가 동기화를
# 기다리며 정지하는 현상을 PMA_EX_SS_AUX_AUG_DISP에서 확인(py-spy로 원인 특정) — 로그를
# 로컬(.wandb/)에만 쌓고 클라우드 동기화를 안 한다. 나중에 인터넷 되는 로그인 노드에서
# `wandb sync .wandb/<run-dir>`로 올린다.
export WANDB_MODE=offline

# 2026-07-30: train_multi.py(M1/M2/M3/PMA 공유 backbone 학습)는 모델을 여러 개 한 프로세스에서
# 순서대로 생성하다 보니 PyTorch 전역 RNG 스트림이 이어져서, "같은 --seed"를 줘도 어떤 모델과
# 같이 도느냐에 따라 실제 초기 가중치가 달라지는 문제가 있었다(재시딩 브래킷으로 완화했지만
# 완전히 검증되진 않음). 이 스크립트는 M1 하나만 train.py로 단독 학습시켜 그 문제 자체를
# 원천 차단한다 — 대신 backbone 공유로 얻는 속도 이득은 없다(다만 backbone forward가 비용의
# 99%라, M1 하나만 도는 epoch당 시간은 4개 모델 같이 돌 때와 비슷할 것으로 추정 — 실측 필요).
# SS(patch dropout)+AUG(실시간 augmentation)+DISP(attention dispersion) 레시피 — AUX(RNA
# 보조과제)는 RNA가 없는 M1엔 대응 항목이 없어 제외.
#
# --tile-decode-workers 8: 타일 디코딩+증강 스레드풀 크기(위 --cpus-per-task=8과 맞춤, 그동안
# 4로 하드코딩되어 있었음) — build_tile_cache 프리로드에도 동일하게 적용된다.
# --cache-val-tiles: evaluate()가 train_eval/val을 매 epoch 디스크에서 새로 디코딩하던 게 이
# 클러스터에서 epoch 1도 못 넘기는 병목이었다(train_eval은 train_ds의 캐시를 재사용하도록 이미
# 고쳤고, val은 이 플래그로 별도 캐시를 만든다 — 128GB 예산이면 여유 충분).
python -u ./train.py --M1 --dataset tcga --external --seed 84 \
    --tile-augment --image --patch-keep-frac 0.8 --attn-dispersion \
    --tile-decode-workers 8 --cache-val-tiles --cache-external-tiles "$@"
