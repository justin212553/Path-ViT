#!/bin/bash
#SBATCH --job-name=PVT-MCAT-diagnose-pilot
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/mcat_diagnose_pilot_seed84_fold0.log

# mcat_uni2_coxadd_stg_r_pilot_seed84_fold0_hpc.sh 파일럿 결과가 internal C=0.46(<chance)
# + HR=0.712(<1, 방향 반전)로 나와, PMA/PM4/M4A seed84 파일럿(0.68/0.66/0.61)보다 확연히
# 낮았다 — init-seed 노이즈(±0.04~0.05)로는 설명 안 되는 격차. train.py --eval-internal-ckpt에
# 2026-08-31 추가한 MCAT 전용 진단(train.py::args.eval_internal_ckpt 블록, hasattr(model,
# "gene_group_encoder")로 게이팅)을 재학습 없이 그 checkpoint에 대해 돌려, pathway token이
# 환자별로 실제 갈라지는지(cosine similarity/std) + co-attention이 findings_backlog.md
# 최상위 발견(4-view co-attention entropy 붕괴)과 같은 패턴으로 uniform에 수렴했는지 확인한다.
#
# 로컬 1-epoch(미완성) 체크포인트로 미리 돌려본 참고용 결과: pathway token은 collapse 안 됨
# (cosine sim ~0.0004, std 0.56)인데 co-attention entropy는 ~0.9999(uniform) — 실제 30-epoch
# 체크포인트에서도 같은 패턴이면 "query 개수 부족"이 원인이 아니었다는 뜻이 된다.
#
# 필요 조건: mcat_uni2_coxadd_stg_r_pilot_seed84_fold0_hpc.sh가 먼저 완료돼 있어야 함
# (models/checkpoint/에 *MCAT* 체크포인트가 있어야 함).
#
# 제출: sbatch sbatch/mcat_diagnose_pilot_seed84_fold0_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

mapfile -t MATCHES < <(ls models/checkpoint/survival_tcga_uni2_seed84_*MCAT*FOLD0OF5_best_mcat.pt 2>/dev/null)
if [ "${#MATCHES[@]}" -eq 0 ]; then
  echo "[에러] MCAT 파일럿 checkpoint를 못 찾음 — mcat_uni2_coxadd_stg_r_pilot_seed84_fold0_hpc.sh가 먼저 끝나야 합니다."
  exit 1
fi
if [ "${#MATCHES[@]}" -gt 1 ]; then
  echo "[경고] checkpoint가 ${#MATCHES[@]}개 매칭됨 — 첫 번째만 사용: ${MATCHES[0]}"
fi
CKPT="${MATCHES[0]}"

echo "=== MCAT 진단(seed84 fold0) ckpt=${CKPT} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --MCAT --rna-genes literature_1500_intersection --dataset tcga --seed 84 \
    --backbone uni2 \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --fold 0 --n-folds 5 --group-ts 0831mcat_uni2_coxadd_stg_r_pilot \
    --eval-internal-ckpt "${CKPT}"
echo "=== MCAT 진단(seed84 fold0) Complete: $(date) ==="
