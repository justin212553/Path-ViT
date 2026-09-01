#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-diagnose-reliance
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_diagnose_reliance.log

# scripts/diagnose_porpoise_reliance.py를 GPU에서 돌린다(CNN+Nystromformer forward를
# 환자 수만큼 반복 — login node에서 돌리면 안 됨). seed84/fold0 파일럿(sbatch/
# porpoise_uni2_stg_r_pilot_seed84_fold0_hpc.sh, internal C=0.7063, dispersion+aux 둘 다 켬)과
# 3종 ablation(sbatch/porpoise_uni2_stg_r_ablation_seed84_fold0_hpc.sh, array 0-2)이 전부
# 끝나 있어야 한다 — models/checkpoint/에 PORPOISE 체크포인트 4개가 있어야 함.
#
# 체크포인트 파일명을 하드코딩하지 않고 glob으로 찾은 뒤, 파일명에 "_AUX_"/"_DISP_" 토큰이
# 있는지로 4개(pilot=둘 다 있음 / no_dispersion=AUX만 / no_aux=DISP만 / no_both=둘 다 없음)를
# 분류한다(정확한 tag 순서를 몰라도 되게, m4a_..._external_eval_hpc.sh와 같은 glob 관례).
#
# 제출: sbatch sbatch/porpoise_diagnose_reliance_hpc.sh
# (파일럿 + ablation array 3개가 전부 끝난 뒤에 제출할 것)

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mapfile -t ALL_CKPTS < <(ls models/checkpoint/survival_tcga_uni2_seed84_*PORPOISE*FOLD0OF5_best_porpoise.pt 2>/dev/null)
echo "찾은 PORPOISE 체크포인트 ${#ALL_CKPTS[@]}개:"
printf '  %s\n' "${ALL_CKPTS[@]}"

PILOT_CKPT=""
NO_DISP_CKPT=""
NO_AUX_CKPT=""
NO_BOTH_CKPT=""
for f in "${ALL_CKPTS[@]}"; do
  if [[ "$f" == *_AUX_* && "$f" == *_DISP_* ]]; then
    PILOT_CKPT="$f"
  elif [[ "$f" == *_AUX_* && "$f" != *_DISP_* ]]; then
    NO_DISP_CKPT="$f"
  elif [[ "$f" != *_AUX_* && "$f" == *_DISP_* ]]; then
    NO_AUX_CKPT="$f"
  else
    NO_BOTH_CKPT="$f"
  fi
done

CKPTS=() ; LABELS=() ; DISPS=()
if [ -n "$PILOT_CKPT" ]; then CKPTS+=("$PILOT_CKPT"); LABELS+=("full"); DISPS+=("1"); fi
if [ -n "$NO_DISP_CKPT" ]; then CKPTS+=("$NO_DISP_CKPT"); LABELS+=("no_dispersion"); DISPS+=("0"); fi
if [ -n "$NO_AUX_CKPT" ]; then CKPTS+=("$NO_AUX_CKPT"); LABELS+=("no_aux"); DISPS+=("1"); fi
if [ -n "$NO_BOTH_CKPT" ]; then CKPTS+=("$NO_BOTH_CKPT"); LABELS+=("no_dispersion_no_aux"); DISPS+=("0"); fi

if [ "${#CKPTS[@]}" -eq 0 ]; then
  echo "[에러] PORPOISE 체크포인트를 하나도 못 찾음 — 파일럿/ablation job이 먼저 끝나야 합니다."
  exit 1
fi
echo "분류 결과: ${LABELS[*]}"

echo "=== diagnose_porpoise_reliance Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.diagnose_porpoise_reliance \
    --ckpt "${CKPTS[@]}" \
    --labels "${LABELS[@]}" \
    --attn-dispersion "${DISPS[@]}" \
    --backbone uni2 --fold 0 --n-folds 5
echo "=== diagnose_porpoise_reliance Complete: $(date) ==="
