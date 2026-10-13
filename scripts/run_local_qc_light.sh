#!/bin/bash
# 2026-10-12(D3, 데이터 QC 재실행): WSI 없는 가벼운 모델(M5~M7 계열)을 로컬에서 5 seeds x 5 folds로 돌린다.
# HPC의 sbatch/fs_paad_array.sh, sbatch/fs_brca_array.sh와 같은 인자에 QC 옵션만 더한다.
#   PAAD: --exclude-nonpositive-os (TCGA-PAAD 1명, CPTAC-PDAC 1명 제외, 태그 _QC)
#   BRCA: --external-tss instcvqc  (21명 제외, 태그 _INSTCVQC)
# 두 경우 모두 fold 배정은 제외 전 코호트로 계산한 뒤 해당 환자만 빼므로 나머지 배정은 기존과 같다.
#
# 사용법 (Git Bash, 저장소 루트에서):
#   bash scripts/run_local_qc_light.sh                 # PAAD 5개 + BRCA 3개 모델 전부
#   bash scripts/run_local_qc_light.sh paad M5 M6      # 일부만
# 끝난 task는 .logs/local_qc/*.done으로 표시되어 다시 실행하면 건너뛴다(중간에 꺼져도 이어서 돌림).
# 로그: .logs/local_qc/{cohort}_{model}_seed{s}_fold{f}.log, 진행 요약: .logs/local_qc/progress.txt

cd "$(dirname "$0")/.."
PY=${PY:-"C:/Users/wonse/anaconda3/envs/PathViT-ray/python.exe"}
export WANDB_MODE=offline
export PYTHONIOENCODING=utf-8

WHICH=${1:-all}
shift || true
ONLY=("$@")

SEEDS=(84 126 42 168 210)
N_FOLDS=5
OUT=.logs/local_qc
mkdir -p "$OUT"

PAAD_MODELS=(M5 M6 M7 M6NOCNV M7NOCNV)
BRCA_MODELS=(M5 M6 M7)

want() {
  [ ${#ONLY[@]} -eq 0 ] && return 0
  for m in "${ONLY[@]}"; do [ "$m" = "$1" ] && return 0; done
  return 1
}

paad_cmd() {  # $1=model $2=seed $3=fold
  local RNA="--rna-genes pdac_consistency_1500"
  local CLIN="--clinical-staging --clinical-margin --clinical-lr-mult 100"
  local COMMON="--dataset tcga --external --seed $2 --fold $3 --n-folds ${N_FOLDS} \
    --fold-safe --restrict-cohort-file data/cohort_n110.txt --exclude-nonpositive-os \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 --group-ts 1012qc_paad_$1"
  case "$1" in
    M5)      echo "train_light.py --M5 --raw-linear --clinical-margin --clinical-staging $COMMON" ;;
    M6)      echo "train_light.py --M6 $RNA --use-cnv $COMMON" ;;
    M7)      echo "train_light.py --M7 $RNA --use-cnv --combine-mode cox_add $CLIN --clinical-mutation $COMMON" ;;
    M6NOCNV) echo "train_light.py --M6 $RNA $COMMON" ;;
    M7NOCNV) echo "train_light.py --M7 $RNA --combine-mode cox_add $CLIN --clinical-mutation $COMMON" ;;
  esac
}

brca_cmd() {
  local COMMON="--seed $2 --fold $3 --n-folds ${N_FOLDS} --external-tss instcvqc --fold-safe \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 --group-ts 1012qc_brca_$1"
  case "$1" in
    M5) echo "-m scripts.train_brca_m5 --clinical-staging $COMMON" ;;
    M6) echo "-m scripts.train_brca_m6 --gene-selection consistency $COMMON" ;;
    M7) echo "-m scripts.train_brca_m7 --gene-selection consistency --clinical-staging --clinical-lr-mult 100 $COMMON" ;;
  esac
}

run_one() {  # $1=cohort $2=model $3=seed $4=fold
  local name="$1_$2_seed$3_fold$4"
  [ -f "$OUT/$name.done" ] && return 0
  local cmd
  if [ "$1" = paad ]; then cmd=$(paad_cmd "$2" "$3" "$4"); else cmd=$(brca_cmd "$2" "$3" "$4"); fi
  echo "$(date '+%F %T') START $name" | tee -a "$OUT/progress.txt"
  echo "$PY -u $cmd" > "$OUT/$name.log"
  if $PY -u $cmd >> "$OUT/$name.log" 2>&1; then
    touch "$OUT/$name.done"
    echo "$(date '+%F %T') DONE  $name" | tee -a "$OUT/progress.txt"
  else
    echo "$(date '+%F %T') FAIL  $name (로그: $OUT/$name.log)" | tee -a "$OUT/progress.txt"
  fi
}

run_cohort() {  # $1=cohort, 나머지=모델들
  local cohort=$1; shift
  for m in "$@"; do
    want "$m" || continue
    for s in "${SEEDS[@]}"; do
      for f in $(seq 0 $((N_FOLDS - 1))); do run_one "$cohort" "$m" "$s" "$f"; done
    done
  done
}

[ "$WHICH" = all ] || [ "$WHICH" = paad ] && run_cohort paad "${PAAD_MODELS[@]}"
[ "$WHICH" = all ] || [ "$WHICH" = brca ] && run_cohort brca "${BRCA_MODELS[@]}"
n_done=$(ls "$OUT"/*.done 2>/dev/null | wc -l)
n_fail=$(grep -c " FAIL " "$OUT/progress.txt" 2>/dev/null || echo 0)
echo "$(date '+%F %T') 종료: 완료 표시 ${n_done}개, 이번까지 누적 FAIL 줄 ${n_fail}개" | tee -a "$OUT/progress.txt"
