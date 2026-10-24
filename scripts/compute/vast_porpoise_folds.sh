#!/bin/bash
# 2026-10-23: PORPOISE의 특정 fold만 다시 돈다(vast_run_all.sh의 PORPOISE 구간과 같은 명령).
# QC 코호트 seed 42 fold 1·2가 빈 class로 0 나누기 오류(porpoise/utils/utils.py 수정) — 그 fold만 재실행.
#   setsid nohup bash scripts/compute/vast_porpoise_folds.sh n110 42:1 42:2 >> .logs/vast_porpoise_folds.log 2>&1 < /dev/null &
set -e
cd "$(dirname "$0")/../.."
V=$1; shift
ROOT_ABS="$(pwd)"
DATA_ROOT="${ROOT_ABS}/porpoise/data_root_true_resnet50"
if [ "$V" = n110 ]; then RES=results_n110_mmf_qc; SPL=5foldcv_n110qc; EXTRA=""
  TCSV=datasets_csv/tcga_paad_all_clean.csv.zip; CCSV=datasets_csv/cptac_paad_n110_external.csv.zip; TAG=PORPOISE_N110_QC_MMF
else RES=results_official_n110_mmf_qc; SPL=5foldcv_n110officialqc; EXTRA="--apply_mutsig"
  TCSV=datasets_csv_mutsig/tcga_paad_all_clean.csv.zip; CCSV=datasets_csv_mutsig/cptac_paad_external_cbio.csv.zip; TAG=PORPOISE_OFFICIAL_N110_QC_CBIO_MMF; fi
cd porpoise
for SF in "$@"; do
  SEED=${SF%%:*}; FOLD=${SF##*:}
  OUT_CSV="${ROOT_ABS}/.logs/external_preds/cptac_${TAG}_seed${SEED}_fold${FOLD}of5.csv"
  [ -f "$OUT_CSV" ] && { echo "PORPOISE ${V} seed=${SEED} fold=${FOLD} 이미 끝남"; continue; }
  LOG="${ROOT_ABS}/.logs/compute_jobs/porpoise_${V}_s${SEED}_f${FOLD}.log"
  python -u main.py --data_root_dir "${DATA_ROOT}" --which_splits "$SPL" --split_dir "tcga_paad_seed${SEED}" \
    --mode pathomic --model_type porpoise_mmf --bag_loss nll_surv --reg_type pathomic $EXTRA \
    --fusion bilinear --gate_path --gate_omic --skip --dropinput 0.10 \
    --k_start "${FOLD}" --k_end "$((FOLD + 1))" --results_dir "./${RES}" --seed "${SEED}" --overwrite > "$LOG" 2>&1 \
    || { echo "!! PORPOISE ${V} train seed=${SEED} fold=${FOLD} 실패 — $LOG"; continue; }
  python -u eval_external.py --seed "${SEED}" --fold "${FOLD}" --results-dir "$RES" --which-splits "$SPL" \
    --tcga-csv "$TCSV" --tcga-split-dir "splits/${SPL}/tcga_paad_seed${SEED}" --cptac-csv "$CCSV" \
    --tcga-data-root "${DATA_ROOT}" --cptac-data-root "${ROOT_ABS}/data/porpoise_style_features/cptac" \
    --out-csv "$OUT_CSV" >> "$LOG" 2>&1 && echo "PORPOISE ${V} seed=${SEED} fold=${FOLD} 끝" \
    || echo "!! PORPOISE ${V} eval seed=${SEED} fold=${FOLD} 실패 — $LOG"
done
