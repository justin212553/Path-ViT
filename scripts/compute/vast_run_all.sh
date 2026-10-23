#!/bin/bash
# 2026-10-22: Vast.ai에서 BRCA QC 재실행(WSI 모델 6개) + MMP(instcvqc) + PORPOISE QC 2종을 돌린다.
# 끊겨도 다시 실행하면 끝난 것은 건너뛴다(.logs/compute_done, 결과 pkl 존재 여부).
# 세션이 끊겨도 계속 돌게 tmux 안에서:  tmux new -s qc  ->  bash scripts/compute/vast_run_all.sh 2>&1 | tee .logs/vast_run_all.log
# 워커 수: WORKERS=3 (기본). RAM이 64GB 미만이면 2로.
# 결과 회수(로컬에서):  scp -P <port> -r root@<host>:/workspace/Path-ViT/.logs/{kfold_preds,external_preds,compute_done} .vast_out/.logs/
#                       python -m scripts.compute.collect_outputs .vast_out
set -e
cd "$(dirname "$0")/../.."
WORKERS=${WORKERS:-3}
SEEDS=(84 126 42 168 210)
mkdir -p .logs/compute_jobs

# ---- 1) BRCA WSI 모델 6개 (옛 분석 번호. 원고 번호 M1 M5 M4 M7 M4-SA M7-SA) ----
python -m scripts.compute.qc_jobs --cohort brca --models M1 M2 M3 M4 M3SA M4SA --workers "$WORKERS"

# ---- 2) MMP (instcvqc): fold마다 prototype -> survival ----
FEAT='extracted-vit_large_patch16_224.dinov2.uni_mass100k'
MMP_DATAROOT="$(pwd)/data/mmp_brca_dataroot"
pushd mmp/src > /dev/null
for SEED in "${SEEDS[@]}"; do for FOLD in 0 1 2 3 4; do
  SPLIT_DIR="survival/TCGA_BRCA_k${FOLD}_seed${SEED}_instcvqc"
  PROTO="splits/${SPLIT_DIR}/prototypes/prototypes_c16_${FEAT}_faiss_num_1.0e+05.pkl"
  EXP="BRCA_INSTCVQC_officialRNA_seed${SEED}_k${FOLD}::PANTHER_default::${FEAT}"
  if [ -n "$(find "results/${EXP}" -name test_results.pkl 2>/dev/null | head -1)" ]; then
    echo "MMP seed=${SEED} fold=${FOLD} 이미 끝남"; continue; fi
  # set -e 아래에서 prototype 실패가 스크립트 전체를 멈추지 않도록(2026-10-23: transformers 누락으로 멈춤)
  if [ ! -f "$PROTO" ] && ! python -u -m training.main_prototype --mode faiss \
      --data_source "${MMP_DATAROOT}/extracted_mag20x_patch256_fp/${FEAT}/feats_pt" \
      --split_dir "${SPLIT_DIR}" --split_names train --in_dim 1024 --n_proto_patches 100000 \
      --n_proto 16 --n_init 3 --seed "${SEED}" > "../../.logs/compute_jobs/mmp_proto_s${SEED}_f${FOLD}.log" 2>&1; then
    echo "!! MMP proto seed=${SEED} fold=${FOLD} 실패 — .logs/compute_jobs/mmp_proto_s${SEED}_f${FOLD}.log"; continue
  fi
  python -u -m training.main_survival \
      --data_source "${MMP_DATAROOT}/extracted_mag20x_patch256_fp/${FEAT}/feats_pt" \
      --results_dir "results/${EXP}" --split_dir "${SPLIT_DIR}" --split_names train,val,test,external \
      --task BRCA_INSTCVQC_officialRNA_survival --target_col os_survival_days \
      --model_histo_type PANTHER --model_histo_config PANTHER_default --n_fc_layers 0 --in_dim 1024 \
      --opt adamW --lr 0.0001 --lr_scheduler cosine --accum_steps 1 --wd 0.00001 --warmup_epochs 1 \
      --max_epochs 50 --train_bag_size -1 --batch_size 64 --seed "${SEED}" --num_workers 8 --em_iter 1 \
      --tau 1.0 --n_proto 16 --out_type allcat --loss_fn cox --nll_alpha 0.5 --n_label_bins 4 \
      --early_stopping 0 --ot_eps 1 --fix_proto --num_coattn_layers 1 --model_mm_type coattn \
      --append_embed random --histo_agg mean --net_indiv --load_proto --proto_path "${PROTO}" \
      > "../../.logs/compute_jobs/mmp_surv_s${SEED}_f${FOLD}.log" 2>&1 \
    && echo "MMP seed=${SEED} fold=${FOLD} 끝" || echo "!! MMP seed=${SEED} fold=${FOLD} 실패 — .logs/compute_jobs/mmp_surv_s${SEED}_f${FOLD}.log"
done; done
popd > /dev/null
python scripts/export_baseline_preds.py mmp --qc || true

# ---- 3) PORPOISE QC 2종 (HPC 묶음이 풀려 있을 때만) ----
# 묶음이 setup 뒤에 도착했으면 여기서 푼다.
if [ -f pathvit_bundle_porpoise.tar ] && [ ! -d data/porpoise_style_features ]; then
  tar -xf pathvit_bundle_porpoise.tar && echo "PORPOISE 묶음 풀림"
fi
if [ -d porpoise ] && [ -d data/porpoise_style_features ]; then
  ROOT_ABS="$(pwd)"
  pushd porpoise > /dev/null
  DATA_ROOT="${ROOT_ABS}/porpoise/data_root_true_resnet50"
  mkdir -p "${DATA_ROOT}/tcga_paad_20x_features"
  ln -sfn "${ROOT_ABS}/data/porpoise_style_features/tcga/pt_files" "${DATA_ROOT}/tcga_paad_20x_features/pt_files"
  # 변형: (results_dir, which_splits, 추가 인자, tcga csv, cptac csv, external 태그)
  for V in n110 official; do
    if [ "$V" = n110 ]; then RES=results_n110_mmf_qc; SPL=5foldcv_n110qc; EXTRA=""
      TCSV=datasets_csv/tcga_paad_all_clean.csv.zip; CCSV=datasets_csv/cptac_paad_n110_external.csv.zip; TAG=PORPOISE_N110_QC_MMF
    else RES=results_official_n110_mmf_qc; SPL=5foldcv_n110officialqc; EXTRA="--apply_mutsig"
      TCSV=datasets_csv_mutsig/tcga_paad_all_clean.csv.zip; CCSV=datasets_csv_mutsig/cptac_paad_external_cbio.csv.zip; TAG=PORPOISE_OFFICIAL_N110_QC_CBIO_MMF; fi
    for SEED in "${SEEDS[@]}"; do for FOLD in 0 1 2 3 4; do
      OUT_CSV="${ROOT_ABS}/.logs/external_preds/cptac_${TAG}_seed${SEED}_fold${FOLD}of5.csv"
      [ -f "$OUT_CSV" ] && { echo "PORPOISE ${V} seed=${SEED} fold=${FOLD} 이미 끝남"; continue; }
      LOG="${ROOT_ABS}/.logs/compute_jobs/porpoise_${V}_s${SEED}_f${FOLD}.log"
      if [ -z "$(find "$RES" -path "*_s${SEED}/split_latest_val_${FOLD}_results.pkl" 2>/dev/null | head -1)" ]; then
        python -u main.py --data_root_dir "${DATA_ROOT}" --which_splits "$SPL" --split_dir "tcga_paad_seed${SEED}" \
          --mode pathomic --model_type porpoise_mmf --bag_loss nll_surv --reg_type pathomic $EXTRA \
          --fusion bilinear --gate_path --gate_omic --skip --dropinput 0.10 \
          --k_start "${FOLD}" --k_end "$((FOLD + 1))" --results_dir "./${RES}" --seed "${SEED}" --overwrite > "$LOG" 2>&1 \
          || { echo "!! PORPOISE ${V} train seed=${SEED} fold=${FOLD} 실패 — $LOG"; continue; }
      fi
      mkdir -p "${ROOT_ABS}/.logs/external_preds"
      python -u eval_external.py --seed "${SEED}" --fold "${FOLD}" --results-dir "$RES" --which-splits "$SPL" \
        --tcga-csv "$TCSV" --tcga-split-dir "splits/${SPL}/tcga_paad_seed${SEED}" --cptac-csv "$CCSV" \
        --tcga-data-root "${DATA_ROOT}" --cptac-data-root "${ROOT_ABS}/data/porpoise_style_features/cptac" \
        --out-csv "$OUT_CSV" >> "$LOG" 2>&1 && echo "PORPOISE ${V} seed=${SEED} fold=${FOLD} 끝" \
        || echo "!! PORPOISE ${V} eval seed=${SEED} fold=${FOLD} 실패 — $LOG"
    done; done
  done
  popd > /dev/null
  python scripts/export_baseline_preds.py porpoise --qc || true
  python scripts/export_baseline_preds.py porpoise_official --qc || true
else
  echo "PORPOISE 묶음 없음 — 건너뜀 (scripts/compute/hpc_make_bundle.sh 참고)"
fi
echo "== 전부 끝. 결과는 .logs/{kfold_preds,external_preds}"
