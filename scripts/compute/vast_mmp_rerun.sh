#!/bin/bash
# 2026-10-23: MMP survival만 다시 돌린다. 공식 training/trainer.py 끝의 writer.close()가 정의 안 된 writer를
# 불러 NameError로 죽어(학습·평가는 끝났지만 결과 pkl 저장 전) 25회가 전부 실패했다. trainer.py를 고친 뒤
# (vast_setup.sh가 하는 것과 같은 sed) 저장된 fold별 prototype을 그대로 써서 survival만 다시 돈다.
# 실행 중인 vast_run_all.sh가 MMP 구간을 벗어날 때까지 기다렸다가 시작하므로 같은 fold를 동시에 돌지 않는다.
#   WAIT_FROM_LINE=<지금 vast_all.log 줄 수> nohup bash scripts/compute/vast_mmp_rerun.sh >> .logs/vast_mmp_rerun.log 2>&1 &
set -e
cd "$(dirname "$0")/../.."
SEEDS=(84 126 42 168 210)
START=${WAIT_FROM_LINE:-1}
while pgrep -f "training.main_(prototype|survival)" > /dev/null \
      || ! tail -n +"$START" .logs/vast_all.log | grep -qE "PORPOISE|== 전부 끝"; do
  sleep 60
done
echo "[$(date +%H:%M)] vast_run_all.sh가 MMP 구간을 지남 — survival 재실행 시작"

FEAT='extracted-vit_large_patch16_224.dinov2.uni_mass100k'
MMP_DATAROOT="$(pwd)/data/mmp_brca_dataroot"
pushd mmp/src > /dev/null
grep -q "^    writer.close()$" training/trainer.py && { echo "!! trainer.py writer.close() 미수정 — 중단"; exit 1; }
for SEED in "${SEEDS[@]}"; do for FOLD in 0 1 2 3 4; do
  SPLIT_DIR="survival/TCGA_BRCA_k${FOLD}_seed${SEED}_instcvqc"
  PROTO="splits/${SPLIT_DIR}/prototypes/prototypes_c16_${FEAT}_faiss_num_1.0e+05.pkl"
  EXP="BRCA_INSTCVQC_officialRNA_seed${SEED}_k${FOLD}::PANTHER_default::${FEAT}"
  if [ -n "$(find "results/${EXP}" -name test_results.pkl 2>/dev/null | head -1)" ]; then
    echo "MMP seed=${SEED} fold=${FOLD} 이미 끝남"; continue; fi
  [ -f "$PROTO" ] || { echo "!! MMP seed=${SEED} fold=${FOLD} prototype 없음 — 건너뜀"; continue; }
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
echo "[$(date +%H:%M)] == MMP 재실행 끝"
