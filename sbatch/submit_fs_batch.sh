#!/bin/bash
# 2026-09-29: 리뷰 대응 재실행 배치 제출기 — fs_paad_array.sh / fs_brca_array.sh를 모델마다 한 번씩
# 제출한다(모델당 25 task = 5 seeds x 5 folds). 로그인 노드에서 실행:
#
#   bash sbatch/submit_fs_batch.sh            # PAAD 14개 + BRCA 9개 전부
#   bash sbatch/submit_fs_batch.sh paad       # PAAD만
#   bash sbatch/submit_fs_batch.sh brca M4 M4SA   # BRCA의 특정 모델만
#   bash sbatch/submit_fs_batch.sh paad M4FIXC M4SAFIXC   # 2026-10-04 고정 centroid 진단(목록 밖이라 명시 필요)
#
# 일부 task만 다시 돌릴 때(예: seed 84 fold 2 = index 2):
#   sbatch --array=2 --job-name=FS-PAAD-M4 --export=ALL,MODEL=M4 sbatch/fs_paad_array.sh

set -e
cd /pub/wonseukl/Path-ViT/

WHICH=${1:-all}
shift || true
ONLY=("$@")

PAAD_MODELS=(M1 M2 M3 M4 M5 M6 M7 M3SA M4SA M3NOCNV M4NOCNV M6NOCNV M7NOCNV M4NOMUT)
PAAD_EXTRA=(M4FIXC M4SAFIXC)
BRCA_MODELS=(M1 M2 M3 M4 M5 M6 M7 M3SA M4SA)
# WSI 없는 모델은 가벼운 자원으로
LIGHT=" M5 M6 M7 M6NOCNV M7NOCNV "

want() {  # 특정 모델만 지정했으면 그것만
  [ ${#ONLY[@]} -eq 0 ] && return 0
  for m in "${ONLY[@]}"; do [ "$m" = "$1" ] && return 0; done
  return 1
}

submit() {  # $1=cohort(paad|brca) $2=model
  local cohort=$1 model=$2 res=""
  if [[ "$LIGHT" == *" $model "* ]]; then
    if [ "$cohort" = paad ]; then res="--cpus-per-task=4 --mem=32G --time=8:00:00"
    else res="--cpus-per-task=4 --mem=32G --time=12:00:00"; fi
  fi
  local up=$(echo "$cohort" | tr a-z A-Z)
  sbatch $res --job-name="FS-${up}-${model}" --export=ALL,MODEL="$model" "sbatch/fs_${cohort}_array.sh"
}

for f in data/cohort_n110.txt data/rna_tcga_raw_log2.csv data/rna_cptac_raw_log2.csv data/rna_brca_raw_log2.csv; do
  [ -f "$f" ] || { echo "선행 파일 없음: $f — 로컬에서 옮겨올 것"; exit 1; }
done

if [ "$WHICH" = all ] || [ "$WHICH" = paad ]; then
  # 진단 모델(PAAD_EXTRA)은 기본 전체 제출에서 빠지고, 이름을 명시했을 때만 제출된다
  for m in "${PAAD_MODELS[@]}"; do want "$m" && submit paad "$m"; done
  if [ ${#ONLY[@]} -gt 0 ]; then
    for m in "${PAAD_EXTRA[@]}"; do want "$m" && submit paad "$m"; done
  fi
fi
if [ "$WHICH" = all ] || [ "$WHICH" = brca ]; then
  for m in "${BRCA_MODELS[@]}"; do want "$m" && submit brca "$m"; done
fi
