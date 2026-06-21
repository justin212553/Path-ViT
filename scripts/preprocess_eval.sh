#!/bin/bash
#SBATCH --job-name=preprocess_eval
#SBATCH --partition=standard       # GPU 파티션 말고 일반 고성능 연산 파티션 선택
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # ★핵심: 파이썬 코드의 max_workers와 동일하게 16코어 징집
#SBATCH --mem=64G                  # 대용량 이미지 버퍼 핸들링을 위해 64GB 탑재
#SBATCH --time=12:00:00            # 충분히 넉넉하게 12시간 요새 방어
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/preprocess_progress_eval.log

cd /pub/wonseukl/Path-ViT/
source .venv/bin/activate

# 파이썬 실행
./.venv/bin/python -u ./data/preprocess_eval.py