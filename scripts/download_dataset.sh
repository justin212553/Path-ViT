cat << 'EOF' > download_dataset.sh
#!/bin/bash
#SBATCH --job-name=camelyon_download  # 작업 이름
#SBATCH --partition=standard          # 대용량 다운로드를 위한 표준 파티션
#SBATCH --nodes=1                     # 단일 노드 점유
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # 병렬 다운로드를 위한 CPU 코어 4개 탑재
#SBATCH --mem=16G                     # 안정적인 버퍼링을 위해 RAM 16G 확보
#SBATCH --time=48:00:00               # 500GB 용량을 고려해 제한시간을 48시간으로 넉넉히 설정
#SBATCH --output=download_progress.log # 실시간 tqdm 다운로드 게이지가 기록될 로그 파일

# 1. 우리의 대용량 메인 기지로 진입
cd /pub/wonseukl/Path-ViT/utils

# 2. 오타 없이 정렬된 가상환경 깨우기
conda activate ./.venv

# 3. 데이터 다운로드 엔진 가동
python dataset_download.py
EOF