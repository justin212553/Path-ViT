import os

# ── [보스의 환경에 맞게 경로 설정] ─────────────────────────────────────────
MANIFEST_PATH = "./scripts/gdc_manifest.txt"  # GDC 매니페스트 파일 경로
DOWNLOAD_DIR  = "./data/tcga_paad_wsi" # WSI가 저장 중인 경로
# ─────────────────────────────────────────────────────────────────────────────

# 1. 매니페스트 파일에서 실제 다운로드 받아야 할 파일 UUID 목록 파싱
target_uuids = set()
with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
    header = f.readline() # 첫 줄(id, filename 등 헤더) 건너뛰기
    for line in f:
        if line.strip():
            # 매니페스트의 첫 번째 항목이 File UUID입니다.
            file_uuid = line.split('\t')[0].strip()
            target_uuids.add(file_uuid)

# 2. 현재 디렉토리에 실제로 생성된 폴더(UUID) 목록 확인
downloaded_items = os.listdir(DOWNLOAD_DIR)
current_uuids = set()

for item in downloaded_items:
    item_path = os.path.join(DOWNLOAD_DIR, item)
    # 폴더 구조이고, 내부에 실제 .svs 파일이 완전히 안착했는지 체크
    if os.path.isdir(item_path):
        has_svs = any(f.endswith('.svs') for f in os.listdir(item_path))
        if has_svs:
            current_uuids.add(item)

# 3. 데이터 대조 및 메트릭 산출
total_count = len(target_uuids)
success_count = len(target_uuids.intersection(current_uuids))
missing_uuids = target_uuids - current_uuids
progress_percent = (success_count / total_count) * 100 if total_count > 0 else 0

# 4. 결과 출력
print("\n=======================================================")
print(f"📡 [TCGA-PAAD 코호트 WSI 다운로드 진척도 보고 리포트]")
print("=======================================================")
print(f"📊 총 목표 파일 개수   : {total_count} 개")
print(f"✅ 다운로드 성공 개수  : {success_count} 개")
print(f"❌ 누락/진행중 개수    : {len(missing_uuids)} 개")
print(f"📈 최종 실질 진척도    : {progress_percent:.2f} %")
print("=======================================================\n")

if missing_uuids:
    print("📋 [누락된 파일 GDC UUID 리스트 (아직 안 받아졌거나 깨진 파일)]")
    for i, muuid in enumerate(sorted(missing_uuids), 1):
        print(f"  {i}. {muuid}")
else:
    print("🎉 매니페스트의 모든 WSI 파일이 무결하게 전원 다운로드되었습니다!")