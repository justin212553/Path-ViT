$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

Write-Host "=== TCGA 다운로드 완료 대기 시작: $(Get-Date) ==="

# gdc-client 프로세스가 완전히 끝나고, 남은 .partial 파일이 없을 때까지 대기.
while ($true) {
    $proc = Get-Process -Name "gdc-client" -ErrorAction SilentlyContinue
    $partials = Get-ChildItem -Path ".\tcga_paad_wsi" -Recurse -Filter "*.partial" -ErrorAction SilentlyContinue
    if (-not $proc -and (-not $partials -or $partials.Count -eq 0)) {
        break
    }
    Start-Sleep -Seconds 30
}
Write-Host "=== TCGA 다운로드 완료 확인: $(Get-Date) ==="

$svsCount = (Get-ChildItem -Path ".\tcga_paad_wsi" -Recurse -Filter "*.svs" -ErrorAction SilentlyContinue).Count
Write-Host "다운로드된 .svs 파일 수: $svsCount"

# 1) 프로젝트 루트 tcga_paad_wsi/<uuid>/*.svs -> data/tcga_paad_wsi/<uuid>/ 로 이동(같은 드라이브라
#    rename 수준으로 빠름). data/tcga_paad_wsi는 이미 존재하는 빈 디렉토리.
Write-Host "=== data/tcga_paad_wsi로 이동 시작: $(Get-Date) ==="
Get-ChildItem -Path ".\tcga_paad_wsi" -Directory | ForEach-Object {
    $dest = Join-Path ".\data\tcga_paad_wsi" $_.Name
    Move-Item -Path $_.FullName -Destination $dest -Force
}
Write-Host "=== 이동 완료: $(Get-Date) ==="

# 2) UUID 서브디렉토리 평탄화 (data/tcga_paad_wsi 바로 아래로 .svs 모으고 빈 서브디렉토리 삭제)
Write-Host "=== flatten_tcga_paad_wsi 시작: $(Get-Date) ==="
python -m data.flatten_tcga_paad_wsi
Write-Host "=== flatten 완료: $(Get-Date) ==="

$flatCount = (Get-ChildItem -Path ".\data\tcga_paad_wsi" -Filter "*.svs").Count
Write-Host "data/tcga_paad_wsi 최종 .svs 파일 수: $flatCount"

# 3) 재타일링 + feature 추출 — findings_backlog.md 4번 항목. CPTAC 재타일링(data/patches_cptac_512)과
#    동일 스펙(target-mpp=0.5, tile-size=512, Lunit SwAV 사전학습 해상도에 맞춤).
Write-Host "=== TCGA 재타일링+feature 추출 시작: $(Get-Date) ==="
python -u -m data.preprocess --dataset tcga --target-mpp 0.5 --tile-size 512 --output-dir data/patches_tcga_512 2>&1 | Tee-Object -FilePath ".logs\tcga_preprocess_512.log"
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: TCGA preprocess" }
Write-Host "=== TCGA 재타일링+feature 추출 완료: $(Get-Date) ==="

Write-Host "=== ALL TCGA POSTPROCESS COMPLETE: $(Get-Date) ==="
