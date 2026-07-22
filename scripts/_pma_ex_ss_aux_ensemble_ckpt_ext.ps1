$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:\SSL_CERT_FILE -ErrorAction SilentlyContinue

# 오늘 여러 아키텍처 변형(TILEFUSION/REFDIM 등)이 같은 체크포인트 파일명을 계속 덮어써서
# 신뢰할 수 없는 상태 - 현재(최종 원복) 코드로 PMA_EX_SS_AUX를 3시드 깨끗하게 재학습해
# 시드 앙상블용 체크포인트를 새로 만든다. RNA는 이미 수정된 파이프라인 기준.

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$GroupTs = "0721pma_ensemble_ckpt_ext"

# 체크포인트 파일명에 seed가 안 들어가므로(survival_tcga_EX_SS_AUX_best_pma.pt, 시드 무관하게
# 항상 같은 경로) 시드마다 학습 직후 바로 seed-tagged 이름으로 복사해둔다 - 안 그러면 다음
# 시드가 이전 시드의 체크포인트를 덮어써서 마지막 시드 것만 남는다.
$CkptSrc = "models\checkpoint\survival_tcga_EX_SS_AUX_best_pma.pt"

foreach ($seed in $Seeds) {
    Write-Host "=== PMA_EX_SS_AUX(ensemble ckpt) seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${seed}_PMA_EX_SS_AUX_ensemble_ext.log"
    python -u .\train.py --dataset tcga --seed $seed --PMA --rna-genes literature_1500 `
        --patch-keep-frac 0.8 --rna-aux-weight 1.0 `
        --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: seed=$seed" }
    $CkptDst = "models\checkpoint\survival_tcga_EX_SS_AUX_best_pma_seed${seed}.pt"
    Copy-Item -Path $CkptSrc -Destination $CkptDst -Force
    Write-Host "  -> checkpoint 복사: $CkptDst"
    Write-Host "=== PMA_EX_SS_AUX(ensemble ckpt) seed=$seed Complete: $(Get-Date) ==="
}
Write-Host "=== ALL ENSEMBLE CHECKPOINT RUNS COMPLETE: $(Get-Date) ==="
