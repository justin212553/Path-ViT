$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$LogDir = ".logs"
$Seeds = @(42, 84, 126)

# 1) M7_EX risk head Dropout(0.4)만 추가(은닉층 없음), RNA 수정 후 재검증
$GroupTs = "0721m7_riskhead_dropout_only_rnafix_ext"
foreach ($seed in $Seeds) {
    Write-Host "=== M7_EX_RISKHEADDROPONLY seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_light_tcga_seed${seed}_M7_EX_RISKHEADDROPONLY_ext.log"
    python -u .\train_light.py --dataset tcga --seed $seed --M7 --rna-genes literature_1500 `
        --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M7_EX_RISKHEADDROPONLY seed=$seed" }
    Write-Host "=== M7_EX_RISKHEADDROPONLY seed=$seed Complete: $(Get-Date) ==="
}

# 2) PMA: RNA=256/Clinical=16(레퍼런스 사양) + tile-fusion GELU/Dropout(이미 적용됨)
$GroupTs2 = "0721pma_refdim_ext"
foreach ($seed in $Seeds) {
    Write-Host "=== PMA_EX_SS_AUX_REFDIM seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${seed}_PMA_EX_SS_AUX_REFDIM_ext.log"
    python -u .\train.py --dataset tcga --seed $seed --PMA --rna-genes literature_1500 `
        --patch-keep-frac 0.8 --rna-aux-weight 1.0 `
        --external --group-ts $GroupTs2 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PMA_EX_SS_AUX_REFDIM seed=$seed" }
    Write-Host "=== PMA_EX_SS_AUX_REFDIM seed=$seed Complete: $(Get-Date) ==="
}

Write-Host "=== ALL M7DROP+PMAREFDIM RUNS COMPLETE: $(Get-Date) ==="
