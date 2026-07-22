$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:\SSL_CERT_FILE -ErrorAction SilentlyContinue

# 나이스트롬 landmark가 고정된(좌표순 정렬) 패치 순서로 매 epoch 똑같이 그룹핑되는 문제를
# 순서 셔플로 검증(findings_backlog.md 참조) - PMA_EX_AUX(baseline, frac 자체 미사용) vs
# PMA_EX_SS_AUX(frac=1.0, shuffle-patches만 켬 - 순수 셔플 효과 격리). 둘 다 seed=42,
# --external, precomputed 모드라 가볍고 빠르다.

$LogDir = ".logs"
$GroupTs = "0722pma_shuffle_ext"

Write-Host "=== [1/2] PMA_EX_AUX(baseline, no shuffle) seed=42 Start: $(Get-Date) ==="
$log1 = Join-Path $LogDir "train_tcga_seed42_PMA_EX_AUX_baseline_ext.log"
python -u .\train.py --dataset tcga --seed 42 --PMA --rna-genes literature_1500 `
    --rna-aux-weight 1.0 --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log1
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PMA_EX_AUX baseline" }
Write-Host "=== [1/2] Complete: $(Get-Date) ==="

Write-Host "=== [2/2] PMA_EX_SS_AUX(frac=1.0, shuffle-patches) seed=42 Start: $(Get-Date) ==="
$log2 = Join-Path $LogDir "train_tcga_seed42_PMA_EX_SS_AUX_shuffle_ext.log"
python -u .\train.py --dataset tcga --seed 42 --PMA --rna-genes literature_1500 `
    --patch-keep-frac 1.0 --rna-aux-weight 1.0 --shuffle-patches `
    --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log2
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PMA_EX_SS_AUX shuffle" }
Write-Host "=== [2/2] Complete: $(Get-Date) ==="

Write-Host "=== ALL SHUFFLE-PATCHES RUNS COMPLETE: $(Get-Date) ==="
