$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# _rnafix_revalidate_ext.ps1이 PMA seed84 도중 조용히 죽어서(에러 트레이스백 없음, CUDA/드라이버
# 추정) 이어서 재시작. PMA seed84/126 재시도 + PM4_EX_SS_AUX(레퍼런스 M4의 실제 결합 방식인
# post-hoc 게이트에 더 가까운 구조) 3시드까지 이어서 돈다.

$LogDir = ".logs"
$Seeds84126 = @(84, 126)
$Seeds = @(42, 84, 126)
$GroupTs = "0721rnafix_revalidate_ext"

$Total = $Seeds84126.Count + $Seeds.Count
$Run = 0

foreach ($seed in $Seeds84126) {
    $Run++
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_RNAFIX(resume) seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${seed}_PMA_EX_SS_AUX_RNAFIX_ext.log"
    python -u .\train.py --dataset tcga --seed $seed --PMA --rna-genes literature_1500 `
        --patch-keep-frac 0.8 --rna-aux-weight 1.0 `
        --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PMA_EX_SS_AUX_RNAFIX seed=$seed" }
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_RNAFIX(resume) seed=$seed Complete: $(Get-Date) ==="
}

foreach ($seed in $Seeds) {
    $Run++
    Write-Host "=== [$Run/$Total] PM4_EX_SS_AUX_RNAFIX seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${seed}_PM4_EX_SS_AUX_RNAFIX_ext.log"
    python -u .\train.py --dataset tcga --seed $seed --PM4 --rna-genes literature_1500 `
        --patch-keep-frac 0.8 --rna-aux-weight 1.0 `
        --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PM4_EX_SS_AUX_RNAFIX seed=$seed" }
    Write-Host "=== [$Run/$Total] PM4_EX_SS_AUX_RNAFIX seed=$seed Complete: $(Get-Date) ==="
}

Write-Host "=== ALL RESUME+PM4 RUNS COMPLETE: $(Get-Date) ==="
