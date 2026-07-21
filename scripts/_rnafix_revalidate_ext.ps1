$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# RNA-seq 전처리 버그 수정(tpm_unstranded, 로그 없음 -> log2(fpkm_uq_unstranded+1)) 이후
# 대표 모델 3개를 --external(tcga->cptac, 3시드)로 재검증한다. findings_backlog.md 최상위 발견 항목.

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$GroupTs = "0721rnafix_revalidate_ext"

$Total = 3 * $Seeds.Count
$Run = 0

foreach ($seed in $Seeds) {
    $Run++
    Write-Host "=== [$Run/$Total] M7_EX_RNAFIX seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_light_tcga_seed${seed}_M7_EX_RNAFIX_ext.log"
    python -u .\train_light.py --dataset tcga --seed $seed --M7 --rna-genes literature_1500 `
        --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M7_EX_RNAFIX seed=$seed" }
    Write-Host "=== [$Run/$Total] M7_EX_RNAFIX seed=$seed Complete: $(Get-Date) ==="
}

foreach ($seed in $Seeds) {
    $Run++
    Write-Host "=== [$Run/$Total] M4A_EX_SS_AUX_RNAFIX seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${seed}_M4A_EX_SS_AUX_RNAFIX_ext.log"
    python -u .\train.py --dataset tcga --seed $seed --M4A --rna-genes literature_1500 `
        --patch-keep-frac 0.8 --rna-aux-weight 1.0 `
        --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M4A_EX_SS_AUX_RNAFIX seed=$seed" }
    Write-Host "=== [$Run/$Total] M4A_EX_SS_AUX_RNAFIX seed=$seed Complete: $(Get-Date) ==="
}

foreach ($seed in $Seeds) {
    $Run++
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_RNAFIX seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${seed}_PMA_EX_SS_AUX_RNAFIX_ext.log"
    python -u .\train.py --dataset tcga --seed $seed --PMA --rna-genes literature_1500 `
        --patch-keep-frac 0.8 --rna-aux-weight 1.0 `
        --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PMA_EX_SS_AUX_RNAFIX seed=$seed" }
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_RNAFIX seed=$seed Complete: $(Get-Date) ==="
}

Write-Host "=== ALL RNAFIX REVALIDATION RUNS COMPLETE: $(Get-Date) ==="
