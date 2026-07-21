$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# findings_backlog.md 14번 항목 절충안: --one-slide-per-case(대표 1장으로 축소)는 M4A/PMA 둘 다
# negative였다. 이번엔 확인된 정상 조직 슬라이드만 빼고(TCGA 평균 2.52->2.28, CPTAC 3.22->2.76)
# 나머지는 케이스당 그대로 두는 훨씬 덜 급진적인 --exclude-normal-slides로 PMA_EX_SS_AUX 기준
# 재검증. external은 tcga train -> cptac test 단일 방향, 3시드.

$LogDir = ".logs"
$Datasets = @("tcga")
$Seeds = @(42, 84, 126)
$GroupTs = "0721pma_nonormal_ext"

$Total = $Datasets.Count * $Seeds.Count
$Run = 0

foreach ($dataset in $Datasets) {
    foreach ($seed in $Seeds) {
        $Run++
        Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_NONORMAL $dataset seed=$seed Start: $(Get-Date) ==="
        $log = Join-Path $LogDir "train_${dataset}_seed${seed}_PMA_EX_SS_AUX_NONORMAL_ext.log"
        python -u .\train.py --dataset $dataset --seed $seed --PMA --rna-genes literature_1500 `
            --patch-keep-frac 0.8 --rna-aux-weight 1.0 --exclude-normal-slides `
            --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PMA_EX_SS_AUX_NONORMAL $dataset seed=$seed" }
        Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_NONORMAL $dataset seed=$seed Complete: $(Get-Date) ==="
    }
}
Write-Host "=== ALL PMA_EX_SS_AUX_NONORMAL EXTERNAL RUNS COMPLETE: $(Get-Date) ==="
