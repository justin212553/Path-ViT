$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# findings_backlog.md 14번 항목: --one-slide-per-case(TCGA DX 우선, CPTAC GDC 확인 tumor 우선)를
# external 최고 기록인 PMA_EX_SS_AUX(다성분 pooling+co-attention, external C=0.619)로 재검증한다 —
# M4A_EX_SS_AUX_1SLIDE 배치 다음에 이어서 실행.
# external은 이 세션의 표준 관례대로 tcga train -> cptac test 단일 방향만 돈다(양방향 금지).

$LogDir = ".logs"
$Datasets = @("tcga")
$Seeds = @(42, 84, 126)
$GroupTs = "0721pma1slide_ext"

$Total = $Datasets.Count * $Seeds.Count
$Run = 0

foreach ($dataset in $Datasets) {
    foreach ($seed in $Seeds) {
        $Run++
        Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_1SLIDE $dataset seed=$seed Start: $(Get-Date) ==="
        $log = Join-Path $LogDir "train_${dataset}_seed${seed}_PMA_EX_SS_AUX_1SLIDE_ext.log"
        python -u .\train.py --dataset $dataset --seed $seed --PMA --rna-genes literature_1500 `
            --patch-keep-frac 0.8 --rna-aux-weight 1.0 --one-slide-per-case `
            --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PMA_EX_SS_AUX_1SLIDE $dataset seed=$seed" }
        Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_1SLIDE $dataset seed=$seed Complete: $(Get-Date) ==="
    }
}
Write-Host "=== ALL PMA_EX_SS_AUX_1SLIDE EXTERNAL RUNS COMPLETE: $(Get-Date) ==="
