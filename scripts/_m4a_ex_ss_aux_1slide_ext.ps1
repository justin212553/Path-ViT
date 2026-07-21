$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# findings_backlog.md 14번 항목: 레퍼런스는 환자당 대표 슬라이드 1장만 쓰는데(TCGA: diagnostic
# WSI, CPTAC: tumor series 중 최대 용량) 우리는 case당 존재하는 슬라이드를 전부 써왔다(TCGA 평균
# 2.52장/case, CPTAC 평균 3.22장/case). --one-slide-per-case로 이 격차를 좁혀 지금까지 external
# 최고 기록인 M4A_EX_SS_AUX 기준으로 재검증한다.
# external은 이 세션의 표준 관례대로 tcga train -> cptac test 단일 방향만 돈다(양방향 금지).

$LogDir = ".logs"
$Datasets = @("tcga")
$Seeds = @(42, 84, 126)
$GroupTs = "0721m4a1slide_ext"

$Total = $Datasets.Count * $Seeds.Count
$Run = 0

foreach ($dataset in $Datasets) {
    foreach ($seed in $Seeds) {
        $Run++
        Write-Host "=== [$Run/$Total] M4A_EX_SS_AUX_1SLIDE $dataset seed=$seed Start: $(Get-Date) ==="
        $log = Join-Path $LogDir "train_${dataset}_seed${seed}_M4A_EX_SS_AUX_1SLIDE_ext.log"
        python -u .\train.py --dataset $dataset --seed $seed --M4A --rna-genes literature_1500 `
            --patch-keep-frac 0.8 --rna-aux-weight 1.0 --one-slide-per-case `
            --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M4A_EX_SS_AUX_1SLIDE $dataset seed=$seed" }
        Write-Host "=== [$Run/$Total] M4A_EX_SS_AUX_1SLIDE $dataset seed=$seed Complete: $(Get-Date) ==="
    }
}
Write-Host "=== ALL M4A_EX_SS_AUX_1SLIDE EXTERNAL RUNS COMPLETE: $(Get-Date) ==="
