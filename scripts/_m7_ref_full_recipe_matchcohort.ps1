$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# M7 레퍼런스 완전 재현 + 케이스 코호트까지 레퍼런스 기준(24개월 시점 생존 확정 + WSI 보유)에
# 맞춤(--match-reference-cohort, data/reference_cohort.py). 311->225 cases로 줄어든다.

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$GroupTs = "0719m7fullrecipe_matchcohort"
$Lr = "5e-5"
$Wd = "1e-3"
$Epochs = 100
$Patience = 20

$Total = $Seeds.Count
$Run = 0

foreach ($seed in $Seeds) {
    $Run++
    Write-Host "=== [$Run/$Total] M7_EX_fullrecipe_matchcohort both seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_light_both_seed${seed}_M7_EX_fullrecipe_matchcohort_both.log"
    python -u .\train_light.py --dataset both --seed $seed --M7 --rna-genes literature_1500 `
        --lr $Lr --weight-decay $Wd --epochs $Epochs --patience $Patience --match-reference-cohort --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M7_EX_fullrecipe_matchcohort both seed=$seed" }
    Write-Host "=== [$Run/$Total] M7_EX_fullrecipe_matchcohort both seed=$seed Complete: $(Get-Date) ==="
}
Write-Host "=== ALL M7_EX FULLRECIPE MATCHCOHORT RUNS COMPLETE: $(Get-Date) ==="
