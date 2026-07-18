$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# findings_backlog.md 11번 항목: AUX2(stage 보조과제)를 얹으면 external C가 0.61~0.62 -> 0.49~0.51로
# 급락했다. STG(clinical에 진짜 T/N/M/grade 주입)만 단독으로 켜서(--stage-aux-weight는 기본 0=끔),
# AUX2 간섭 없이 "진짜 병기 주입" 자체의 순수 효과를 M4A_EX_SS_AUX/PMA_EX_SS_AUX 기준선과 비교한다.

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$Datasets = @("tcga", "cptac")
$GroupTs = "0717stgonly"
$KeepFrac = "0.8"
$AuxWeight = "1.0"

$Models = @("--M4A", "--PMA")
$Tags   = @("M4A", "PMA")

$Total = $Models.Count * $Datasets.Count * $Seeds.Count
$Run = 0

for ($i = 0; $i -lt $Models.Count; $i++) {
    $flag = $Models[$i]; $tag = $Tags[$i]
    foreach ($ds in $Datasets) {
        foreach ($seed in $Seeds) {
            $Run++
            Write-Host "=== [$Run/$Total] ${tag}_SS_AUX_STG ds=$ds seed=$seed Start: $(Get-Date) ==="
            $log = Join-Path $LogDir "train_${ds}_seed${seed}_${tag}_EX_SS_AUX_STG_ext.log"
            python -u .\train.py --dataset $ds --seed $seed $flag --rna-genes literature_1500 `
                --patch-keep-frac $KeepFrac --rna-aux-weight $AuxWeight --clinical-staging `
                --external --group-ts $GroupTs | Tee-Object -FilePath $log
            if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: ${tag}_SS_AUX_STG ds=$ds seed=$seed" }
            Write-Host "=== [$Run/$Total] ${tag}_SS_AUX_STG ds=$ds seed=$seed Complete: $(Get-Date) ==="
        }
    }
}
Write-Host "=== ALL STG-ONLY EXTERNAL RUNS COMPLETE: $(Get-Date) ==="
